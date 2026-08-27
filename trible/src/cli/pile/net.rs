//! CLI commands for distributed pile sync.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Parser, ValueEnum};
use ed25519_dalek::SigningKey;
use iroh_base::{EndpointAddr, EndpointId};
use iroh_tickets::endpoint::EndpointTicket;

use triblespace_net::inventory::{InventoryComponent, InventoryManifest, InventorySnapshot};
use triblespace_net::peer::{
    BlobReconcileMode, Peer, PeerConfig, ReconcileDirection, ReconcileQos,
};
use triblespace_net::protocol::PILE_SYNC_ALPN;

use triblespace_core::repo::pile::Pile;
use triblespace_core::repo::{StoreRevision, StoreScope};

fn open_pile(path: &PathBuf) -> Result<Pile> {
    crate::cli::pile::open_refreshed(path)
}

/// Parse `--peers` as canonical iroh endpoint tickets or bare endpoint ids.
///
/// Tickets preserve explicit direct/fabric addresses. Bare ids deliberately
/// delegate address selection to iroh discovery. A malformed peer is an error:
/// silently dropping a mistyped fabric route can otherwise make a deployment
/// fall back to a relay or management network while appearing configured.
fn parse_peers(strs: &[String]) -> Result<Vec<EndpointAddr>> {
    strs.iter()
        .map(|s| {
            if let Ok(ticket) = s.parse::<EndpointTicket>() {
                return Ok(ticket.into());
            }
            let pk = s.parse::<iroh_base::PublicKey>().map_err(|_| {
                anyhow!(
                    "invalid peer {s:?}: expected an iroh endpoint ticket or 64-char endpoint id"
                )
            })?;
            Ok(EndpointAddr::from(EndpointId::from(pk)))
        })
        .collect()
}

fn load_existing_key(path: Option<PathBuf>, pile_path: &PathBuf) -> Result<SigningKey> {
    let path = triblespace_core::signing_key_file::resolve_path(path.as_deref(), pile_path);
    triblespace_core::signing_key_file::load_existing(&path).map_err(Into::into)
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum DirectionArg {
    Bidirectional,
    ReadOnly,
    WriteOnly,
}

impl From<DirectionArg> for ReconcileDirection {
    fn from(direction: DirectionArg) -> Self {
        match direction {
            DirectionArg::Bidirectional => Self::Bidirectional,
            DirectionArg::ReadOnly => Self::ReadOnly,
            DirectionArg::WriteOnly => Self::WriteOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum BlobsArg {
    Demand,
    Mirror,
}

impl From<BlobsArg> for BlobReconcileMode {
    fn from(blobs: BlobsArg) -> Self {
        match blobs {
            BlobsArg::Demand => Self::Demand,
            BlobsArg::Mirror => Self::Mirror,
        }
    }
}

// ── CLI ──────────────────────────────────────────────────────────────

#[derive(Parser)]
pub enum Command {
    /// Show this node's network identity.
    Identity {
        /// Path to the node's signing key.
        #[arg(long)]
        key: Option<PathBuf>,
    },
    /// Print the exact canonical inventory sampled from one bound pile.
    Inventory { pile: PathBuf },
    /// Resolve and show the exact CONNECT and SYNC_TEAM proofs this node would present.
    Status {
        /// Pile containing the native proof and its exact claim closure.
        pile: PathBuf,
        /// Path to the node's signing key.
        #[arg(long)]
        key: Option<PathBuf>,
        /// Team root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact CONNECT proof id (BLAKE3 of canonical proof bytes).
        #[arg(long)]
        connect_proof: String,
        /// Exact SYNC_TEAM proof id (BLAKE3 of canonical proof bytes).
        #[arg(long)]
        sync_proof: String,
    },
    /// Reconcile one authorized team inventory with peers.
    Sync {
        pile: PathBuf,
        #[arg(long, value_delimiter = ',')]
        peers: Vec<String>,
        #[arg(long)]
        key: Option<PathBuf>,
        /// Team root public key (32-byte hex).
        #[arg(long)]
        team_root: String,
        /// Exact CONNECT proof id (BLAKE3 of canonical proof bytes).
        #[arg(long)]
        connect_proof: String,
        /// Exact SYNC_TEAM proof id (BLAKE3 of canonical proof bytes).
        #[arg(long)]
        sync_proof: String,
        /// Whether to pull inventories, serve them, or do both.
        #[arg(long, value_enum, default_value = "bidirectional")]
        direction: DirectionArg,
        /// Fetch blobs only for durable WANTs, or mirror the complete blob inventory.
        #[arg(long, value_enum, default_value = "demand")]
        blobs: BlobsArg,
        /// Stop after at most N seconds. Without this flag (and without
        /// `--quiescent-for`), sync runs until interrupted with Ctrl-C —
        /// "done" isn't a knowable state in a team swarm (two-generals).
        #[arg(long, value_name = "SECS")]
        duration: Option<u64>,
        /// Stop after N seconds without any admitted inventory event or
        /// durable WANT being serviced. Best-effort "we appear to have
        /// caught up" signal — useful for bounded sync in scripts where
        /// you accept the two-generals caveat. Wants that stay pending
        /// (nobody reachable holds them) do NOT hold off quiescence —
        /// a pending want is normal, not unfinished work.
        #[arg(long, value_name = "SECS")]
        quiescent_for: Option<u64>,
    },
}

pub fn run(cmd: Command) -> Result<()> {
    match cmd {
        Command::Identity { key } => run_identity(key),
        Command::Inventory { pile } => run_inventory(pile),
        Command::Status {
            pile,
            key,
            team_root,
            connect_proof,
            sync_proof,
        } => run_status(pile, key, team_root, connect_proof, sync_proof),
        Command::Sync {
            pile,
            peers,
            key,
            team_root,
            connect_proof,
            sync_proof,
            direction,
            blobs,
            duration,
            quiescent_for,
        } => run_sync(
            pile,
            peers,
            key,
            team_root,
            connect_proof,
            sync_proof,
            ReconcileQos {
                direction: direction.into(),
                blobs: blobs.into(),
            },
            duration,
            quiescent_for,
        ),
    }
}

// ── Inventory evidence ──────────────────────────────────────────────

fn stable_inventory_manifest(
    pile: &mut Pile,
) -> Result<(ed25519_dalek::VerifyingKey, InventoryManifest)> {
    let scope_before = pile
        .store_scope()?
        .ok_or_else(|| anyhow!("pile has no bound team scope"))?;
    let revision_before = pile.store_revision()?;
    let snapshot = InventorySnapshot::from_store(pile, scope_before)?;
    let manifest = snapshot.manifest().clone();
    drop(snapshot);
    let revision_after = pile.store_revision()?;
    let scope_after = pile.store_scope()?;

    ensure_inventory_sample_unchanged(
        &revision_before,
        &revision_after,
        scope_before,
        scope_after,
    )?;
    Ok((scope_before, manifest))
}

fn ensure_inventory_sample_unchanged<R: Eq>(
    revision_before: &R,
    revision_after: &R,
    scope_before: ed25519_dalek::VerifyingKey,
    scope_after: Option<ed25519_dalek::VerifyingKey>,
) -> Result<()> {
    if revision_before != revision_after || scope_after != Some(scope_before) {
        return Err(anyhow!(
            "sync-visible inventory changed while it was sampled; retry"
        ));
    }
    Ok(())
}

fn component_label(component: InventoryComponent) -> &'static str {
    match component {
        InventoryComponent::Peer => "peer",
        InventoryComponent::CollectionRecord => "collection_record",
        InventoryComponent::CapabilityProof => "capability_proof",
        InventoryComponent::Blob => "blob",
    }
}

fn format_inventory_manifest(
    team: ed25519_dalek::VerifyingKey,
    manifest: &InventoryManifest,
) -> String {
    use std::fmt::Write as _;

    let mut output = String::new();
    writeln!(
        output,
        "protocol:            {}",
        std::str::from_utf8(PILE_SYNC_ALPN).expect("pile sync ALPN is ASCII")
    )
    .expect("writing to a String cannot fail");
    writeln!(
        output,
        "team_root:           {}",
        hex::encode(team.to_bytes())
    )
    .expect("writing to a String cannot fail");
    writeln!(
        output,
        "generation:          {}",
        hex::encode(manifest.generation().into_bytes())
    )
    .expect("writing to a String cannot fail");
    for component in InventoryComponent::ALL {
        let entry = manifest.component(component);
        let label = component_label(component);
        writeln!(output, "{label}_count: {count}", count = entry.leaf_count())
            .expect("writing to a String cannot fail");
        writeln!(
            output,
            "{label}_root:  {}",
            entry
                .root()
                .map(hex::encode)
                .unwrap_or_else(|| "empty".into())
        )
        .expect("writing to a String cannot fail");
    }
    output
}

fn print_inventory_manifest(team: ed25519_dalek::VerifyingKey, manifest: &InventoryManifest) {
    print!("{}", format_inventory_manifest(team, manifest));
}

fn run_inventory(pile_path: PathBuf) -> Result<()> {
    let mut pile = open_pile(&pile_path)?;
    let sampled = stable_inventory_manifest(&mut pile);
    let closed = pile
        .close()
        .map_err(|error| anyhow!("close pile: {error:?}"));
    let (team, manifest) = sampled?;
    closed?;
    print_inventory_manifest(team, &manifest);
    Ok(())
}

// ── Identity ─────────────────────────────────────────────────────────

fn run_identity(sk: Option<PathBuf>) -> Result<()> {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let default_anchor = cwd.join("identity.pile");
    let path = triblespace_core::signing_key_file::resolve_path(sk.as_deref(), &default_anchor);
    let key = triblespace_core::signing_key_file::init(&path)?;
    let public = triblespace_net::identity::iroh_secret(&key).public();
    println!("node: {public}");
    Ok(())
}

// ── Status ───────────────────────────────────────────────────────────

fn run_status(
    pile_path: PathBuf,
    key_path: Option<PathBuf>,
    team_root_text: String,
    connect_proof_text: String,
    sync_proof_text: String,
) -> Result<()> {
    let key = load_existing_key(key_path, &pile_path)?;
    let public = triblespace_net::identity::iroh_secret(&key).public();
    let team_root = crate::cli::team::parse_team_root(&team_root_text)?;
    let connect_proof_id = crate::cli::team::parse_proof_id(&connect_proof_text)?;
    let sync_proof_id = crate::cli::team::parse_proof_id(&sync_proof_text)?;
    let mut pile = open_pile(&pile_path)?;
    let connect_bundle = match crate::cli::team::resolve_connect_bundle(
        &mut pile,
        team_root,
        connect_proof_id,
        key.verifying_key(),
    ) {
        Ok(proof) => proof,
        Err(error) => {
            let _ = pile.close();
            return Err(error);
        }
    };
    let sync_bundle = match crate::cli::team::resolve_sync_bundle(
        &mut pile,
        team_root,
        sync_proof_id,
        key.verifying_key(),
    ) {
        Ok(proof) => proof,
        Err(error) => {
            let _ = pile.close();
            return Err(error);
        }
    };
    pile.close()
        .map_err(|error| anyhow!("close pile: {error:?}"))?;

    println!("node:                {public}");
    println!("team_root:           {}", hex::encode(team_root.to_bytes()));
    println!(
        "connect_proof_id:     {}",
        hex::encode(connect_proof_id.raw)
    );
    println!(
        "connect_proof_steps:  {}",
        connect_bundle.proof().step_count()
    );
    println!("sync_proof_id:        {}", hex::encode(sync_proof_id.raw));
    println!("sync_proof_steps:     {}", sync_bundle.proof().step_count());
    println!("authorization:       CONNECT + SYNC_TEAM accepted");
    Ok(())
}

// ── Sync ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn run_sync(
    pile_path: PathBuf,
    peer_strs: Vec<String>,
    key_path: Option<PathBuf>,
    team_root_text: String,
    connect_proof_text: String,
    sync_proof_text: String,
    qos: ReconcileQos,
    duration: Option<u64>,
    quiescent_for: Option<u64>,
) -> Result<()> {
    let key = load_existing_key(key_path, &pile_path)?;
    // Endpoint tickets retain caller-selected direct/fabric routes; bare
    // endpoint ids intentionally use iroh's discovery layer.
    let peers = parse_peers(&peer_strs)?;

    // One pile handle wrapped directly in a Peer. Inventory reconciliation is
    // a monotone union, so no mutable branch mirror is involved.
    let mut pile = open_pile(&pile_path)?;
    let team_root = crate::cli::team::parse_team_root(&team_root_text)?;
    let connect_proof_id = crate::cli::team::parse_proof_id(&connect_proof_text)?;
    let sync_proof_id = crate::cli::team::parse_proof_id(&sync_proof_text)?;
    let connect_proof = match crate::cli::team::resolve_connect_bundle(
        &mut pile,
        team_root,
        connect_proof_id,
        key.verifying_key(),
    ) {
        Ok(proof) => proof,
        Err(error) => {
            let _ = pile.close();
            return Err(error);
        }
    };
    let sync_proof = match crate::cli::team::resolve_sync_bundle(
        &mut pile,
        team_root,
        sync_proof_id,
        key.verifying_key(),
    ) {
        Ok(proof) => proof,
        Err(error) => {
            let _ = pile.close();
            return Err(error);
        }
    };
    pile.bind_store_scope(team_root)?;
    let mut peer = Peer::new(
        pile,
        key.clone(),
        PeerConfig {
            peers,
            team: team_root,
            connect_proof,
            sync_proof,
            qos,
        },
    )?;
    eprintln!("node: {}", peer.id());
    eprintln!("team_root: {}", hex::encode(team_root.to_bytes()));
    let dir_label = match qos.direction {
        ReconcileDirection::Bidirectional => "bidirectional",
        ReconcileDirection::ReadOnly => "read-only (no serve)",
        ReconcileDirection::WriteOnly => "write-only (no fetch)",
    };
    eprintln!("direction: {dir_label}");
    let blob_label = match qos.blobs {
        BlobReconcileMode::Demand => "demand (durable WANTs only)",
        BlobReconcileMode::Mirror => "mirror (complete authorized blob inventory)",
    };
    eprintln!("blobs: {blob_label}");
    if let Some(d) = duration {
        eprintln!("stop after: {d}s");
    }
    if let Some(q) = quiescent_for {
        eprintln!("quiescent stop: {q}s without events");
    }
    // Demand reconciliation is intrinsic: readable nodes always service
    // durable WANTs. Write-only suppresses every fetch by definition.
    let services_wants = qos.direction != ReconcileDirection::WriteOnly;
    eprintln!("live sync active. (Ctrl-C to stop)\n");

    let started = std::time::Instant::now();
    let duration_limit = duration.map(std::time::Duration::from_secs);
    let quiescent_limit = quiescent_for.map(std::time::Duration::from_secs);

    // Want-reconcile state. The Reconciler (triblespace-net) owns the
    // per-want retry bookkeeping (exponential backoff, capped at 60s);
    // the wants themselves live durably in the pile's WantStore. The
    // tick is async (the swarm fetch awaits the host), so we drive it
    // on a small current-thread runtime; dial and operation deadlines
    // use tokio timers and therefore need a runtime context.
    let mut reconciler = triblespace_net::reconcile::Reconciler::new();
    let reconcile_every = std::time::Duration::from_secs(1);
    let mut next_reconcile = std::time::Instant::now();
    let mut wants_fulfilled_total: u64 = 0;
    let mut wants_pending: usize = 0;
    let mut last_pending_logged: Option<usize> = None;
    // Most recent time a want was actually serviced — lazy progress
    // counts as activity for --quiescent-for (pending wants do NOT:
    // an unsatisfiable want is steady state, not unfinished work).
    let mut last_want_progress = std::time::Instant::now();
    let reconcile_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow!("reconcile runtime: {e}"))?;

    loop {
        // Bounded run-time. The host_loop also does periodic re-broadcasts
        // (30s) from the current snapshot, so the CLI does not need to drive
        // another publication tick.
        if let Some(limit) = duration_limit {
            if started.elapsed() >= limit {
                eprintln!(
                    "\nreached --duration limit ({}s); stopping",
                    limit.as_secs()
                );
                break;
            }
        }
        // Quiescence stop: no NetEvent absorbed AND no want serviced for
        // the configured window. The two-generals caveat applies —
        // "looks idle" isn't "synced" — but for bounded sync in scripts
        // the caller has explicitly opted into this trade-off. Wants
        // still pending don't hold quiescence off: a want nobody
        // reachable holds may stay pending forever, and that's its
        // normal state (it survives in the pile for the next run).
        if let Some(limit) = quiescent_limit {
            if peer.last_event_at().elapsed() >= limit && last_want_progress.elapsed() >= limit {
                eprintln!("\nquiescent for {}s; stopping", limit.as_secs());
                break;
            }
        }

        // Drain authenticated inventory and install any externally appended
        // local evidence through one durability barrier.
        peer.refresh();

        // Want-reconcile tick: a want IS a durable want-marker —
        // "I would like this blob; fetch it if absent; evictable."
        // Each pass re-reads the pile (want records appended by
        // OTHER processes since the last pass become visible), diffs
        // the want set against the blobs present, and swarm-fetches
        // the missing ones. Their existing wants become cache-retention
        // interest after the bytes land. Failed fetches retry with
        // per-want exponential backoff
        // inside the Reconciler; a want nobody serves stays pending —
        // normal, never an error, never dropped. Strong pins/branches
        // are untouched.
        if services_wants && next_reconcile <= std::time::Instant::now() {
            let stats = reconcile_rt.block_on(reconciler.tick(&mut peer));
            next_reconcile = std::time::Instant::now() + reconcile_every;
            wants_fulfilled_total += stats.fulfilled as u64;
            wants_pending = stats.pending;
            if stats.fulfilled > 0 {
                last_want_progress = std::time::Instant::now();
            }
            // Trace on change (a want serviced, or the pending count
            // moved), not per tick — pending wants are steady state.
            if stats.fulfilled > 0 || last_pending_logged != Some(stats.pending) {
                eprintln!(
                    "  wants: {} seen, {} fulfilled this pass ({} total), {} pending",
                    stats.wants, stats.fulfilled, wants_fulfilled_total, stats.pending,
                );
                last_pending_logged = Some(stats.pending);
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    if services_wants {
        eprintln!(
            "wants: {wants_fulfilled_total} fulfilled this run; {wants_pending} still pending \
             (pending is normal — the wants stay in the pile's WantStore \
             and are serviced whenever a holder becomes reachable)"
        );
    }
    peer.into_store()
        .close()
        .map_err(|error| anyhow!("close pile: {error}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh_base::{SecretKey, TransportAddr};
    use tempfile::NamedTempFile;
    use triblespace_core::blob::{encodings::UnknownBlob, Bytes};
    use triblespace_core::repo::offer::{ArtifactHandle, ArtifactOfferStore};
    use triblespace_core::repo::peer::{PeerEvidence, PeerStore};
    use triblespace_core::repo::BlobStorePut;

    #[test]
    fn peers_accept_bare_ids_and_endpoint_tickets() {
        let secret = SecretKey::from_bytes(&[7; 32]);
        let id = EndpointId::from(secret.public());
        let direct =
            EndpointAddr::from_parts(id, [TransportAddr::Ip("10.55.0.2:49152".parse().unwrap())]);
        let ticket = EndpointTicket::new(direct.clone()).to_string();

        assert_eq!(parse_peers(&[id.to_string()]).unwrap(), vec![id.into()]);
        assert_eq!(parse_peers(&[ticket]).unwrap(), vec![direct]);
        assert!(parse_peers(&["not-a-peer".to_owned()]).is_err());
    }

    #[test]
    fn inventory_evidence_requires_a_bound_scope_without_mutating_the_pile() {
        let file = NamedTempFile::new().unwrap();
        let mut pile = Pile::open(file.path()).unwrap();
        let length_before = std::fs::metadata(file.path()).unwrap().len();

        let error = stable_inventory_manifest(&mut pile).unwrap_err();

        assert!(error.to_string().contains("no bound team scope"));
        assert_eq!(std::fs::metadata(file.path()).unwrap().len(), length_before);
        pile.close().unwrap();
    }

    #[test]
    fn inventory_evidence_derives_scope_and_ignores_local_offer_intent() {
        let file = NamedTempFile::new().unwrap();
        let mut pile = Pile::open(file.path()).unwrap();
        let team = SigningKey::from_bytes(&[0x51; 32]).verifying_key();
        pile.bind_store_scope(team).unwrap();
        let (observed_team, before) = stable_inventory_manifest(&mut pile).unwrap();

        pile.offer(ArtifactHandle::new([0xA7; 32])).unwrap();
        let (_, after) = stable_inventory_manifest(&mut pile).unwrap();

        assert_eq!(observed_team, team);
        assert_eq!(before, after);
        assert!(InventoryComponent::ALL
            .into_iter()
            .all(|component| before.component(component).leaf_count() == 0));
        pile.close().unwrap();
    }

    #[test]
    fn changed_inventory_or_scope_rejects_the_sample() {
        let team = SigningKey::from_bytes(&[0x61; 32]).verifying_key();
        let other = SigningKey::from_bytes(&[0x62; 32]).verifying_key();

        assert!(ensure_inventory_sample_unchanged(&0, &0, team, Some(team)).is_ok());
        for error in [
            ensure_inventory_sample_unchanged(&0, &1, team, Some(team)).unwrap_err(),
            ensure_inventory_sample_unchanged(&0, &0, team, Some(other)).unwrap_err(),
            ensure_inventory_sample_unchanged(&0, &0, team, None).unwrap_err(),
        ] {
            assert!(error.to_string().contains("sync-visible inventory changed"));
        }
    }

    #[test]
    fn populated_inventory_output_is_complete_and_canonically_ordered() {
        let file = NamedTempFile::new().unwrap();
        let mut pile = Pile::open(file.path()).unwrap();
        let team = SigningKey::from_bytes(&[0x71; 32]).verifying_key();
        let peer = SigningKey::from_bytes(&[0x72; 32]).verifying_key();
        pile.bind_store_scope(team).unwrap();
        pile.insert_peer(PeerEvidence::new(team, peer)).unwrap();
        pile.put::<UnknownBlob, _>(Bytes::from_source(b"inventory fixture".to_vec()))
            .unwrap();

        let (observed_team, manifest) = stable_inventory_manifest(&mut pile).unwrap();
        let output = format_inventory_manifest(observed_team, &manifest);
        let keys: Vec<_> = output
            .lines()
            .map(|line| line.split_once(':').unwrap().0)
            .collect();

        assert_eq!(
            keys,
            [
                "protocol",
                "team_root",
                "generation",
                "peer_count",
                "peer_root",
                "collection_record_count",
                "collection_record_root",
                "capability_proof_count",
                "capability_proof_root",
                "blob_count",
                "blob_root",
            ]
        );
        assert!(output.contains("protocol:            /triblespace/pile-sync/14\n"));
        assert!(output.contains("peer_count: 1\n"));
        assert!(output.contains("collection_record_count: 0\n"));
        assert!(output.contains("collection_record_root:  empty\n"));
        assert!(output.contains("capability_proof_count: 0\n"));
        assert!(output.contains("capability_proof_root:  empty\n"));
        assert!(output.contains("blob_count: 1\n"));
        pile.close().unwrap();
    }
}
