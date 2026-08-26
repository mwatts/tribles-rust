//! Foreground private-custody replication over fixed endpoint tickets.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use clap::{Args, Parser};
use ed25519_dalek::SigningKey;
use iroh_base::{EndpointAddr, EndpointId, TransportAddr};
use iroh_tickets::endpoint::EndpointTicket;

use triblespace_core::capability::CapabilityProofId;
use triblespace_core::repo::pile::Pile;
use triblespace_net::replica::{
    CustodyInventoryStats, CustodyReconcileOutcome, CustodyReplica, CustodyReplicaConfig,
};

#[derive(Parser)]
pub enum Command {
    /// Validate the complete local configuration and print its static ticket.
    Status {
        #[command(flatten)]
        config: ConfigArgs,
    },
    /// Run foreground anti-entropy until SIGINT or SIGTERM.
    Run {
        #[command(flatten)]
        config: ConfigArgs,
        /// Seconds between complete anti-entropy sweeps.
        #[arg(long, value_name = "SECS", default_value_t = 30)]
        interval: u64,
    },
}

#[derive(Args)]
pub struct ConfigArgs {
    /// Pile whose complete semantic contents are held in custody.
    pile: PathBuf,
    /// Existing per-machine network key, distinct from content signing keys.
    #[arg(long)]
    network_key: PathBuf,
    /// Exact private-fabric socket. Wildcard addresses and port zero are refused.
    #[arg(long)]
    bind: SocketAddr,
    /// Static peer EndpointTicket. Repeat once per peer; bare endpoint ids fail.
    #[arg(long = "peer", value_name = "ENDPOINT_TICKET")]
    peers: Vec<String>,
    /// External root for the ordinary CONNECT handshake (32-byte hex).
    #[arg(long)]
    connect_root: String,
    /// Exact CONNECT proof id resident in the pile.
    #[arg(long)]
    connect_proof: String,
    /// Independent external root for custody replication (32-byte hex).
    #[arg(long)]
    replica_root: String,
    /// Exact 32-byte custody replica-set identity.
    #[arg(long)]
    replica_set: String,
    /// Exact REPLICATE proof id whose effective leaf mode is invoke-only.
    #[arg(long)]
    replica_proof: String,
    /// Existing caller-controlled directory for incomplete large receives.
    #[arg(long)]
    temp_dir: PathBuf,
}

struct Prepared {
    pile_path: PathBuf,
    pile: Pile,
    network_key: SigningKey,
    endpoint_addr: EndpointAddr,
    connect_proof_id: CapabilityProofId,
    replica_proof_id: CapabilityProofId,
    config: CustodyReplicaConfig,
}

pub fn run(command: Command) -> Result<()> {
    match command {
        Command::Status { config } => run_status(config),
        Command::Run { config, interval } => run_service(config, interval),
    }
}

fn run_status(args: ConfigArgs) -> Result<()> {
    let mut prepared = prepare(args)?;
    let inventory = match CustodyReplica::<Pile>::preflight(
        &mut prepared.pile,
        prepared.network_key.verifying_key(),
        &prepared.config,
    ) {
        Ok(inventory) => inventory,
        Err(error) => {
            return Err(close_after_error(
                prepared.pile,
                error.context("preflight custody inventory"),
            ));
        }
    };
    print_configuration("configured", &prepared, &inventory);
    prepared
        .pile
        .close()
        .map_err(|error| anyhow!("close pile: {error:?}"))?;
    Ok(())
}

fn run_service(args: ConfigArgs, interval: u64) -> Result<()> {
    if interval == 0 {
        bail!("--interval must be at least one second");
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build custody runtime")?;
    let prepared = prepare(args)?;
    let Prepared {
        pile_path,
        pile,
        network_key,
        config,
        ..
    } = prepared;
    let mut replica = match CustodyReplica::new(pile, network_key, config) {
        Ok(replica) => replica,
        Err(start_error) => {
            let (pile, error) = start_error.into_parts();
            return Err(close_after_error(
                pile,
                error.context("start custody replica"),
            ));
        }
    };

    let endpoint_addr = replica.endpoint_addr();
    println!("state:          listening");
    println!("node:           {}", replica.id());
    println!(
        "bind:           {}",
        endpoint_addr.ip_addrs().next().unwrap()
    );
    println!("ticket:         {}", endpoint_ticket(&endpoint_addr));
    eprintln!("custody replication active; SIGINT or SIGTERM stops cleanly");

    let service_result = runtime.block_on(service_loop(
        &mut replica,
        &pile_path,
        Duration::from_secs(interval),
    ));
    drop(runtime);
    finish_service(replica, service_result)
}

fn prepare(args: ConfigArgs) -> Result<Prepared> {
    validate_bind(args.bind)?;
    validate_temp_dir(&args.temp_dir)?;
    let network_key = triblespace_core::signing_key_file::load_existing(&args.network_key)
        .context("load custody network key")?;
    let endpoint_id: EndpointId = triblespace_net::identity::iroh_secret(&network_key)
        .public()
        .into();
    let peers = parse_peer_tickets(&args.peers, endpoint_id)?;
    let endpoint_addr = EndpointAddr::from_parts(endpoint_id, [TransportAddr::Ip(args.bind)]);
    let connect_root = crate::cli::team::parse_public_key(&args.connect_root, "CONNECT root")?;
    let connect_proof_id = crate::cli::team::parse_proof_id(&args.connect_proof)?;
    let replica_root = crate::cli::team::parse_public_key(&args.replica_root, "replica root")?;
    let replica_set = crate::cli::team::parse_replica_set(&args.replica_set)?;
    let replica_proof_id = crate::cli::team::parse_proof_id(&args.replica_proof)?;

    let mut pile = super::super::open_refreshed(&args.pile)?;
    let proofs = (|| {
        reject_opaque_records(&mut pile, &args.pile)?;
        let connect_proof = crate::cli::team::resolve_connect_bundle(
            &mut pile,
            connect_root,
            connect_proof_id,
            network_key.verifying_key(),
        )?;
        let replica_proof = crate::cli::team::resolve_replica_bundle(
            &mut pile,
            replica_root,
            replica_set,
            replica_proof_id,
            network_key.verifying_key(),
        )?;
        Ok::<_, anyhow::Error>((connect_proof, replica_proof))
    })();
    let (connect_proof, replica_proof) = match proofs {
        Ok(proofs) => proofs,
        Err(error) => return Err(close_after_error(pile, error)),
    };

    Ok(Prepared {
        pile_path: args.pile,
        pile,
        network_key,
        endpoint_addr,
        connect_proof_id,
        replica_proof_id,
        config: CustodyReplicaConfig {
            peers,
            connect_root,
            connect_proof,
            replica_root,
            replica_set,
            replica_proof,
            bind_addr: args.bind,
            receive_temp_dir: args.temp_dir,
        },
    })
}

fn print_configuration(state: &str, prepared: &Prepared, inventory: &CustodyInventoryStats) {
    println!("state:          {state}");
    println!("node:           {}", prepared.endpoint_addr.id);
    println!("bind:           {}", prepared.config.bind_addr);
    println!(
        "ticket:         {}",
        endpoint_ticket(&prepared.endpoint_addr)
    );
    println!(
        "connect_root:   {}",
        hex::encode(prepared.config.connect_root.to_bytes())
    );
    println!(
        "connect_proof:  {}",
        hex::encode(prepared.connect_proof_id.raw)
    );
    println!(
        "replica_root:   {}",
        hex::encode(prepared.config.replica_root.to_bytes())
    );
    println!(
        "replica_set:    {}",
        hex::encode(prepared.config.replica_set.into_bytes())
    );
    println!(
        "replica_proof:  {}",
        hex::encode(prepared.replica_proof_id.raw)
    );
    println!("peers:          {}", prepared.config.peers.len());
    println!(
        "temp_dir:       {}",
        prepared.config.receive_temp_dir.display()
    );
    println!("inventory_blobs:       {}", inventory.blobs);
    println!("inventory_blob_bytes:  {}", inventory.blob_bytes);
    println!("inventory_records:     {}", inventory.collection_records);
    println!("inventory_proofs:      {}", inventory.capability_proofs);
    println!(
        "inventory_generation:  {}",
        hex::encode(inventory.generation)
    );
    println!("inventory_build:       {:?}", inventory.build_elapsed);
    println!("authorization:  CONNECT accepted");
    println!("authorization:  REPLICATE_STORE accepted");
}

fn endpoint_ticket(address: &EndpointAddr) -> String {
    EndpointTicket::new(address.clone()).to_string()
}

fn validate_bind(bind: SocketAddr) -> Result<()> {
    validate_direct_socket(bind, "custody bind")
}

fn validate_temp_dir(path: &Path) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("inspect receive temp directory {}", path.display()))?;
    if !metadata.is_dir() {
        bail!("receive temp path {} is not a directory", path.display());
    }
    Ok(())
}

fn reject_opaque_records(pile: &mut Pile, path: &Path) -> Result<()> {
    let opaque_records = pile
        .opaque_record_count()
        .map_err(|error| super::super::pile_read_error(path, error))?;
    if opaque_records != 0 {
        bail!(
            "custody refuses {} opaque pile record(s): complete semantic replication cannot preserve unknown record kinds",
            opaque_records,
        );
    }
    Ok(())
}

fn parse_peer_tickets(values: &[String], local: EndpointId) -> Result<Vec<EndpointAddr>> {
    let mut peers = BTreeMap::<EndpointId, EndpointAddr>::new();
    for value in values {
        let ticket = value.parse::<EndpointTicket>().map_err(|error| {
            anyhow!("invalid custody peer {value:?}: expected an EndpointTicket ({error})")
        })?;
        let address: EndpointAddr = ticket.into();
        if address.id == local {
            bail!("custody peer ticket names the local network identity");
        }
        if address.addrs.len() != 1 {
            bail!(
                "custody peer ticket {} must name exactly one explicit private-fabric IP socket",
                address.id,
            );
        }
        let Some(TransportAddr::Ip(socket)) = address.addrs.first() else {
            bail!(
                "custody peer ticket {} contains a non-IP route; relay/custom discovery is forbidden",
                address.id
            );
        };
        validate_direct_socket(*socket, "custody peer")?;
        if let Some(existing) = peers.insert(address.id, address.clone()) {
            if existing == address {
                bail!("custody peer ticket repeats endpoint {}", address.id);
            }
            bail!("conflicting custody tickets name endpoint {}", address.id);
        }
    }
    Ok(peers.into_values().collect())
}

fn validate_direct_socket(socket: SocketAddr, label: &str) -> Result<()> {
    if socket.port() == 0 {
        bail!("{label} port must be restart-stable and nonzero");
    }
    if socket.ip().is_unspecified() || socket.ip().is_multicast() {
        bail!("{label} address {socket} is not one explicit unicast interface");
    }
    if matches!(socket.ip(), std::net::IpAddr::V4(ip) if ip.is_broadcast()) {
        bail!("{label} address {socket} is broadcast");
    }
    Ok(())
}

async fn service_loop(
    replica: &mut CustodyReplica<Pile>,
    pile_path: &Path,
    interval: Duration,
) -> Result<()> {
    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);
    loop {
        reject_opaque_records(replica.store_mut(), pile_path)?;
        let reconcile = replica.reconcile_once();
        tokio::pin!(reconcile);
        let outcome = tokio::select! {
            result = &mut reconcile => result?,
            signal = &mut shutdown => {
                signal?;
                // Dropping the sweep cancels only asynchronous page/range work.
                // Store mutations themselves are synchronous, the anonymous
                // receive file closes on drop, and finish_service closes (and
                // therefore flushes) every Pile before returning.
                eprintln!("shutdown requested; cancelling the active custody sweep");
                return Ok(());
            }
        };
        print_outcome(&outcome);
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            signal = &mut shutdown => {
                signal?;
                eprintln!("shutdown requested");
                return Ok(());
            }
        }
    }
}

fn print_outcome(outcome: &CustodyReconcileOutcome) {
    eprintln!(
        "custody sweep: {}/{} peers; +{} blobs ({} bytes), +{} collection records, +{} proofs; {} pages; generation {}",
        outcome.peers_completed,
        outcome.peers_attempted,
        outcome.blobs_added,
        outcome.blob_bytes_added,
        outcome.collection_records_added,
        outcome.capability_proofs_added,
        outcome.pages_read,
        hex::encode(outcome.generation),
    );
    for error in &outcome.peer_errors {
        eprintln!("  peer unavailable: {error}");
    }
}

#[cfg(unix)]
async fn shutdown_signal() -> Result<()> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("install SIGTERM handler")?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result.context("listen for SIGINT"),
        _ = terminate.recv() => Ok(()),
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> Result<()> {
    tokio::signal::ctrl_c()
        .await
        .context("listen for interrupt signal")
}

fn finish_service(replica: CustodyReplica<Pile>, service_result: Result<()>) -> Result<()> {
    let (pile, shutdown_result) = match replica.shutdown() {
        Ok(pile) => (pile, Ok(())),
        Err(shutdown_error) => {
            let (pile, error) = shutdown_error.into_parts();
            (pile, Err(error.context("shut down custody host")))
        }
    };
    let service_result = match (service_result, shutdown_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(service_error), Err(shutdown_error)) => Err(anyhow!(
            "{service_error:#}; additionally custody shutdown failed: {shutdown_error:#}"
        )),
    };
    let close_result = pile
        .close()
        .map_err(|error| anyhow!("close pile after custody shutdown: {error:?}"));
    match (service_result, close_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(close_error)) => Err(anyhow!(
            "{error:#}; additionally pile close failed: {close_error:#}"
        )),
    }
}

fn close_after_error(pile: Pile, error: anyhow::Error) -> anyhow::Error {
    match pile.close() {
        Ok(()) => error,
        Err(close_error) => anyhow!("{error:#}; additionally pile close failed: {close_error:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh_base::SecretKey;

    #[test]
    fn custody_peers_require_static_ip_tickets() {
        let local = EndpointId::from(SecretKey::from_bytes(&[1; 32]).public());
        let remote = EndpointId::from(SecretKey::from_bytes(&[2; 32]).public());
        let direct = EndpointAddr::from_parts(
            remote,
            [TransportAddr::Ip("10.242.0.2:49152".parse().unwrap())],
        );
        let ticket = EndpointTicket::new(direct.clone()).to_string();
        assert_eq!(
            parse_peer_tickets(&[ticket.clone()], local).unwrap(),
            vec![direct]
        );
        assert!(parse_peer_tickets(&[ticket.clone(), ticket], local).is_err());
        assert!(parse_peer_tickets(&[remote.to_string()], local).is_err());

        let addressless = EndpointTicket::new(EndpointAddr::from(remote)).to_string();
        assert!(parse_peer_tickets(&[addressless], local).is_err());

        let multiple = EndpointTicket::new(EndpointAddr::from_parts(
            remote,
            [
                TransportAddr::Ip("10.242.0.2:49152".parse().unwrap()),
                TransportAddr::Ip("10.242.0.3:49152".parse().unwrap()),
            ],
        ))
        .to_string();
        assert!(parse_peer_tickets(&[multiple], local).is_err());

        let multicast = EndpointTicket::new(EndpointAddr::from_parts(
            remote,
            [TransportAddr::Ip("239.1.2.3:49152".parse().unwrap())],
        ))
        .to_string();
        assert!(parse_peer_tickets(&[multicast], local).is_err());
    }

    #[test]
    fn custody_bind_is_exact_and_restart_stable() {
        assert!(validate_bind("10.242.0.1:49152".parse().unwrap()).is_ok());
        assert!(validate_bind("0.0.0.0:49152".parse().unwrap()).is_err());
        assert!(validate_bind("10.242.0.1:0".parse().unwrap()).is_err());
        assert!(validate_bind("239.1.2.3:49152".parse().unwrap()).is_err());
        assert!(validate_bind("255.255.255.255:49152".parse().unwrap()).is_err());
    }
}
