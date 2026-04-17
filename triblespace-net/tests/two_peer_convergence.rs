//! Two-peer convergence properties, exercised deterministically without
//! the iroh transport in the loop. "Gossip" is simulated by copying
//! blobs directly between two independent `Repository<MemoryRepo>`
//! instances and hand-creating tracking branches — the interesting bit
//! is what `merge_tracking_into_local` does on each side, which is
//! where the distributed sync design actually lives.
//!
//! Key property documented here: **sequential gossip converges in one
//! round-pair.** When peers see each other's states one-at-a-time
//! (the realistic gossip ordering), the first peer to merge produces a
//! merge commit `AM` whose ancestry already contains the other peer's
//! original commit. The second peer's sync then sees `AM` in its
//! tracking branch, finds its own head (`commit_B`) already in
//! `ancestors(AM)`, and fast-forwards. No second merge commit is needed.
//!
//! Second property exercised here: **parallel gossip merges diverge for
//! one round-pair, then converge on the next.** When both peers merge
//! simultaneously (before either sees the other's merge), they produce
//! two different merge commits with the same parent set. The next
//! sync round resolves this: one side sees the other's merge, produces
//! a merge whose direct parents include the other's prior head, and the
//! second side fast-forwards to it (because `merge_commit`'s
//! fast-forward check finds the local head in `ancestors(other)` via
//! the direct parent link). No infinite divergence, just a one-round
//! delay relative to the sequential case.

use ed25519_dalek::SigningKey;
use triblespace_core::blob::schemas::simplearchive::SimpleArchive;
use triblespace_core::id::Id;
use triblespace_core::prelude::{BlobStore, BranchStore};
use triblespace_core::repo::memoryrepo::MemoryRepo;
use triblespace_core::repo::{
    BlobStoreGet, BlobStoreList, BlobStorePut, Repository,
};
use triblespace_core::trible::TribleSet;
use triblespace_core::value::schemas::hash::{Blake3, Handle};
use triblespace_core::value::Value;
use triblespace_net::tracking::{
    ensure_tracking_branch, merge_tracking_into_local, MergeOutcome,
};

fn new_repo(seed: u8) -> Repository<MemoryRepo> {
    let signing_key = SigningKey::from_bytes(&[seed; 32]);
    let store = MemoryRepo::default();
    Repository::new(store, signing_key, TribleSet::new()).expect("repo")
}

/// Copy every blob from `src`'s store into `dst`'s store. Content-addressed,
/// so dupes are harmless. Simulates a fire-hose "pull everything reachable
/// from head" fetch.
fn copy_all_blobs(src: &mut Repository<MemoryRepo>, dst: &mut Repository<MemoryRepo>) {
    let reader = src.storage_mut().reader().expect("src reader");
    let handles: Vec<_> = reader
        .blobs()
        .filter_map(|r| r.ok())
        .collect();
    for handle in handles {
        let bytes: anybytes::Bytes = reader
            .get::<anybytes::Bytes, triblespace_core::blob::schemas::UnknownBlob>(handle)
            .expect("src has the blob");
        let _ = dst
            .storage_mut()
            .put::<triblespace_core::blob::schemas::UnknownBlob, _>(bytes);
    }
}

/// Return the hash of the branch-metadata blob that `repo`'s named
/// branch currently points at. This is what a real gossip message would
/// carry for that branch.
fn remote_head_hash(repo: &mut Repository<MemoryRepo>, name: &str) -> [u8; 32] {
    let branch_id = repo
        .lookup_branch(name)
        .expect("lookup branch")
        .expect("branch exists");
    repo.storage_mut()
        .head(branch_id)
        .expect("head")
        .expect("branch has head")
        .raw
}

fn lookup_id(repo: &mut Repository<MemoryRepo>, name: &str) -> Id {
    repo.lookup_branch(name).unwrap().unwrap()
}

/// Simulate one sync round from `remote` into `local`:
/// - copy all of remote's blobs into local
/// - ensure/update a tracking branch in local pointing at remote's HEAD
/// - run `merge_tracking_into_local` on `local` for the named branch
fn sync_round(
    local: &mut Repository<MemoryRepo>,
    remote: &mut Repository<MemoryRepo>,
    branch_name: &str,
    remote_publisher: &[u8; 32],
) -> MergeOutcome {
    copy_all_blobs(remote, local);
    let remote_branch_id = lookup_id(remote, branch_name);
    let remote_head = remote_head_hash(remote, branch_name);
    let tracking_id = ensure_tracking_branch(
        local.storage_mut(),
        remote_branch_id,
        &remote_head,
        branch_name,
        remote_publisher,
    )
    .expect("ensure tracking");
    merge_tracking_into_local(local, tracking_id, branch_name).expect("merge")
}

fn head_commit(repo: &mut Repository<MemoryRepo>, name: &str) -> Value<Handle<Blake3, SimpleArchive>> {
    let id = lookup_id(repo, name);
    let ws = repo.pull(id).unwrap();
    ws.head().expect("branch has head")
}

#[test]
fn sequential_sync_converges_under_divergent_commits() {
    let mut a = new_repo(0x0A);
    let mut b = new_repo(0x0B);
    let pub_a = [0x0Au8; 32];
    let pub_b = [0x0Bu8; 32];

    // Both peers independently commit to "main".
    {
        let id = a.ensure_branch("main", None).unwrap();
        let mut ws = a.pull(id).unwrap();
        ws.commit(TribleSet::new(), "A's commit");
        a.push(&mut ws).unwrap();
    }
    {
        let id = b.ensure_branch("main", None).unwrap();
        let mut ws = b.pull(id).unwrap();
        ws.commit(TribleSet::new(), "B's commit");
        b.push(&mut ws).unwrap();
    }

    let initial_a = head_commit(&mut a, "main");
    let initial_b = head_commit(&mut b, "main");
    assert_ne!(initial_a, initial_b, "peers start with divergent commits");

    // First sync: A pulls B's commit, merges into A's local "main" →
    // produces a merge commit AM whose parents are (commit_A, commit_B).
    let out_a = sync_round(&mut a, &mut b, "main", &pub_b);
    assert!(
        matches!(out_a, MergeOutcome::Merged { .. }),
        "A must produce a merge commit (commits are divergent)"
    );
    let a_after_merge = head_commit(&mut a, "main");
    assert_ne!(a_after_merge, initial_a, "A's main should advance");
    assert_ne!(a_after_merge, initial_b, "A's main must not equal B's commit");

    // Second sync: B pulls A's state — which now includes AM — and
    // observes that its own local head (commit_B) is already in the
    // ancestors of AM. merge_commit takes the fast-forward path.
    let out_b = sync_round(&mut b, &mut a, "main", &pub_a);
    assert!(
        matches!(out_b, MergeOutcome::Merged { .. }),
        "B must advance (fast-forward reports Merged too)"
    );

    // Converged: both peers now point at AM.
    let final_a = head_commit(&mut a, "main");
    let final_b = head_commit(&mut b, "main");
    assert_eq!(
        final_a, final_b,
        "sequential sync must converge in one round-pair"
    );
    assert_eq!(final_a, a_after_merge, "B converges to A's merge, not a new one");

    // A third sync round is now a no-op on both sides.
    let a_again = sync_round(&mut a, &mut b, "main", &pub_b);
    let b_again = sync_round(&mut b, &mut a, "main", &pub_a);
    assert!(matches!(a_again, MergeOutcome::UpToDate));
    assert!(matches!(b_again, MergeOutcome::UpToDate));
}

#[test]
fn parallel_merges_diverge_once_then_converge() {
    // Simulated parallel gossip: both peers see each other's original
    // commits first, then BOTH merge before either has seen the other's
    // merge. The first cross-round leaves them at different merge
    // commits (same parents, different `created_at` / `commit_entity`).
    // The *next* round-pair converges: whichever side merges first
    // produces a merge whose direct parent includes the other side's
    // prior head, and the second side then fast-forwards.
    let mut a = new_repo(0x0A);
    let mut b = new_repo(0x0B);
    let pub_a = [0x0Au8; 32];
    let pub_b = [0x0Bu8; 32];

    // Both peers commit independently.
    {
        let id = a.ensure_branch("main", None).unwrap();
        let mut ws = a.pull(id).unwrap();
        ws.commit(TribleSet::new(), "A's commit");
        a.push(&mut ws).unwrap();
    }
    {
        let id = b.ensure_branch("main", None).unwrap();
        let mut ws = b.pull(id).unwrap();
        ws.commit(TribleSet::new(), "B's commit");
        b.push(&mut ws).unwrap();
    }

    // Exchange only the original commits — no merges in the store yet.
    copy_all_blobs(&mut a, &mut b);
    copy_all_blobs(&mut b, &mut a);

    let a_branch_id = lookup_id(&mut a, "main");
    let b_branch_id = lookup_id(&mut b, "main");
    let a_head = remote_head_hash(&mut a, "main");
    let b_head = remote_head_hash(&mut b, "main");

    let tracking_in_a = ensure_tracking_branch(
        a.storage_mut(), b_branch_id, &b_head, "main", &pub_b,
    )
    .unwrap();
    let tracking_in_b = ensure_tracking_branch(
        b.storage_mut(), a_branch_id, &a_head, "main", &pub_a,
    )
    .unwrap();

    // Simulated parallel merge: both merge against their pre-merge
    // views, without seeing each other's resulting merge commits.
    merge_tracking_into_local(&mut a, tracking_in_a, "main").unwrap();
    merge_tracking_into_local(&mut b, tracking_in_b, "main").unwrap();

    let a_after_parallel = head_commit(&mut a, "main");
    let b_after_parallel = head_commit(&mut b, "main");
    assert_ne!(
        a_after_parallel, b_after_parallel,
        "parallel merges produce divergent merge commits"
    );

    // Resync sequentially. A sees B's merge and produces AM' whose
    // direct parents include both AM and BM. B then sees AM' and
    // observes that its own local head (BM) is a direct parent of AM',
    // so it fast-forwards — no BM' produced.
    sync_round(&mut a, &mut b, "main", &pub_b);
    sync_round(&mut b, &mut a, "main", &pub_a);

    assert_eq!(
        head_commit(&mut a, "main"),
        head_commit(&mut b, "main"),
        "one round-pair after parallel divergence is enough to converge \
         (direct-parent check in merge_commit's fast-forward path)"
    );
}

#[test]
fn single_round_converges_when_only_one_side_advanced() {
    // If only A commits and B is empty, one sync round fast-forwards B
    // without producing a merge commit.
    let mut a = new_repo(0x0A);
    let mut b = new_repo(0x0B);
    let pub_a = [0x0Au8; 32];

    {
        let id = a.ensure_branch("main", None).unwrap();
        let mut ws = a.pull(id).unwrap();
        ws.commit(TribleSet::new(), "A's only commit");
        a.push(&mut ws).unwrap();
    }

    let outcome = sync_round(&mut b, &mut a, "main", &pub_a);
    assert!(
        matches!(outcome, MergeOutcome::Merged { .. }),
        "fast-forward still reports Merged (advance-to-tip)"
    );
    assert_eq!(
        head_commit(&mut a, "main"),
        head_commit(&mut b, "main"),
        "one round is enough when only one side advanced"
    );

    // Second round is a no-op on both sides.
    let again = sync_round(&mut b, &mut a, "main", &pub_a);
    assert!(matches!(again, MergeOutcome::UpToDate));
}
