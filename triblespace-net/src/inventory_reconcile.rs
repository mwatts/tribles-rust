//! Root- and count-proven traversal of one remote inventory component.
//!
//! The remote authenticated trie shape is the only traversal plan. A local
//! immutable observation may discharge a remote subtree only when both its
//! digest and leaf count match; local and remote compressed paths never need
//! to align. Every remaining locator is derived from an authenticated remote
//! branch, so an absent locator is a protocol contradiction rather than
//! evidence of convergence.

use anyhow::{Result, anyhow, bail};

use crate::inventory::{ComponentManifest, InventoryComponent};
use crate::inventory_wire::{
    InventoryLeaf, InventoryNode, InventoryNodeRequest, InventoryNodeResponse,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingNode {
    prefix: Vec<u8>,
    digest: [u8; 32],
    leaf_count: u64,
}

/// Count-proven outcome of one completed component walk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InventoryWalkResult {
    pub(crate) component: InventoryComponent,
    pub(crate) root: Option<[u8; 32]>,
    pub(crate) leaf_count: u64,
    pub(crate) missing_count: u64,
}

/// Pipelined traversal state for one immutable remote component root.
///
/// A request is its own authenticated locator and each bidirectional stream is
/// its correlation envelope, so callers may keep a bounded number in flight
/// and accept them in any order. `local_summary` and `local_contains` passed to
/// the methods below must describe the same immutable local observation for the
/// whole walk. Later local admissions can only make work redundant; they must
/// not be used to manufacture subtree skips against a moving inventory.
pub(crate) struct InventoryWalker {
    team: ed25519_dalek::VerifyingKey,
    component: InventoryComponent,
    root: Option<[u8; 32]>,
    leaf_count: u64,
    frontier: Vec<PendingNode>,
    in_flight: Vec<PendingNode>,
    accounted: u64,
    missing_count: u64,
    failed: bool,
}

impl InventoryWalker {
    pub(crate) fn new(
        team: ed25519_dalek::VerifyingKey,
        manifest: ComponentManifest,
    ) -> Result<Self> {
        let root = manifest.root();
        let leaf_count = manifest.leaf_count();
        if root.is_none() != (leaf_count == 0) {
            bail!("inventory manifest root and leaf count disagree");
        }
        let frontier = root
            .map(|digest| {
                vec![PendingNode {
                    prefix: Vec::new(),
                    digest,
                    leaf_count,
                }]
            })
            .unwrap_or_default();
        let walker = Self {
            team,
            component: manifest.component(),
            root,
            leaf_count,
            frontier,
            in_flight: Vec::new(),
            accounted: 0,
            missing_count: 0,
            failed: false,
        };
        walker.verify_count_invariant()?;
        Ok(walker)
    }

    /// Return the next pinned remote request, skipping exact local subtrees.
    ///
    /// `None` means there is currently no ready frontier work. Outstanding
    /// requests may still reveal children, so the caller must wait for them
    /// before consuming the walker with [`Self::finish`]. A local summary is
    /// an exact `(digest, leaf_count)` for the protocol-relative prefix.
    pub(crate) fn next_request(
        &mut self,
        mut local_summary: impl FnMut(InventoryComponent, &[u8]) -> Option<([u8; 32], u64)>,
    ) -> Result<Option<InventoryNodeRequest>> {
        if self.failed {
            bail!("inventory walker is failed");
        }
        while let Some(pending) = self.frontier.pop() {
            if local_summary(self.component, &pending.prefix)
                == Some((pending.digest, pending.leaf_count))
            {
                self.accounted = self
                    .accounted
                    .checked_add(pending.leaf_count)
                    .ok_or_else(|| anyhow!("inventory accounted count overflow"))?;
                self.verify_count_invariant()?;
                continue;
            }

            let request = InventoryNodeRequest::new(
                self.team,
                self.component,
                self.root.expect("a pending node has a nonempty root"),
                self.leaf_count,
                pending.prefix.clone(),
                pending.digest,
            )?;
            self.in_flight.push(pending);
            self.verify_count_invariant()?;
            return Ok(Some(request));
        }

        self.verify_count_invariant()?;
        if self.in_flight.is_empty() && self.accounted != self.leaf_count {
            self.failed = true;
            bail!(
                "inventory frontier ended after accounting for {} of {} leaves",
                self.accounted,
                self.leaf_count
            );
        }
        Ok(None)
    }

    /// Consume the response correlated with `request` and yield one missing
    /// authenticated leaf immediately. Responses may arrive in any order.
    ///
    /// Snapshot eviction, an absent authenticated child, and every digest or
    /// exact-count mismatch fail the walk. None of them discharges frontier
    /// work or can be mistaken for convergence. Callers may admit each yielded
    /// leaf monotonically before the walk completes; a later failure preserves
    /// valid prefix progress and a future periodic sweep repairs the remainder.
    pub(crate) fn accept(
        &mut self,
        request: &InventoryNodeRequest,
        response: InventoryNodeResponse,
        mut local_contains: impl FnMut(InventoryComponent, &[u8]) -> bool,
    ) -> Result<Option<InventoryLeaf>> {
        if self.failed {
            bail!("inventory walker is failed");
        }
        let result = self.accept_inner(request, response, &mut local_contains);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn accept_inner(
        &mut self,
        request: &InventoryNodeRequest,
        response: InventoryNodeResponse,
        local_contains: &mut impl FnMut(InventoryComponent, &[u8]) -> bool,
    ) -> Result<Option<InventoryLeaf>> {
        if request.component != self.component
            || Some(request.root) != self.root
            || request.leaf_count != self.leaf_count
        {
            bail!("inventory response request does not belong to this pinned walk");
        }
        let position = self
            .in_flight
            .iter()
            .position(|pending| {
                pending.prefix == request.prefix && pending.digest == request.expected_digest
            })
            .ok_or_else(|| anyhow!("inventory response has no matching in-flight request"))?;
        let pending = self.in_flight.swap_remove(position);
        let node = match response {
            InventoryNodeResponse::SnapshotUnavailable => {
                bail!("pinned inventory snapshot is unavailable; request a fresh manifest")
            }
            InventoryNodeResponse::PrefixAbsent => bail!(
                "authenticated inventory prefix {} is absent",
                hex::encode(&pending.prefix)
            ),
            InventoryNodeResponse::Found(node) => node,
        };

        if node.digest() != pending.digest {
            bail!("inventory node digest does not match its authenticated parent");
        }
        if node.leaf_count() != pending.leaf_count {
            bail!(
                "inventory node leaf count {} does not match authenticated pending count {}",
                node.leaf_count(),
                pending.leaf_count
            );
        }

        let missing = match node {
            InventoryNode::Leaf { leaf, .. } => {
                if pending.leaf_count != 1 || !leaf.key.starts_with(&pending.prefix) {
                    bail!("inventory leaf does not discharge its authenticated locator");
                }
                let missing = (!local_contains(self.component, &leaf.key)).then_some(leaf);
                if missing.is_some() {
                    self.missing_count = self
                        .missing_count
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("inventory missing count overflow"))?;
                }
                self.accounted = self
                    .accounted
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("inventory accounted count overflow"))?;
                missing
            }
            InventoryNode::Branch { branch, .. } => {
                let end_depth = usize::from(branch.end_depth);
                if end_depth >= branch.representative.len()
                    || !branch.representative[..end_depth].starts_with(&pending.prefix)
                {
                    bail!("inventory branch does not extend its authenticated locator");
                }

                let mut previous = None;
                let mut child_total = 0u64;
                for child in &branch.children {
                    if previous.is_some_and(|edge| child.edge <= edge) || child.leaf_count == 0 {
                        bail!("inventory branch children are not canonical");
                    }
                    previous = Some(child.edge);
                    child_total = child_total
                        .checked_add(child.leaf_count)
                        .ok_or_else(|| anyhow!("inventory child count overflow"))?;
                }
                if child_total != pending.leaf_count {
                    bail!(
                        "inventory branch child counts {child_total} do not discharge pending count {}",
                        pending.leaf_count
                    );
                }

                // Reverse push makes the LIFO frontier walk ascending edges.
                // Bytes after end_depth are representative-only routing
                // witnesses and never participate in a child locator.
                for child in branch.children.into_iter().rev() {
                    let mut prefix = branch.representative[..end_depth].to_vec();
                    prefix.push(child.edge);
                    self.frontier.push(PendingNode {
                        prefix,
                        digest: child.digest,
                        leaf_count: child.leaf_count,
                    });
                }
                None
            }
        };
        self.verify_count_invariant()?;
        Ok(missing)
    }

    /// Prove completion and return its bounded counters.
    pub(crate) fn finish(self) -> Result<InventoryWalkResult> {
        if self.failed {
            bail!("inventory walker failed before completion");
        }
        if !self.in_flight.is_empty()
            || !self.frontier.is_empty()
            || self.accounted != self.leaf_count
        {
            bail!(
                "inventory walk is incomplete: accounted {} of {} leaves",
                self.accounted,
                self.leaf_count
            );
        }
        Ok(InventoryWalkResult {
            component: self.component,
            root: self.root,
            leaf_count: self.leaf_count,
            missing_count: self.missing_count,
        })
    }

    fn verify_count_invariant(&self) -> Result<()> {
        let mut total = self.accounted;
        for in_flight in &self.in_flight {
            total = total
                .checked_add(in_flight.leaf_count)
                .ok_or_else(|| anyhow!("inventory count invariant overflow"))?;
        }
        for pending in &self.frontier {
            total = total
                .checked_add(pending.leaf_count)
                .ok_or_else(|| anyhow!("inventory count invariant overflow"))?;
        }
        if total != self.leaf_count {
            bail!(
                "inventory count invariant accounts for {total} of {} leaves",
                self.leaf_count
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use triblespace_core::patch::{Blake3Merkle, IdentitySchema, PATCH};

    use super::*;
    use crate::inventory_wire::{InventoryBranch, InventoryChild, InventoryLeafValue};

    type TestInventory = PATCH<32, IdentitySchema, (), Blake3Merkle>;

    fn key(bytes: [u8; 4]) -> [u8; 32] {
        let mut key = [0; 32];
        key[..4].copy_from_slice(&bytes);
        key
    }

    fn team() -> ed25519_dalek::VerifyingKey {
        SigningKey::from_bytes(&[0x51; 32]).verifying_key()
    }

    fn manifest(inventory: &TestInventory) -> ComponentManifest {
        ComponentManifest::new(
            InventoryComponent::Blob,
            inventory.len(),
            inventory.merkle_root(),
        )
    }

    fn response(inventory: &TestInventory, prefix: &[u8]) -> InventoryNodeResponse {
        let Some(node) = inventory.merkle_node(prefix) else {
            return InventoryNodeResponse::PrefixAbsent;
        };
        if node.is_leaf() {
            return InventoryNodeResponse::Found(InventoryNode::Leaf {
                digest: node.digest(),
                leaf: InventoryLeaf {
                    key: node.representative().to_vec(),
                    value: InventoryLeafValue::Blob,
                },
            });
        }
        InventoryNodeResponse::Found(InventoryNode::Branch {
            digest: node.digest(),
            leaf_count: node.leaf_count(),
            branch: InventoryBranch {
                representative: node.representative().to_vec(),
                end_depth: u8::try_from(node.end_depth()).unwrap(),
                children: node
                    .children()
                    .map(|(edge, child)| InventoryChild {
                        edge,
                        digest: child.digest(),
                        leaf_count: child.leaf_count(),
                    })
                    .collect(),
            },
        })
    }

    fn run_walk(
        remote: &TestInventory,
        local: &TestInventory,
    ) -> (InventoryWalkResult, Vec<Vec<u8>>, Vec<InventoryLeaf>) {
        let mut walker = InventoryWalker::new(team(), manifest(remote)).unwrap();
        let mut requests = Vec::new();
        let mut missing = Vec::new();
        loop {
            let request = walker
                .next_request(|component, prefix| {
                    assert_eq!(component, InventoryComponent::Blob);
                    local
                        .merkle_node(prefix)
                        .map(|node| (node.digest(), node.leaf_count()))
                })
                .unwrap();
            let Some(request) = request else {
                break;
            };
            requests.push(request.prefix.clone());
            if let Some(leaf) = walker
                .accept(
                    &request,
                    response(remote, &request.prefix),
                    |component, raw| {
                        assert_eq!(component, InventoryComponent::Blob);
                        let key: [u8; 32] = raw.try_into().unwrap();
                        local.get(&key).is_some()
                    },
                )
                .unwrap()
            {
                missing.push(leaf);
            }
        }
        (walker.finish().unwrap(), requests, missing)
    }

    fn missing_keys(missing: &[InventoryLeaf]) -> Vec<[u8; 32]> {
        missing
            .iter()
            .map(|leaf| leaf.key.as_slice().try_into().unwrap())
            .collect()
    }

    #[test]
    fn remote_split_shallower_than_local_compressed_path() {
        let a = key([1, 2, 3, 0]);
        let b = key([1, 2, 4, 0]);
        let c = key([1, 2, 3, 255]);
        let remote = TestInventory::from_keys([a, b]);
        let local = TestInventory::from_keys([a, c]);

        let (result, _, missing) = run_walk(&remote, &local);
        assert_eq!(result.leaf_count, 2);
        assert_eq!(result.missing_count, 1);
        assert_eq!(missing_keys(&missing), [b]);
    }

    #[test]
    fn local_split_shallower_than_remote_compressed_path() {
        let a = key([1, 2, 3, 0]);
        let b = key([1, 2, 4, 0]);
        let c = key([1, 2, 3, 255]);
        let remote = TestInventory::from_keys([a, c]);
        let local = TestInventory::from_keys([a, b]);

        let (result, _, missing) = run_walk(&remote, &local);
        assert_eq!(result.leaf_count, 2);
        assert_eq!(result.missing_count, 1);
        assert_eq!(missing_keys(&missing), [c]);
    }

    #[test]
    fn singleton_and_branch_reconcile_in_both_directions() {
        let a = key([1, 2, 3, 0]);
        let c = key([1, 2, 3, 255]);

        let remote_singleton = TestInventory::from_keys([a]);
        let local_branch = TestInventory::from_keys([a, c]);
        let (singleton, _, singleton_missing) = run_walk(&remote_singleton, &local_branch);
        assert_eq!(singleton.leaf_count, 1);
        assert_eq!(singleton.missing_count, 0);
        assert!(singleton_missing.is_empty());

        let remote_branch = TestInventory::from_keys([a, c]);
        let local_singleton = TestInventory::from_keys([a]);
        let (branch, _, branch_missing) = run_walk(&remote_branch, &local_singleton);
        assert_eq!(branch.leaf_count, 2);
        assert_eq!(branch.missing_count, 1);
        assert_eq!(missing_keys(&branch_missing), [c]);
    }

    #[test]
    fn child_locator_may_end_inside_remote_compression() {
        let d = key([0, 7, 8, 1]);
        let e = key([0, 7, 8, 2]);
        let f = key([9, 0, 0, 0]);
        let remote = TestInventory::from_keys([d, e, f]);
        let local = TestInventory::from_keys([d, f]);

        let (result, requests, missing) = run_walk(&remote, &local);
        assert_eq!(result.missing_count, 1);
        assert_eq!(missing_keys(&missing), [e]);
        assert!(requests.iter().any(|prefix| prefix == &[0]));
        assert!(requests.iter().any(|prefix| prefix == &[0, 7, 8, 2]));
    }

    #[test]
    fn independent_frontier_responses_may_arrive_out_of_order() {
        let remote = TestInventory::from_keys((0..32u8).map(|edge| key([edge, 7, 8, 9])));
        let mut walker = InventoryWalker::new(team(), manifest(&remote)).unwrap();

        let root = walker.next_request(|_, _| None).unwrap().unwrap();
        walker
            .accept(&root, response(&remote, &root.prefix), |_, _| false)
            .unwrap();

        let mut frontier = Vec::new();
        while let Some(request) = walker.next_request(|_, _| None).unwrap() {
            frontier.push(request);
        }
        assert_eq!(frontier.len(), 32);
        assert!(walker.finish().is_err(), "outstanding requests are counted");

        // Reconstruct the same state because `finish` consumes the walker.
        let mut walker = InventoryWalker::new(team(), manifest(&remote)).unwrap();
        let root = walker.next_request(|_, _| None).unwrap().unwrap();
        walker
            .accept(&root, response(&remote, &root.prefix), |_, _| false)
            .unwrap();
        let mut frontier = Vec::new();
        while let Some(request) = walker.next_request(|_, _| None).unwrap() {
            frontier.push(request);
        }

        let mut missing = Vec::new();
        for request in frontier.into_iter().rev() {
            missing.push(
                walker
                    .accept(&request, response(&remote, &request.prefix), |_, _| false)
                    .unwrap()
                    .expect("empty local inventory misses every leaf"),
            );
        }
        assert!(walker.next_request(|_, _| None).unwrap().is_none());
        let result = walker.finish().unwrap();
        assert_eq!(result.leaf_count, 32);
        assert_eq!(result.missing_count, 32);
        assert_eq!(missing.len(), 32);
    }

    #[test]
    fn unavailable_or_absent_prefix_never_completes() {
        let remote = TestInventory::from_keys([key([1, 2, 3, 0])]);
        for response in [
            InventoryNodeResponse::SnapshotUnavailable,
            InventoryNodeResponse::PrefixAbsent,
        ] {
            let mut walker = InventoryWalker::new(team(), manifest(&remote)).unwrap();
            let request = walker.next_request(|_, _| None).unwrap().unwrap();
            assert!(walker.accept(&request, response, |_, _| false).is_err());
            assert!(walker.finish().is_err());
        }

        let branch =
            TestInventory::from_keys([key([0, 7, 8, 1]), key([0, 7, 8, 2]), key([9, 0, 0, 0])]);
        let mut child_absent = InventoryWalker::new(team(), manifest(&branch)).unwrap();
        let root = child_absent.next_request(|_, _| None).unwrap().unwrap();
        child_absent
            .accept(&root, response(&branch, &root.prefix), |_, _| false)
            .unwrap();
        let child = child_absent.next_request(|_, _| None).unwrap().unwrap();
        assert!(!child.prefix.is_empty());
        assert!(
            child_absent
                .accept(&child, InventoryNodeResponse::PrefixAbsent, |_, _| false)
                .is_err()
        );
        assert!(child_absent.finish().is_err());
    }

    #[test]
    fn wrong_digest_or_exact_child_count_never_completes() {
        let a = key([1, 2, 3, 0]);
        let b = key([1, 2, 3, 1]);
        let c = key([1, 2, 3, 2]);
        let d = key([1, 2, 4, 0]);
        let remote = TestInventory::from_keys([a, b, c, d]);

        let mut wrong_digest = InventoryWalker::new(team(), manifest(&remote)).unwrap();
        let root = wrong_digest.next_request(|_, _| None).unwrap().unwrap();
        let InventoryNodeResponse::Found(mut node) = response(&remote, &root.prefix) else {
            unreachable!()
        };
        match &mut node {
            InventoryNode::Leaf { digest, .. } | InventoryNode::Branch { digest, .. } => {
                digest[0] ^= 1;
            }
        }
        assert!(
            wrong_digest
                .accept(&root, InventoryNodeResponse::Found(node), |_, _| false)
                .is_err()
        );
        assert!(wrong_digest.finish().is_err());

        // Blake3Merkle v2 makes this redistribution fail immediately in the
        // wire validator. This direct state-machine test deliberately bypasses
        // that boundary to retain defense in depth: even an internally forged
        // 3/1 -> 2/2 response cannot make the walk complete because the first
        // exact child response disagrees with its pending count.
        let mut wrong_count = InventoryWalker::new(team(), manifest(&remote)).unwrap();
        let root = wrong_count.next_request(|_, _| None).unwrap().unwrap();
        let InventoryNodeResponse::Found(InventoryNode::Branch {
            digest,
            leaf_count,
            mut branch,
        }) = response(&remote, &root.prefix)
        else {
            unreachable!()
        };
        assert_eq!(
            branch
                .children
                .iter()
                .map(|child| child.leaf_count)
                .collect::<Vec<_>>(),
            [3, 1]
        );
        branch.children[0].leaf_count = 2;
        branch.children[1].leaf_count = 2;
        wrong_count
            .accept(
                &root,
                InventoryNodeResponse::Found(InventoryNode::Branch {
                    digest,
                    leaf_count,
                    branch,
                }),
                |_, _| false,
            )
            .unwrap();
        let child = wrong_count.next_request(|_, _| None).unwrap().unwrap();
        assert!(
            wrong_count
                .accept(&child, response(&remote, &child.prefix), |_, _| false)
                .is_err()
        );
        assert!(wrong_count.finish().is_err());
    }
}
