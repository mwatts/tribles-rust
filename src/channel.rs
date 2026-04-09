//! Channel types bridging the async Host thread and the sync store layer.
//!
//! `NetCommand`: sent from Leader → Host (outgoing effects)
//! `NetEvent`: sent from Host → Follower (incoming data)

use crate::protocol::{RawHash, RawBranchId};

/// Commands sent to the Host thread.
#[derive(Debug)]
pub enum NetCommand {
    /// Announce a blob hash to the DHT.
    Announce(RawHash),
    /// Gossip a HEAD change for a branch.
    Gossip { branch: RawBranchId, head: RawHash },
    /// Request sync: fetch blobs reachable from a remote branch head.
    Fetch { peer: iroh_base::EndpointId, branch: RawBranchId },
}

/// Events received from the Host thread.
#[derive(Debug)]
pub enum NetEvent {
    /// A blob was fetched from the network.
    Blob(Vec<u8>),
    /// A remote branch HEAD was learned (via gossip or fetch).
    Head { branch: RawBranchId, head: RawHash },
}
