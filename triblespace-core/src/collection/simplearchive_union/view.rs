//! Logical fact views over typed `SimpleArchive` covers.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;

use crate::blob::encodings::simplearchive::{SimpleArchive, UnarchiveError};
use crate::blob::Blob;
use crate::collection::{CollectionData, Cover, TryFromCover, TryFromCoverError};
use crate::inline::encodings::hash::Handle;
use crate::repo::BlobStoreGet;
use crate::trible::{Fragment, TribleSet};

use super::join_many;

/// Failure to form a logical fact union from an already validated exact cover.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactViewError {
    /// Member whose canonical bytes failed to decode.
    pub member: CollectionData,
    /// Exact SimpleArchive decoding failure.
    pub source: UnarchiveError,
}

impl fmt::Display for FactViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "collection member {} could not form the fact view: {}",
            hex::encode_upper(self.member.raw),
            self.source,
        )
    }
}

impl Error for FactViewError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

impl TryFromCover<SimpleArchive> for TribleSet {
    type Error = FactViewError;

    fn try_from_cover<R>(
        cover: &Cover<SimpleArchive>,
        _descriptor: &Fragment,
        snapshot: &R,
    ) -> Result<Self, TryFromCoverError<R::GetError<Infallible>, Self::Error>>
    where
        R: BlobStoreGet,
    {
        let mut members = Vec::with_capacity(cover.len());
        for handle in cover.members() {
            let member = Handle::<SimpleArchive>::to_hash(handle);
            let blob = snapshot
                .get::<Blob<SimpleArchive>, SimpleArchive>(handle)
                .map_err(|source| TryFromCoverError::MemberGet { member, source })?;
            members.push((handle, blob));
        }
        match members.as_slice() {
            [] => Ok(TribleSet::new()),
            [(handle, blob)] => blob.clone().try_from_blob().map_err(|source| {
                TryFromCoverError::View(FactViewError {
                    member: Handle::<SimpleArchive>::to_hash(*handle),
                    source,
                })
            }),
            _ => {
                let union = join_many(members.iter().map(|(_, blob)| blob)).map_err(
                    |(index, source)| {
                        TryFromCoverError::View(FactViewError {
                            member: Handle::<SimpleArchive>::to_hash(members[index].0),
                            source,
                        })
                    },
                )?;
                union.try_from_blob().map_err(|source| {
                    TryFromCoverError::View(FactViewError {
                        member: Handle::<SimpleArchive>::to_hash(members[0].0),
                        source,
                    })
                })
            }
        }
    }
}
