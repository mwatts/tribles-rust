# Triblespace checkout metadata

Commits store a `content` handle plus an optional `metadata` handle, each
pointing to a SimpleArchive TribleSet. `Workspace::checkout` unions the
`content` TribleSet for the selected commits. `Workspace::checkout_metadata`
unions the referenced metadata TribleSets. `Workspace::checkout_with_metadata`
returns both in one pass as `(content, metadata)`.

Commits without a metadata handle contribute an empty metadata TribleSet. To
attach metadata at write time, pass `Some(metadata)` to `Workspace::commit`
(or use the lower-level `repo::commit::commit_metadata` helper). You can also
configure a workspace default metadata TribleSet with
`Workspace::set_default_metadata`, which `Workspace::commit` will attach
whenever `metadata` is `None`. Supplying metadata does not change the default.
`Repository::set_default_metadata` seeds new workspaces created via `pull` with
a default metadata handle. Use `Repository::pull_with_metadata` to override
the workspace default for a specific pull. Defaults are runtime configuration
and are not stored in branch metadata.
If you need to seed metadata blobs directly in the repository, borrow the
underlying blob store via `Repository::storage_mut` before calling
`set_default_metadata`.
