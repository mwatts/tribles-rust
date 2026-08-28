# Incremental Queries

The query engine normally evaluates a pattern over a complete `TribleSet`.
Continuous ingest pipelines often need only the solutions which involve at
least one newly added fact. `pattern_changes!` implements that semi-naive delta
evaluation without tying the query engine to any storage or history model.

## Supply full facts and one delta

Maintain two sets:

- `full`, which contains every fact visible for the current observation; and
- `changed`, the subset added since the previous observation.

`full` must include `changed` before the query runs:

```rust,ignore
let mut full = initial;
let changed = entity! {
    &messiah @
    literature::title: "Dune Messiah",
    literature::author: &herbert,
}.into_facts();
full += &changed;

let new_titles: Vec<String> = find!(
    title: String,
    pattern_changes!(&full, &changed, [
        { _?author @ literature::firstname: "Frank" },
        { _?book @
            literature::author: _?author,
            literature::title: ?title
        }
    ])
)
.collect();
```

The complete runnable version is deliberately storage-independent:

```rust,ignore
{{#include ../../examples/pattern_changes.rs:pattern_changes_example}}
```

Applications may obtain `changed` directly from an importer, an event batch, a
collection-ticket difference, or any other source. The query algorithm only
needs the two fact sets.

## How semi-naive evaluation works

For a pattern with several trible constraints, the macro evaluates one variant
per constraint:

1. that constraint reads from `changed`;
2. every other constraint reads from `full`; and
3. the variants are combined with `or!`.

Every returned binding therefore has at least one witness in the delta. Work
scales with the changed set and the number of trible constraints instead of
requiring each constraint to scan the complete dataset.

The union constraint deduplicates proposed candidate values at each search
level, so the same complete binding supported by several variants is enumerated
once per invocation. The usual `find!` projection semantics still apply:
hidden variables can create multiple rows when genuinely distinct complete
bindings project to the same tuple. Collect projected results into a set when
the application wants projection-level uniqueness.

Nothing is remembered between invocations. A later delta may return a tuple
seen earlier when a new fact supplies another proof. Applications that require
global once-only delivery retain the projected tuples they have consumed;
applications interested in witness events should project the witness identity.

## Use collection tickets as continuation tokens

A native collection ticket is the exact commit set observed by one discovery
pass. Its complete signed records make a natural storage-level continuation
token. [`exact_ticket_additions`](triblespace::core::collection::exact_ticket_additions)
compares two observations, verifies that the earlier support remains a subset
of the later support, and returns only the newly observed commits:

```rust,ignore
{{#include ../../examples/collection_pattern_changes.rs:collection_pattern_changes_observe}}
```

This computes a support-set difference over immutable records; it does not walk
a parent chain, inspect a physical merge cover, or ask an ambient head what
changed. If a previous member is absent, the helper returns
`ExactTicketAdvanceError::ResetRequired`: additions-only maintenance is no
longer sound, so rebuild the accumulated application state from `current`.
Advance the saved ticket only after the complete fallible fold succeeds, as the
example does, to make a failed fold retry the same support.

The two pattern inputs need not share a representation. The runnable example
(`cargo run --example collection_pattern_changes`) keeps a
`SuccinctArchiveCollection::exact_view()`. An unchanged ticket reuses its
owned, already-admitted immutable archive without storage access. For a strict
extension, the view runs ordinary exact admission only over the added commits
and unions those shards with its previous archive; a shrinking observation
rebuilds. The independent consumption checkpoint still controls the cheap
`SimpleArchive`-backed change set, so a failed consumer retries the same delta
even though constructing the full view already succeeded. Exact tickets ensure
that commits first observed after `current` cannot leak into either input merely
because their blobs are already resident.

Commit support is deliberately not an exact fact difference. A new commit may
repeat a fact already present, and that new witness may legitimately make a
projected result recur. Consumers requiring global once-only delivery retain
their consumed result identities independently.

When an ingestion API already returns its newly produced fragment, using that
fragment's facts directly is cheaper than rematerializing a ticket subset. The
ticket pattern is useful across process boundaries or after reopening storage.

## Monotonicity and CALM

Removed results are not tracked. Facts and collection commits are monotone:
new input can add witnesses but does not invalidate a previous conclusion.
This is the [CALM principle](https://arxiv.org/abs/1901.01930) in executable
form—monotone results can be distributed without consensus.

When a domain needs versions or supersession, represent those relationships as
facts and query the explicit DAG. Do not infer a winner from insertion order.

## Trade-offs

- The caller supplies the changed set; the query engine keeps no hidden cursor.
- Each trible constraint adds one query variant, so selective constraints keep
  delta evaluation efficient.
- A changed set which grows without bound loses its advantage; advance the
  continuation after successful consumption.
- A result may recur in a later invocation when the later delta provides a new
  witness.
