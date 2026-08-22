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
pass. Its intrinsic commit IDs make a natural storage-level continuation token:

```rust,ignore
use std::collections::BTreeSet;
use triblespace::core::collection::reach;

let current = collection.ticket()?;
let new_commits: Vec<_> = current
    .iter()
    .copied()
    .filter(|commit| !seen.contains(&commit.id()))
    .collect();

let view = SimpleArchiveCollection::new(
    name.clone(),
    team,
    reach::private(),
);
let changed = view.attach_exact(collection.storage_mut(), &new_commits)?;
let full = view.attach_exact(collection.storage_mut(), &current)?;

for row in find!(result, pattern_changes!(&full, &changed, [
    /* constraints */
])) {
    // consume row
}

seen = current.iter().map(|commit| commit.id()).collect::<BTreeSet<_>>();
```

This computes a set difference over immutable records; it does not walk a
parent chain or ask an ambient head what changed. Exact-ticket attachment also
ensures that commits first observed after `current` cannot leak into either
fact view merely because their blobs are already resident.

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
