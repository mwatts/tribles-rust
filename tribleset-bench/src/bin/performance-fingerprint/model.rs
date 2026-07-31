//! In-memory adapter from the demand-curve TSV receipt to the fingerprint
//! matrix. The adapter is deliberately independent of any pile schema: the
//! benchmark receipt remains canonical, while this module owns only a small
//! presentation model.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SubjectId {
    pub engine: String,
    pub variant: String,
    pub backend: String,
    pub substrate: String,
    pub parallelism: String,
}

impl SubjectId {
    pub fn label(&self) -> String {
        format!(
            "{} ({}) — {} / {} / {}",
            self.engine,
            self.variant,
            title_component(&self.backend),
            title_component(&self.substrate),
            title_component(&self.parallelism),
        )
    }
}

fn title_component(value: &str) -> String {
    match value.to_ascii_lowercase().as_str() {
        "tribleset" => "TribleSet".to_owned(),
        "succinct" => "Succinct".to_owned(),
        "cpu" => "CPU".to_owned(),
        "wgpu" => "WGPU".to_owned(),
        "rayon" => "Rayon".to_owned(),
        "sequential" => "sequential".to_owned(),
        _ => value.to_owned(),
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Demand {
    Construct,
    Rows(u64),
    Full,
}

impl Demand {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "construct" | "setup" => Ok(Self::Construct),
            "full" | "drain" => Ok(Self::Full),
            value => value
                .parse::<u64>()
                .ok()
                .filter(|rows| *rows > 0)
                .map(Self::Rows)
                .ok_or_else(|| format!("invalid demand {raw:?}")),
        }
    }

    pub fn label(self) -> String {
        match self {
            Self::Construct => "setup".to_owned(),
            Self::Rows(rows) => compact_u64(rows),
            Self::Full => "full".to_owned(),
        }
    }

    pub fn is_curve_point(self) -> bool {
        !matches!(self, Self::Construct)
    }
}

fn compact_u64(value: u64) -> String {
    for (divisor, suffix) in [
        (1_000_000_000_u64, "G"),
        (1_000_000_u64, "M"),
        (1_000_u64, "k"),
    ] {
        if value >= divisor && value.is_multiple_of(divisor) {
            return format!("{}{}", value / divisor, suffix);
        }
    }
    value.to_string()
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CellKey {
    pub shape: String,
    pub scale: String,
    pub demand: Demand,
}

#[derive(Clone, Debug)]
pub struct MeasuredCell {
    pub median_elapsed_ns: f64,
    /// `None` only for setup/construct, where c(0) is undefined.
    pub ns_per_result: Option<f64>,
    pub observations: usize,
    pub rows: u64,
    pub execution_paths: BTreeSet<String>,
}

#[derive(Clone, Debug)]
pub enum CellStatus {
    Measured(MeasuredCell),
    Missing(String),
    /// An explicit producer status, whose reason belongs in diagnostics.
    Unsupported(String),
    /// A demand outside the producer protocol. It renders under the public
    /// `unsupported` category without flooding the diagnostic ledger.
    NotApplicable(String),
    Error(String),
    CardinalityMismatch(String),
}

impl CellStatus {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Measured(_) => "measured",
            Self::Missing(_) => "missing",
            Self::Unsupported(_) | Self::NotApplicable(_) => "unsupported",
            Self::Error(_) => "error",
            Self::CardinalityMismatch(_) => "cardinality mismatch",
        }
    }
}

#[derive(Clone, Debug)]
pub struct SubjectPanel {
    pub id: SubjectId,
    cells: BTreeMap<CellKey, CellStatus>,
}

impl SubjectPanel {
    pub fn cell(&self, shape: &str, scale: &str, demand: Demand) -> &CellStatus {
        self.cells
            .get(&CellKey {
                shape: shape.to_owned(),
                scale: scale.to_owned(),
                demand,
            })
            .expect("the report materializes the complete matrix")
    }

    pub fn status_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts = BTreeMap::new();
        for status in self.cells.values() {
            *counts.entry(status.name()).or_default() += 1;
        }
        counts
    }

    pub fn diagnostic_cells(&self) -> impl Iterator<Item = (&CellKey, &CellStatus)> {
        self.cells.iter().filter(|(_, status)| {
            matches!(
                status,
                CellStatus::Missing(_)
                    | CellStatus::Unsupported(_)
                    | CellStatus::Error(_)
                    | CellStatus::CardinalityMismatch(_)
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct InputIssue {
    pub line: Option<usize>,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct FingerprintReport {
    pub subjects: Vec<SubjectPanel>,
    pub shapes: Vec<String>,
    pub scales: Vec<String>,
    pub demands: Vec<Demand>,
    pub issues: Vec<InputIssue>,
    pub input_rows: usize,
    pub samples: usize,
}

#[derive(Clone, Copy, Debug)]
enum RecordKind {
    Sample,
    Identity,
    Work,
    Status,
}

#[derive(Clone, Debug)]
enum ExplicitStatus {
    Unsupported(String),
    Error(String),
    CardinalityMismatch(String),
}

#[derive(Clone, Debug)]
struct TelemetryRow {
    subject: SubjectId,
    shape: String,
    scale: String,
    demand: Demand,
    record: RecordKind,
    elapsed_ns: Option<u64>,
    rows: Option<u64>,
    digest: Option<String>,
    harness: Option<String>,
    corpus: Option<String>,
    execution_path: Option<String>,
    abba_position: Option<String>,
    repetition: Option<u64>,
    explicit: Option<ExplicitStatus>,
}

#[derive(Default)]
struct RawCell {
    samples: Vec<Sample>,
    explicit: Vec<ExplicitStatus>,
    provenance: BTreeSet<(String, String)>,
}

#[derive(Clone, Debug)]
struct Sample {
    elapsed_ns: u64,
    rows: u64,
    execution_path: Option<String>,
    slot: Option<ObservationSlot>,
}

#[derive(Default)]
struct Identity {
    rows: BTreeSet<u64>,
    digests: BTreeSet<String>,
    positions: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ObservationSlot {
    abba_position: String,
    repetition: u64,
}

type IdentityKey = (SubjectId, String, String);
type LogicalKey = (String, String);

impl FingerprintReport {
    pub fn from_tsv(input: &str) -> Result<Self, String> {
        let mut lines = input.lines();
        let header_line = lines
            .next()
            .ok_or_else(|| "the receipt is empty".to_owned())?;
        let header = Header::new(header_line)?;
        let mut issues = header.issues();
        let mut rows = Vec::new();
        let mut input_rows = 0usize;

        for (offset, line) in lines.enumerate() {
            let line_number = offset + 2;
            if line.trim().is_empty() {
                continue;
            }
            input_rows += 1;
            match header.parse_row(line, line_number, &mut issues) {
                Some(row) => rows.push(row),
                None => continue,
            }
        }

        if rows.is_empty() {
            return Err(format!(
                "the receipt contains no adaptable rows ({} input issues)",
                issues.len()
            ));
        }
        Ok(Self::from_rows(rows, issues, input_rows))
    }

    fn from_rows(rows: Vec<TelemetryRow>, issues: Vec<InputIssue>, input_rows: usize) -> Self {
        let mut subjects = BTreeSet::new();
        let mut shapes = BTreeSet::new();
        let mut scales = BTreeSet::new();
        let mut demands = BTreeSet::new();
        let mut cells: BTreeMap<(SubjectId, CellKey), RawCell> = BTreeMap::new();
        let mut identities: BTreeMap<IdentityKey, Identity> = BTreeMap::new();
        let mut logical_identities: BTreeMap<LogicalKey, Identity> = BTreeMap::new();
        let mut expected_slots: BTreeMap<SubjectId, BTreeSet<ObservationSlot>> = BTreeMap::new();
        let mut identity_protocol_subjects = BTreeSet::new();
        let mut sample_count = 0usize;

        for row in rows {
            subjects.insert(row.subject.clone());
            shapes.insert(row.shape.clone());
            scales.insert(row.scale.clone());
            demands.insert(row.demand);
            let cell_key = CellKey {
                shape: row.shape.clone(),
                scale: row.scale.clone(),
                demand: row.demand,
            };
            let raw = cells.entry((row.subject.clone(), cell_key)).or_default();
            if let (Some(harness), Some(corpus)) = (row.harness.as_ref(), row.corpus.as_ref()) {
                raw.provenance.insert((harness.clone(), corpus.clone()));
            }
            if let Some(status) = row.explicit {
                raw.explicit.push(status);
            }

            let observation_slot = row.abba_position.as_ref().zip(row.repetition).map(
                |(abba_position, repetition)| ObservationSlot {
                    abba_position: abba_position.clone(),
                    repetition,
                },
            );
            if matches!(row.record, RecordKind::Sample) {
                if let Some(slot) = observation_slot.as_ref() {
                    expected_slots
                        .entry(row.subject.clone())
                        .or_default()
                        .insert(slot.clone());
                }
            }

            match row.record {
                RecordKind::Sample => match (row.elapsed_ns, row.rows) {
                    (Some(elapsed_ns), Some(rows)) => {
                        sample_count += 1;
                        raw.samples.push(Sample {
                            elapsed_ns,
                            rows,
                            execution_path: row.execution_path,
                            slot: observation_slot,
                        });
                    }
                    _ => raw.explicit.push(ExplicitStatus::Error(
                        "sample is missing elapsed_ns or rows".to_owned(),
                    )),
                },
                RecordKind::Identity => {
                    if let Some(position) = row.abba_position.as_ref() {
                        identity_protocol_subjects.insert(row.subject.clone());
                        identities
                            .entry((row.subject.clone(), row.shape.clone(), row.scale.clone()))
                            .or_default()
                            .positions
                            .insert(position.clone());
                    }
                    let identity_key = (row.subject.clone(), row.shape.clone(), row.scale.clone());
                    let logical_key = (row.shape, row.scale);
                    if let Some(expected) = row.rows {
                        identities
                            .entry(identity_key)
                            .or_default()
                            .rows
                            .insert(expected);
                        logical_identities
                            .entry(logical_key.clone())
                            .or_default()
                            .rows
                            .insert(expected);
                    }
                    if let Some(digest) = row.digest.filter(|value| value != "-") {
                        identities
                            .entry((row.subject, logical_key.0.clone(), logical_key.1.clone()))
                            .or_default()
                            .digests
                            .insert(digest.clone());
                        logical_identities
                            .entry(logical_key)
                            .or_default()
                            .digests
                            .insert(digest);
                    }
                }
                RecordKind::Work | RecordKind::Status => {}
            }
        }

        let mut shapes: Vec<String> = shapes.into_iter().collect();
        shapes.sort_by_key(|shape| shape_order(shape));
        let mut scales: Vec<String> = scales.into_iter().collect();
        scales.sort_by_key(|scale| scale_order(scale));
        let demands: Vec<Demand> = demands.into_iter().collect();

        let mut panels = Vec::new();
        for subject in subjects {
            let mut panel_cells = BTreeMap::new();
            for shape in &shapes {
                for scale in &scales {
                    for &demand in &demands {
                        let key = CellKey {
                            shape: shape.clone(),
                            scale: scale.clone(),
                            demand,
                        };
                        let status = match cells.get(&(subject.clone(), key.clone())) {
                            Some(raw) => classify_cell(
                                raw,
                                demand,
                                identities.get(&(subject.clone(), shape.clone(), scale.clone())),
                                logical_identities.get(&(shape.clone(), scale.clone())),
                                expected_slots.get(&subject),
                                identity_protocol_subjects.contains(&subject),
                            ),
                            None => classify_absent_cell(
                                &subject,
                                demand,
                                identities.get(&(subject.clone(), shape.clone(), scale.clone())),
                            ),
                        };
                        panel_cells.insert(key, status);
                    }
                }
            }
            panels.push(SubjectPanel {
                id: subject,
                cells: panel_cells,
            });
        }

        Self {
            subjects: panels,
            shapes,
            scales,
            demands,
            issues,
            input_rows,
            samples: sample_count,
        }
    }
}

fn classify_absent_cell(
    subject: &SubjectId,
    demand: Demand,
    identity: Option<&Identity>,
) -> CellStatus {
    if matches!(demand, Demand::Construct) && subject.parallelism.eq_ignore_ascii_case("rayon") {
        return CellStatus::NotApplicable(
            "the parallel protocol does not report construction as a separate demand".to_owned(),
        );
    }

    if let Demand::Rows(requested) = demand {
        if let Some(full_rows) = identity
            .filter(|identity| identity.rows.len() == 1)
            .and_then(|identity| identity.rows.iter().next())
        {
            if requested >= *full_rows {
                return CellStatus::NotApplicable(format!(
                    "{requested} rows is at or beyond the {full_rows}-row full drain"
                ));
            }
        }
    }

    CellStatus::Missing(
        "no observation or explicit producer status was present for this expected matrix cell"
            .to_owned(),
    )
}

fn classify_cell(
    raw: &RawCell,
    demand: Demand,
    identity: Option<&Identity>,
    logical_identity: Option<&Identity>,
    expected_slots: Option<&BTreeSet<ObservationSlot>>,
    identity_protocol: bool,
) -> CellStatus {
    let join_details = |items: Vec<&String>| {
        items
            .into_iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("; ")
    };
    let mismatches: Vec<&String> = raw
        .explicit
        .iter()
        .filter_map(|status| match status {
            ExplicitStatus::CardinalityMismatch(detail) => Some(detail),
            _ => None,
        })
        .collect();
    if !mismatches.is_empty() {
        return CellStatus::CardinalityMismatch(join_details(mismatches));
    }
    let errors: Vec<&String> = raw
        .explicit
        .iter()
        .filter_map(|status| match status {
            ExplicitStatus::Error(detail) => Some(detail),
            _ => None,
        })
        .collect();
    if !errors.is_empty() {
        return CellStatus::Error(join_details(errors));
    }
    let unsupported: Vec<&String> = raw
        .explicit
        .iter()
        .filter_map(|status| match status {
            ExplicitStatus::Unsupported(detail) => Some(detail),
            _ => None,
        })
        .collect();
    if !unsupported.is_empty() {
        return CellStatus::Unsupported(join_details(unsupported));
    }
    if raw.samples.is_empty() {
        return CellStatus::Missing(
            "the cell has metadata or identity evidence, but no timing samples".to_owned(),
        );
    }
    if raw.provenance.len() > 1 {
        return CellStatus::Error(format!(
            "mixed {} harness/corpus pairs in one cell",
            raw.provenance.len()
        ));
    }

    if let Some(expected_slots) = expected_slots.filter(|slots| !slots.is_empty()) {
        let observed_slots: BTreeSet<ObservationSlot> = raw
            .samples
            .iter()
            .filter_map(|sample| sample.slot.clone())
            .collect();
        if observed_slots != *expected_slots {
            let missing = expected_slots.difference(&observed_slots).count();
            let extra = observed_slots.difference(expected_slots).count();
            return CellStatus::Missing(format!(
                "incomplete observation grid: expected {} ABBA/repetition slots, observed {}; {missing} missing, {extra} unexpected",
                expected_slots.len(),
                observed_slots.len(),
            ));
        }
        if raw.samples.len() != observed_slots.len() {
            return CellStatus::Error(format!(
                "duplicate observation slots: {} samples occupy {} ABBA/repetition slots",
                raw.samples.len(),
                observed_slots.len(),
            ));
        }

        if matches!(demand, Demand::Full) && identity_protocol {
            let expected_positions: BTreeSet<&str> = expected_slots
                .iter()
                .map(|slot| slot.abba_position.as_str())
                .collect();
            let observed_positions: BTreeSet<&str> = identity
                .into_iter()
                .flat_map(|identity| identity.positions.iter().map(String::as_str))
                .collect();
            if observed_positions != expected_positions {
                return CellStatus::Missing(format!(
                    "incomplete identity grid: expected {} ABBA positions, observed {}",
                    expected_positions.len(),
                    observed_positions.len(),
                ));
            }
        }
    }

    let observed_rows: BTreeSet<u64> = raw.samples.iter().map(|sample| sample.rows).collect();
    let expected = match demand {
        Demand::Construct => Some(0),
        Demand::Rows(rows) => Some(rows),
        Demand::Full => {
            if let Some(identity) = identity {
                if identity.rows.len() > 1 {
                    return CellStatus::CardinalityMismatch(format!(
                        "subject identities disagree: {:?}",
                        identity.rows
                    ));
                }
            }
            if let Some(logical) = logical_identity {
                if logical.rows.len() > 1 {
                    return CellStatus::CardinalityMismatch(format!(
                        "subjects disagree on full cardinality: {:?}",
                        logical.rows
                    ));
                }
                if logical.digests.len() > 1 {
                    return CellStatus::CardinalityMismatch(
                        "subjects returned different full-result digests".to_owned(),
                    );
                }
            }
            identity
                .and_then(|value| value.rows.iter().next().copied())
                .or_else(|| observed_rows.iter().next().copied())
        }
    };

    if observed_rows.len() != 1 || expected.is_none() || !observed_rows.contains(&expected.unwrap())
    {
        return CellStatus::CardinalityMismatch(format!(
            "expected {} rows, observed {:?}",
            expected
                .map(|rows| rows.to_string())
                .unwrap_or_else(|| "one stable cardinality".to_owned()),
            observed_rows
        ));
    }
    let rows = expected.expect("checked above");
    if matches!(demand, Demand::Full) && rows == 0 {
        return CellStatus::Error(
            "full drain returned zero rows, so c(full) is undefined".to_owned(),
        );
    }

    let median_elapsed_ns = median(raw.samples.iter().map(|sample| sample.elapsed_ns));
    let divisor = match demand {
        Demand::Construct => None,
        Demand::Rows(rows) => Some(rows),
        Demand::Full => Some(rows),
    };
    let execution_paths = raw
        .samples
        .iter()
        .filter_map(|sample| sample.execution_path.clone())
        .collect();
    CellStatus::Measured(MeasuredCell {
        median_elapsed_ns,
        ns_per_result: divisor.map(|value| median_elapsed_ns / value as f64),
        observations: raw.samples.len(),
        rows,
        execution_paths,
    })
}

fn median(values: impl Iterator<Item = u64>) -> f64 {
    let mut values: Vec<u64> = values.collect();
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] as f64 + values[middle] as f64) / 2.0
    } else {
        values[middle] as f64
    }
}

fn shape_order(shape: &str) -> (usize, String) {
    let rank = match shape {
        "unique_lookup" => 0,
        "bound_star" => 1,
        "parent_batch_confirm" => 2,
        "nested_and_or" => 3,
        _ => usize::MAX,
    };
    (rank, shape.to_owned())
}

fn scale_order(scale: &str) -> (usize, String) {
    let rank = match scale {
        "tiny" => 0,
        "below" => 1,
        "threshold" => 2,
        "above" => 3,
        "wide" => 4,
        _ => usize::MAX,
    };
    (rank, scale.to_owned())
}

struct Header {
    names: BTreeMap<String, usize>,
    width: usize,
}

impl Header {
    fn new(line: &str) -> Result<Self, String> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            return Err("expected a tab-separated header".to_owned());
        }
        let mut names = BTreeMap::new();
        for (index, name) in fields.iter().enumerate() {
            let normalized = normalize_header(name);
            if names.insert(normalized.clone(), index).is_some() {
                return Err(format!("duplicate header {normalized:?}"));
            }
        }
        Ok(Self {
            names,
            width: fields.len(),
        })
    }

    fn issues(&self) -> Vec<InputIssue> {
        let mut issues = Vec::new();
        for (axis, aliases) in [
            ("engine", ENGINE),
            ("backend/storage", BACKEND),
            ("substrate/compute", SUBSTRATE),
            ("parallelism/execution mode", PARALLELISM),
        ] {
            if self.index(aliases).is_none() {
                issues.push(InputIssue {
                    line: None,
                    message: format!("missing subject axis {axis}; using 'unspecified'"),
                });
            }
        }
        issues
    }

    fn index(&self, aliases: &[&str]) -> Option<usize> {
        aliases
            .iter()
            .find_map(|name| self.names.get(*name).copied())
    }

    fn value<'a>(&self, fields: &'a [&str], aliases: &[&str]) -> Option<&'a str> {
        let value = fields.get(self.index(aliases)?)?.trim();
        (!value.is_empty() && value != "-").then_some(value)
    }

    fn parse_row(
        &self,
        line: &str,
        line_number: usize,
        issues: &mut Vec<InputIssue>,
    ) -> Option<TelemetryRow> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != self.width {
            issues.push(InputIssue {
                line: Some(line_number),
                message: format!(
                    "column count mismatch: expected {}, found {}",
                    self.width,
                    fields.len()
                ),
            });
            return None;
        }

        let required = |aliases: &[&str], name: &str, issues: &mut Vec<InputIssue>| {
            let value = self.value(&fields, aliases).map(str::to_owned);
            if value.is_none() {
                issues.push(InputIssue {
                    line: Some(line_number),
                    message: format!("missing {name}"),
                });
            }
            value
        };
        let shape = required(SHAPE, "query shape", issues)?;
        let scale = required(SCALE, "scale", issues)?;
        let demand_raw = required(DEMAND, "demand", issues)?;
        let demand = match Demand::parse(&demand_raw) {
            Ok(demand) => demand,
            Err(message) => {
                issues.push(InputIssue {
                    line: Some(line_number),
                    message,
                });
                return None;
            }
        };

        let raw_record = self
            .value(&fields, RECORD)
            .unwrap_or("sample")
            .to_ascii_lowercase();
        let raw_status = self.value(&fields, STATUS).map(str::to_owned);
        let mut explicit = raw_status.as_deref().and_then(classify_explicit_status);
        let record = match raw_record.as_str() {
            "sample" | "measurement" => RecordKind::Sample,
            "identity" | "oracle" => RecordKind::Identity,
            "work" | "diagnostic" => RecordKind::Work,
            "unsupported" | "skip" => {
                explicit.get_or_insert_with(|| {
                    ExplicitStatus::Unsupported(
                        raw_status.clone().unwrap_or_else(|| raw_record.clone()),
                    )
                });
                RecordKind::Status
            }
            "error" | "panic" => {
                explicit.get_or_insert_with(|| {
                    ExplicitStatus::Error(raw_status.clone().unwrap_or_else(|| raw_record.clone()))
                });
                RecordKind::Status
            }
            "cardinality_mismatch" | "mismatch" => {
                explicit.get_or_insert_with(|| {
                    ExplicitStatus::CardinalityMismatch(
                        raw_status.clone().unwrap_or_else(|| raw_record.clone()),
                    )
                });
                RecordKind::Status
            }
            other => {
                issues.push(InputIssue {
                    line: Some(line_number),
                    message: format!("unknown record kind {other:?}"),
                });
                explicit.get_or_insert_with(|| {
                    ExplicitStatus::Error(format!("unknown record kind {other}"))
                });
                RecordKind::Status
            }
        };

        let elapsed_ns = parse_optional_u64(
            self.value(&fields, ELAPSED_NS),
            "elapsed_ns",
            line_number,
            issues,
        );
        let rows = parse_optional_u64(self.value(&fields, ROWS), "rows", line_number, issues);
        let repetition = parse_optional_u64(
            self.value(&fields, REPETITION),
            "repetition",
            line_number,
            issues,
        );

        Some(TelemetryRow {
            subject: SubjectId {
                engine: self
                    .value(&fields, ENGINE)
                    .unwrap_or("unspecified")
                    .to_owned(),
                variant: self.value(&fields, VARIANT).unwrap_or("default").to_owned(),
                backend: self
                    .value(&fields, BACKEND)
                    .unwrap_or("unspecified")
                    .to_owned(),
                substrate: self
                    .value(&fields, SUBSTRATE)
                    .unwrap_or("unspecified")
                    .to_owned(),
                parallelism: self
                    .value(&fields, PARALLELISM)
                    .unwrap_or("unspecified")
                    .to_owned(),
            },
            shape,
            scale,
            demand,
            record,
            elapsed_ns,
            rows,
            digest: self.value(&fields, DIGEST).map(str::to_owned),
            harness: self.value(&fields, HARNESS).map(str::to_owned),
            corpus: self.value(&fields, CORPUS).map(str::to_owned),
            execution_path: self.value(&fields, EXECUTION_PATH).map(str::to_owned),
            abba_position: self.value(&fields, ABBA_POSITION).map(str::to_owned),
            repetition,
            explicit,
        })
    }
}

fn parse_optional_u64(
    value: Option<&str>,
    field: &str,
    line: usize,
    issues: &mut Vec<InputIssue>,
) -> Option<u64> {
    let raw = value?;
    match raw.parse::<u64>() {
        Ok(value) => Some(value),
        Err(_) => {
            issues.push(InputIssue {
                line: Some(line),
                message: format!("invalid {field} {raw:?}"),
            });
            None
        }
    }
}

fn classify_explicit_status(raw: &str) -> Option<ExplicitStatus> {
    let lower = raw.trim().to_ascii_lowercase();
    if lower.is_empty() || matches!(lower.as_str(), "ok" | "signal" | "success" | "measured") {
        None
    } else if lower.starts_with("skip") || lower.starts_with("unsupported") {
        Some(ExplicitStatus::Unsupported(raw.to_owned()))
    } else if lower.contains("cardinality")
        || lower.starts_with("mismatch")
        || lower.starts_with("gate_fail:rows")
        || lower.starts_with("gate_fail:identity")
    {
        Some(ExplicitStatus::CardinalityMismatch(raw.to_owned()))
    } else {
        Some(ExplicitStatus::Error(raw.to_owned()))
    }
}

fn normalize_header(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

const ENGINE: &[&str] = &["engine", "core_commit", "commit", "subject"];
const VARIANT: &[&str] = &["engine_variant", "variant"];
const BACKEND: &[&str] = &["backend", "storage"];
const SUBSTRATE: &[&str] = &["substrate", "compute", "device"];
const PARALLELISM: &[&str] = &["parallelism", "execution_mode", "mode"];
const SHAPE: &[&str] = &["shape", "query_shape", "query", "workload"];
const SCALE: &[&str] = &["scale", "dataset_scale"];
const DEMAND: &[&str] = &["demand", "limit", "requested_rows"];
const RECORD: &[&str] = &["record", "kind"];
const ELAPSED_NS: &[&str] = &["elapsed_ns", "duration_ns", "time_ns"];
const ROWS: &[&str] = &["rows", "cardinality", "result_rows"];
const DIGEST: &[&str] = &["result_digest", "digest"];
const STATUS: &[&str] = &["status", "outcome"];
const HARNESS: &[&str] = &["harness", "harness_sha256"];
const CORPUS: &[&str] = &["corpus", "corpus_digest"];
const EXECUTION_PATH: &[&str] = &["execution_path", "route"];
const ABBA_POSITION: &[&str] = &["abba_position", "position"];
const REPETITION: &[&str] = &["repetition", "rep"];

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "record\tengine\tengine_variant\tharness\tcorpus\tscale\tbackend\tsubstrate\tparallelism\texecution_path\tshape\tdemand\telapsed_ns\trows\tresult_digest\tstatus";

    fn parse(lines: &[&str]) -> FingerprintReport {
        let mut input = HEADER.to_owned();
        for line in lines {
            input.push('\n');
            input.push_str(line);
        }
        FingerprintReport::from_tsv(&input).unwrap()
    }

    fn row(
        record: &str,
        engine: &str,
        backend: &str,
        substrate: &str,
        parallelism: &str,
        shape: &str,
        scale: &str,
        demand: &str,
        elapsed: &str,
        rows: &str,
        digest: &str,
        status: &str,
    ) -> String {
        format!(
            "{record}\t{engine}\tdefault\th1\tc1\t{scale}\t{backend}\t{substrate}\t{parallelism}\tcpu\t{shape}\t{demand}\t{elapsed}\t{rows}\t{digest}\t{status}"
        )
    }

    #[test]
    fn computes_median_cost_and_distinct_full_drain() {
        let rows = [
            row(
                "identity",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "0",
                "4",
                "same",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "2",
                "100",
                "2",
                "-",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "2",
                "300",
                "2",
                "-",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "800",
                "4",
                "-",
                "",
            ),
        ];
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let report = parse(&refs);
        let panel = &report.subjects[0];
        let CellStatus::Measured(two) = panel.cell("q", "tiny", Demand::Rows(2)) else {
            panic!("numeric demand should be measured")
        };
        assert_eq!(two.median_elapsed_ns, 200.0);
        assert_eq!(two.ns_per_result, Some(100.0));
        let CellStatus::Measured(full) = panel.cell("q", "tiny", Demand::Full) else {
            panic!("full demand should be measured")
        };
        assert_eq!(full.ns_per_result, Some(200.0));
        assert_eq!(report.demands, vec![Demand::Rows(2), Demand::Full]);
    }

    #[test]
    fn materializes_missing_and_explicit_failure_cells() {
        let rows = [
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "1",
                "10",
                "1",
                "-",
                "",
            ),
            row(
                "unsupported",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "-",
                "-",
                "-",
                "skip:gpu",
            ),
            row(
                "error",
                "e2",
                "succinct",
                "wgpu",
                "rayon",
                "q",
                "tiny",
                "1",
                "-",
                "-",
                "-",
                "panic:adapter",
            ),
        ];
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let report = parse(&refs);
        assert_eq!(report.subjects.len(), 2);
        let first = report
            .subjects
            .iter()
            .find(|panel| panel.id.engine == "e1")
            .unwrap();
        assert!(matches!(
            first.cell("q", "tiny", Demand::Full),
            CellStatus::Unsupported(_)
        ));
        let second = report
            .subjects
            .iter()
            .find(|panel| panel.id.engine == "e2")
            .unwrap();
        assert!(matches!(
            second.cell("q", "tiny", Demand::Rows(1)),
            CellStatus::Error(_)
        ));
        assert!(matches!(
            second.cell("q", "tiny", Demand::Full),
            CellStatus::Missing(_)
        ));
    }

    #[test]
    fn rejects_numeric_and_cross_subject_full_cardinality_mismatches() {
        let rows = [
            row(
                "identity",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "0",
                "4",
                "a",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "2",
                "100",
                "1",
                "-",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "400",
                "4",
                "-",
                "",
            ),
            row(
                "identity",
                "e2",
                "succinct",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "0",
                "5",
                "b",
                "",
            ),
            row(
                "sample",
                "e2",
                "succinct",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "500",
                "5",
                "-",
                "",
            ),
        ];
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let report = parse(&refs);
        let first = report
            .subjects
            .iter()
            .find(|panel| panel.id.engine == "e1")
            .unwrap();
        assert!(matches!(
            first.cell("q", "tiny", Demand::Rows(2)),
            CellStatus::CardinalityMismatch(_)
        ));
        assert!(matches!(
            first.cell("q", "tiny", Demand::Full),
            CellStatus::CardinalityMismatch(_)
        ));
    }

    #[test]
    fn understands_fragmented_axis_names_without_a_schema_migration() {
        let input = "kind\tcommit\tstorage\tdevice\tmode\tquery\tdataset-scale\tlimit\tduration-ns\tresult-rows\n\
                     sample\te1\ttribleset\tcpu\tsequential\tq\ttiny\t1\t42\t1";
        let report = FingerprintReport::from_tsv(input).unwrap();
        assert!(report.issues.is_empty());
        assert_eq!(report.subjects[0].id.backend, "tribleset");
        assert!(matches!(
            report.subjects[0].cell("q", "tiny", Demand::Rows(1)),
            CellStatus::Measured(_)
        ));
    }

    #[test]
    fn keeps_setup_first_and_full_terminal() {
        let rows = [
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "full",
                "40",
                "4",
                "-",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "4",
                "30",
                "4",
                "-",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "construct",
                "20",
                "0",
                "-",
                "",
            ),
            row(
                "sample",
                "e1",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "1",
                "10",
                "1",
                "-",
                "",
            ),
        ];
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let report = parse(&refs);
        assert_eq!(
            report.demands,
            vec![
                Demand::Construct,
                Demand::Rows(1),
                Demand::Rows(4),
                Demand::Full
            ]
        );
    }

    #[test]
    fn marks_protocol_omissions_as_unsupported_instead_of_missing() {
        let rows = [
            row(
                "identity", "e1", "succinct", "cpu", "rayon", "q", "tiny", "full", "0", "4",
                "same", "",
            ),
            row(
                "sample", "e1", "succinct", "cpu", "rayon", "q", "tiny", "full", "40", "4", "-", "",
            ),
            // Another subject declares the global demand axes which are
            // intentionally absent from this Rayon/4-row cell.
            row(
                "sample",
                "e2",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "construct",
                "10",
                "0",
                "-",
                "",
            ),
            row(
                "sample",
                "e2",
                "tribleset",
                "cpu",
                "sequential",
                "q",
                "tiny",
                "4",
                "40",
                "4",
                "-",
                "",
            ),
        ];
        let refs: Vec<&str> = rows.iter().map(String::as_str).collect();
        let report = parse(&refs);
        let rayon = report
            .subjects
            .iter()
            .find(|panel| panel.id.engine == "e1")
            .unwrap();
        assert!(matches!(
            rayon.cell("q", "tiny", Demand::Construct),
            CellStatus::NotApplicable(_)
        ));
        assert!(matches!(
            rayon.cell("q", "tiny", Demand::Rows(4)),
            CellStatus::NotApplicable(_)
        ));
    }

    #[test]
    fn exposes_incomplete_sample_and_identity_grids() {
        let input = "record\tengine\tbackend\tsubstrate\tparallelism\tshape\tscale\tdemand\telapsed_ns\trows\tresult_digest\tabba_position\trepetition\n\
sample\te1\ttribleset\tcpu\tsequential\tcomplete\ttiny\t1\t10\t1\t-\tp1\t0\n\
sample\te1\ttribleset\tcpu\tsequential\tcomplete\ttiny\t1\t11\t1\t-\tp1\t1\n\
sample\te1\ttribleset\tcpu\tsequential\tcomplete\ttiny\t1\t12\t1\t-\tp2\t0\n\
sample\te1\ttribleset\tcpu\tsequential\tincomplete\ttiny\t1\t10\t1\t-\tp1\t0\n\
identity\te1\ttribleset\tcpu\tsequential\tidentity_gap\ttiny\tfull\t0\t4\tsame\tp1\t-\n\
sample\te1\ttribleset\tcpu\tsequential\tidentity_gap\ttiny\tfull\t40\t4\t-\tp1\t0\n\
sample\te1\ttribleset\tcpu\tsequential\tidentity_gap\ttiny\tfull\t41\t4\t-\tp1\t1\n\
sample\te1\ttribleset\tcpu\tsequential\tidentity_gap\ttiny\tfull\t42\t4\t-\tp2\t0";
        let report = FingerprintReport::from_tsv(input).unwrap();
        let panel = &report.subjects[0];

        let CellStatus::Missing(sample_detail) = panel.cell("incomplete", "tiny", Demand::Rows(1))
        else {
            panic!("partial repetition coverage should be missing")
        };
        assert!(sample_detail.contains("observation grid"));

        let CellStatus::Missing(identity_detail) = panel.cell("identity_gap", "tiny", Demand::Full)
        else {
            panic!("partial identity coverage should be missing")
        };
        assert!(identity_detail.contains("identity grid"));
    }
}
