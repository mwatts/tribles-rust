mod model;
mod widget;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use model::FingerprintReport;
use widget::FingerprintPanel;
use GORBIE::prelude::*;

struct LoadedFingerprint {
    source: String,
    report: Arc<FingerprintReport>,
}

static LOADED: OnceLock<Result<LoadedFingerprint, String>> = OnceLock::new();

#[notebook(name = "Query-engine performance fingerprint")]
fn performance_fingerprint(nb: &mut NotebookCtx) {
    let padding = GORBIE::cards::DEFAULT_CARD_PADDING;
    match loaded_fingerprint() {
        Ok(loaded) => {
            let report = Arc::clone(&loaded.report);
            let source = loaded.source.clone();
            let summary = Arc::clone(&report);
            nb.view(move |ctx| {
                ctx.with_padding(padding, |ctx| {
                    ctx.heading("Query-engine performance fingerprint");
                    ctx.label(
                        "One structurally identical panel per engine/storage/execution subject. The raw benchmark receipt remains canonical; this notebook performs only an in-memory normalization.",
                    );
                    ctx.add_space(6.0);
                    ctx.label(egui::RichText::new(format!("source: {source}")).monospace());
                    ctx.label(format!(
                        "{} input rows · {} timing samples · {} subjects · {} shapes · {} scales · {} demand labels",
                        summary.input_rows,
                        summary.samples,
                        summary.subjects.len(),
                        summary.shapes.len(),
                        summary.scales.len(),
                        summary.demands.len(),
                    ));
                    ctx.add_space(8.0);
                    ctx.strong("Reading the fingerprint");
                    ctx.label(
                        "The plotted quantity is c(k)=median T(k)/k. k=1 is time to first result. `full` stays a terminal exhaustion demand instead of being collapsed into a numeric limit. Falling c(k) indicates amortization; flat indicates linear work; rising indicates superlinear work.",
                    );
                    ctx.label(
                        "Only measured cells enter curves. The tables preserve the complete observed axis product and print missing, unsupported, producer error, and cardinality mismatch states in place. Rayon setup and numeric demands at or beyond an agreed full cardinality are classified as unsupported protocol points, not false missing data.",
                    );

                    if !summary.issues.is_empty() {
                        ctx.add_space(8.0);
                        ctx.colored_label(
                            egui::Color32::from_rgb(246, 189, 96),
                            format!("{} input issue(s)", summary.issues.len()),
                        );
                        for issue in &summary.issues {
                            let location = issue
                                .line
                                .map(|line| format!("line {line}: "))
                                .unwrap_or_default();
                            ctx.label(
                                egui::RichText::new(format!("• {location}{}", issue.message))
                                    .monospace(),
                            );
                        }
                    }
                });
            });

            for subject_index in 0..report.subjects.len() {
                let report = Arc::clone(&report);
                nb.view(move |ctx| {
                    ctx.with_padding(padding, |ctx| {
                        ctx.add(FingerprintPanel::new(
                            report.as_ref(),
                            &report.subjects[subject_index],
                        ));
                    });
                });
            }
        }
        Err(message) => {
            let message = message.clone();
            nb.view(move |ctx| {
                ctx.with_padding(padding, |ctx| {
                    ctx.heading("Performance fingerprint could not be loaded");
                    ctx.colored_label(egui::Color32::from_rgb(239, 92, 98), &message);
                    ctx.add_space(8.0);
                    ctx.label(
                        "Pass observations.tsv as the first positional argument, set PERFORMANCE_FINGERPRINT_TSV, or use --demo.",
                    );
                });
            });
        }
    }
    nb.settled();
}

fn main() {
    performance_fingerprint();
}

fn loaded_fingerprint() -> &'static Result<LoadedFingerprint, String> {
    LOADED.get_or_init(|| {
        let input = input_selection()?;
        let (source, contents) = match input {
            InputSelection::Demo => ("embedded demonstration receipt".to_owned(), demo_tsv()),
            InputSelection::Path(path) => {
                let path = if path.is_dir() {
                    path.join("observations.tsv")
                } else {
                    path
                };
                let contents = std::fs::read_to_string(&path)
                    .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
                (path.display().to_string(), contents)
            }
        };
        let report = FingerprintReport::from_tsv(&contents)
            .map_err(|error| format!("failed to adapt {source}: {error}"))?;
        Ok(LoadedFingerprint {
            source,
            report: Arc::new(report),
        })
    })
}

enum InputSelection {
    Demo,
    Path(PathBuf),
}

fn input_selection() -> Result<InputSelection, String> {
    let mut positional = None;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--demo" {
            return Ok(InputSelection::Demo);
        }
        if matches!(
            argument.as_str(),
            "--out-dir" | "--scale" | "--headless-wait-ms" | "--headless-max-texture-side"
        ) {
            let _ = args.next();
            continue;
        }
        if argument.starts_with('-') {
            continue;
        }
        positional.get_or_insert_with(|| PathBuf::from(argument));
    }

    if let Some(path) = positional {
        return Ok(InputSelection::Path(path));
    }
    if let Some(path) = std::env::var_os("PERFORMANCE_FINGERPRINT_TSV") {
        return Ok(InputSelection::Path(PathBuf::from(path)));
    }
    Err(
        "no benchmark receipt selected (expected observations.tsv, PERFORMANCE_FINGERPRINT_TSV, or --demo)"
            .to_owned(),
    )
}

fn demo_tsv() -> String {
    include_str!("demo.tsv").to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::{CellStatus, Demand};

    #[test]
    fn demo_exercises_every_visible_status() {
        let report = FingerprintReport::from_tsv(&demo_tsv()).unwrap();
        let names = report
            .subjects
            .iter()
            .flat_map(|subject| subject.status_counts().into_keys())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(names.contains("measured"));
        assert!(names.contains("missing"));
        assert!(names.contains("unsupported"));
        assert!(names.contains("error"));
        assert!(names.contains("cardinality mismatch"));

        let rayon = report
            .subjects
            .iter()
            .find(|subject| subject.id.parallelism == "rayon")
            .unwrap();
        assert!(matches!(
            rayon.cell("bound_star", "wide", Demand::Rows(1)),
            CellStatus::Missing(_)
        ));
    }

    #[test]
    fn directory_inputs_resolve_to_observations_file() {
        let directory = PathBuf::from("a-directory");
        assert_eq!(
            directory.join("observations.tsv"),
            PathBuf::from("a-directory/observations.tsv")
        );
    }
}
