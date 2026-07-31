use egui::{Color32, RichText, Ui, Widget};
use egui_plot::{GridMark, Legend, Line, MarkerShape, Plot, Points};

use crate::model::{CellStatus, Demand, FingerprintReport, MeasuredCell, SubjectPanel};

const SCALE_COLORS: [Color32; 8] = [
    Color32::from_rgb(73, 166, 255),
    Color32::from_rgb(246, 189, 96),
    Color32::from_rgb(97, 210, 163),
    Color32::from_rgb(232, 112, 136),
    Color32::from_rgb(180, 137, 255),
    Color32::from_rgb(106, 205, 216),
    Color32::from_rgb(239, 142, 72),
    Color32::from_rgb(184, 199, 88),
];

const MARKERS: [MarkerShape; 8] = [
    MarkerShape::Circle,
    MarkerShape::Diamond,
    MarkerShape::Square,
    MarkerShape::Up,
    MarkerShape::Down,
    MarkerShape::Cross,
    MarkerShape::Plus,
    MarkerShape::Asterisk,
];

/// A complete, repeatable panel for one benchmark subject.
///
/// Every subject gets this exact structure: provenance/status summary, then a
/// cost curve and complete status matrix for every query shape. The matrix is
/// deliberately not hidden behind plot hover state; a failed or absent point
/// remains part of the fingerprint.
pub struct FingerprintPanel<'a> {
    report: &'a FingerprintReport,
    subject: &'a SubjectPanel,
}

impl<'a> FingerprintPanel<'a> {
    pub fn new(report: &'a FingerprintReport, subject: &'a SubjectPanel) -> Self {
        Self { report, subject }
    }

    fn show(self, ui: &mut Ui) {
        ui.heading(self.subject.id.label());
        ui.horizontal_wrapped(|ui| {
            for (status, count) in self.subject.status_counts() {
                ui.label(
                    RichText::new(format!("{count} {status}"))
                        .monospace()
                        .color(status_color_name(status)),
                );
            }
        });
        ui.label(
            "c(k) = median T(k) / k.  k=1 is time to first result; `full` is a distinct terminal exhaustion demand. Falling curves show amortization, flat curves are linear, and rising curves are superlinear. Setup is reported only in the matrix.",
        );
        ui.add_space(8.0);

        for shape in &self.report.shapes {
            ui.separator();
            ui.heading(shape_title(shape));
            self.show_plot(ui, shape);
            ui.label(
                RichText::new(
                    "Complete cells — timings are medians; every non-measured state is printed in place.",
                )
                .small(),
            );
            self.show_matrix(ui, shape);
            ui.add_space(10.0);
        }

        let diagnostics: Vec<_> = self.subject.diagnostic_cells().collect();
        if !diagnostics.is_empty() {
            ui.separator();
            ui.heading("Diagnostic cells");
            ui.label(
                "Details stay visible in headless/PDF output; the same text is also available by hovering its matrix cell.",
            );
            for (key, status) in diagnostics {
                let detail = match status {
                    CellStatus::Missing(detail)
                    | CellStatus::Unsupported(detail)
                    | CellStatus::Error(detail)
                    | CellStatus::CardinalityMismatch(detail) => detail,
                    CellStatus::Measured(_) | CellStatus::NotApplicable(_) => continue,
                };
                ui.label(
                    RichText::new(format!(
                        "• {}/{}/{}: {} — {detail}",
                        key.shape,
                        key.scale,
                        key.demand.label(),
                        status.name(),
                    ))
                    .monospace()
                    .color(status_color_name(status.name())),
                );
            }
        }
    }

    fn show_plot(&self, ui: &mut Ui, shape: &str) {
        let curve_demands: Vec<Demand> = self
            .report
            .demands
            .iter()
            .copied()
            .filter(|demand| demand.is_curve_point())
            .collect();
        let demand_labels: Vec<String> =
            curve_demands.iter().map(|demand| demand.label()).collect();
        let tooltip_labels = demand_labels.clone();
        let axis_labels = demand_labels.clone();
        let grid_mark_count = demand_labels.len();
        let (minimum_y, maximum_y) = self.shared_y_bounds(shape).unwrap_or((0.0, 1.0));

        let mut plot = Plot::new(("performance-fingerprint", self.subject.id.clone(), shape))
            .height(260.0)
            .legend(Legend::default())
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .include_x(0.0)
            .include_x(curve_demands.len().saturating_sub(1) as f64)
            .include_y(minimum_y)
            .include_y(maximum_y)
            // Headless capture can settle before egui_plot learns its axis
            // thickness from a previous frame. Reserve it explicitly so the
            // first rendered card never clips unit labels.
            .y_axis_min_width(66.0)
            .x_grid_spacer(move |_| {
                (0..grid_mark_count)
                    .map(|index| GridMark {
                        value: index as f64,
                        step_size: 1.0,
                    })
                    .collect()
            })
            .x_axis_formatter(move |mark, _| ordinal_label(mark.value, &axis_labels))
            .y_axis_formatter(|mark, _| format_duration_ns(10_f64.powf(mark.value)))
            .label_formatter(move |name, value| {
                let demand = ordinal_label(value.x, &tooltip_labels);
                if name.is_empty() || demand.is_empty() {
                    String::new()
                } else {
                    format!(
                        "{name}\n{demand}: {}/result",
                        format_duration_ns(10_f64.powf(value.y))
                    )
                }
            });

        if curve_demands.is_empty() {
            plot = plot.show_axes(false);
        }

        plot.show(ui, |plot_ui| {
            for (scale_index, scale) in self.report.scales.iter().enumerate() {
                let color = SCALE_COLORS[scale_index % SCALE_COLORS.len()];
                let marker = MARKERS[scale_index % MARKERS.len()];
                let mut all_points = Vec::new();
                let mut segments: Vec<Vec<[f64; 2]>> = Vec::new();
                let mut segment = Vec::new();

                for (demand_index, &demand) in curve_demands.iter().enumerate() {
                    let point = match self.subject.cell(shape, scale, demand) {
                        CellStatus::Measured(measured) => measured
                            .ns_per_result
                            .filter(|cost| *cost > 0.0 && cost.is_finite())
                            .map(|cost| [demand_index as f64, cost.log10()]),
                        _ => None,
                    };
                    match point {
                        Some(point) => {
                            all_points.push(point);
                            segment.push(point);
                        }
                        None if !segment.is_empty() => {
                            segments.push(std::mem::take(&mut segment));
                        }
                        None => {}
                    }
                }
                if !segment.is_empty() {
                    segments.push(segment);
                }

                for segment in segments.into_iter().filter(|segment| segment.len() > 1) {
                    plot_ui.line(Line::new("", segment).color(color).width(2.0));
                }
                if !all_points.is_empty() {
                    let trend = trend_label(&all_points);
                    plot_ui.points(
                        Points::new(format!("{} — {trend}", scale_title(scale)), all_points)
                            .color(color)
                            .shape(marker)
                            .radius(4.0),
                    );
                }
            }
        });
    }

    fn shared_y_bounds(&self, shape: &str) -> Option<(f64, f64)> {
        let mut minimum = f64::INFINITY;
        let mut maximum = f64::NEG_INFINITY;
        for subject in &self.report.subjects {
            for scale in &self.report.scales {
                for &demand in &self.report.demands {
                    let CellStatus::Measured(measured) = subject.cell(shape, scale, demand) else {
                        continue;
                    };
                    let Some(cost) = measured.ns_per_result else {
                        continue;
                    };
                    if cost <= 0.0 || !cost.is_finite() {
                        continue;
                    }
                    let value = cost.log10();
                    minimum = minimum.min(value);
                    maximum = maximum.max(value);
                }
            }
        }
        if !minimum.is_finite() || !maximum.is_finite() {
            None
        } else if (maximum - minimum).abs() < 0.01 {
            Some((minimum - 0.5, maximum + 0.5))
        } else {
            let margin = (maximum - minimum) * 0.08;
            Some((minimum - margin, maximum + margin))
        }
    }

    fn show_matrix(&self, ui: &mut Ui, shape: &str) {
        const DEMANDS_PER_TABLE: usize = 5;
        for (chunk_index, demands) in self.report.demands.chunks(DEMANDS_PER_TABLE).enumerate() {
            ui.add_space(4.0);
            egui::Grid::new((
                "fingerprint-matrix",
                self.subject.id.clone(),
                shape,
                chunk_index,
            ))
            .min_col_width(88.0)
            .show(ui, |ui| {
                ui.label(RichText::new("scale \\ demand").strong());
                for &demand in demands {
                    ui.label(RichText::new(demand.label()).strong().monospace());
                }
                ui.end_row();

                for scale in &self.report.scales {
                    ui.label(RichText::new(scale_title(scale)).strong());
                    for &demand in demands {
                        show_status_cell(ui, self.subject.cell(shape, scale, demand), demand);
                    }
                    ui.end_row();
                }
            });
        }
    }
}

impl Widget for FingerprintPanel<'_> {
    fn ui(self, ui: &mut Ui) -> egui::Response {
        ui.vertical(|ui| self.show(ui)).response
    }
}

fn show_status_cell(ui: &mut Ui, status: &CellStatus, demand: Demand) {
    let (text, color, detail) = match status {
        CellStatus::Measured(measured) => (
            measured_cell_text(measured, demand),
            Color32::from_rgb(106, 205, 150),
            measured_detail(measured),
        ),
        CellStatus::Missing(detail) => (
            "missing".to_owned(),
            Color32::from_rgb(246, 189, 96),
            detail.clone(),
        ),
        CellStatus::Unsupported(detail) | CellStatus::NotApplicable(detail) => (
            "unsupported".to_owned(),
            Color32::from_gray(190),
            detail.clone(),
        ),
        CellStatus::Error(detail) => (
            "error".to_owned(),
            Color32::from_rgb(239, 92, 98),
            detail.clone(),
        ),
        CellStatus::CardinalityMismatch(detail) => (
            "cardinality\nmismatch".to_owned(),
            Color32::from_rgb(220, 112, 235),
            detail.clone(),
        ),
    };
    ui.label(RichText::new(text).monospace().color(color))
        .on_hover_text(detail);
}

fn measured_cell_text(measured: &MeasuredCell, demand: Demand) -> String {
    match demand {
        Demand::Construct => format!("{} setup", format_duration_ns(measured.median_elapsed_ns)),
        Demand::Rows(_) | Demand::Full => format!(
            "{}/result",
            format_duration_ns(measured.ns_per_result.unwrap_or_default())
        ),
    }
}

fn measured_detail(measured: &MeasuredCell) -> String {
    let paths = if measured.execution_paths.is_empty() {
        "unspecified".to_owned()
    } else {
        measured
            .execution_paths
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "median T = {}; rows = {}; observations = {}; execution path(s) = {paths}",
        format_duration_ns(measured.median_elapsed_ns),
        measured.rows,
        measured.observations,
    )
}

fn trend_label(points: &[[f64; 2]]) -> &'static str {
    let Some((first, rest)) = points.split_first() else {
        return "no samples";
    };
    let Some(last) = rest.last() else {
        return "one point";
    };
    // log10(0.8) and log10(1.25), symmetric around zero.
    let delta = last[1] - first[1];
    if delta < -0.0969 {
        "falling / amortizing"
    } else if delta > 0.0969 {
        "rising / superlinear"
    } else {
        "flat / linear"
    }
}

fn ordinal_label(value: f64, labels: &[String]) -> String {
    let rounded = value.round();
    if (value - rounded).abs() > 0.15 || rounded < 0.0 {
        return String::new();
    }
    labels.get(rounded as usize).cloned().unwrap_or_default()
}

fn status_color_name(status: &str) -> Color32 {
    match status {
        "measured" => Color32::from_rgb(106, 205, 150),
        "missing" => Color32::from_rgb(246, 189, 96),
        "unsupported" => Color32::from_gray(190),
        "error" => Color32::from_rgb(239, 92, 98),
        "cardinality mismatch" => Color32::from_rgb(220, 112, 235),
        _ => Color32::WHITE,
    }
}

fn format_duration_ns(ns: f64) -> String {
    if !ns.is_finite() {
        return "n/a".to_owned();
    }
    if ns < 1_000.0 {
        format_compact(ns, "ns")
    } else if ns < 1_000_000.0 {
        format_compact(ns / 1_000.0, "µs")
    } else if ns < 1_000_000_000.0 {
        format_compact(ns / 1_000_000.0, "ms")
    } else {
        format_compact(ns / 1_000_000_000.0, "s")
    }
}

fn format_compact(value: f64, unit: &str) -> String {
    if value >= 100.0 {
        format!("{value:.0} {unit}")
    } else if value >= 10.0 {
        format!("{value:.1} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

fn shape_title(shape: &str) -> String {
    shape
        .split('_')
        .map(title_word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn scale_title(scale: &str) -> String {
    scale
        .split('_')
        .map(title_word)
        .collect::<Vec<_>>()
        .join(" ")
}

fn title_word(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    first.to_uppercase().chain(chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trend_labels_match_cost_shape() {
        assert_eq!(
            trend_label(&[[0.0, 2.0], [1.0, 1.8]]),
            "falling / amortizing"
        );
        assert_eq!(trend_label(&[[0.0, 2.0], [1.0, 2.01]]), "flat / linear");
        assert_eq!(
            trend_label(&[[0.0, 2.0], [1.0, 2.2]]),
            "rising / superlinear"
        );
    }

    #[test]
    fn formats_time_for_cells_and_axes() {
        assert_eq!(format_duration_ns(420.0), "420 ns");
        assert_eq!(format_duration_ns(4_200.0), "4.20 µs");
        assert_eq!(format_duration_ns(4_200_000.0), "4.20 ms");
    }
}
