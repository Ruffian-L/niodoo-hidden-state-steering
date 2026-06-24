use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::runtime::telemetry::TokenPhysics;

const MAX_WINDOW_SIZE: usize = 64;
const METRIC_COUNT: usize = 16;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TdaDimSummary {
    pub(crate) dim: usize,
    pub(crate) bars: usize,
    pub(crate) finite_bars: usize,
    pub(crate) infinite_bars: usize,
    pub(crate) max_persistence: f32,
    pub(crate) mean_persistence: f32,
    pub(crate) total_persistence: f32,
}

impl TdaDimSummary {
    fn empty(dim: usize) -> Self {
        Self {
            dim,
            bars: 0,
            finite_bars: 0,
            infinite_bars: 0,
            max_persistence: 0.0,
            mean_persistence: 0.0,
            total_persistence: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TdaShadowAction {
    Observe,
    WouldPause,
    WouldFocus,
    WouldUnfold,
    WouldLock,
}

impl TdaShadowAction {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::WouldPause => "would_pause",
            Self::WouldFocus => "would_focus",
            Self::WouldUnfold => "would_unfold",
            Self::WouldLock => "would_lock",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TdaShadowSignals {
    pub(crate) loop_pressure: f32,
    pub(crate) route_fragmentation: f32,
    pub(crate) margin_collapse: f32,
    pub(crate) force_overfire: f32,
    pub(crate) route_churn: f32,
    pub(crate) tag_density: f32,
    pub(crate) repetition_pressure: f32,
    pub(crate) breath_score: f32,
}

impl Default for TdaShadowSignals {
    fn default() -> Self {
        Self {
            loop_pressure: 0.0,
            route_fragmentation: 0.0,
            margin_collapse: 0.0,
            force_overfire: 0.0,
            route_churn: 0.0,
            tag_density: 0.0,
            repetition_pressure: 0.0,
            breath_score: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TdaInvolutionCheck {
    pub(crate) max_double_apply_residual_l2: f32,
    pub(crate) mean_double_apply_residual_l2: f32,
    pub(crate) valid_with_fixed_axis: bool,
}

impl Default for TdaInvolutionCheck {
    fn default() -> Self {
        Self {
            max_double_apply_residual_l2: 0.0,
            mean_double_apply_residual_l2: 0.0,
            valid_with_fixed_axis: true,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TdaShadowDecision {
    pub(crate) action: TdaShadowAction,
    pub(crate) should_breathe: bool,
    pub(crate) reason: &'static str,
    pub(crate) signals: TdaShadowSignals,
    pub(crate) dimensions: [TdaDimSummary; 2],
    pub(crate) involution: TdaInvolutionCheck,
}

#[derive(Debug, Clone)]
pub(crate) struct TdaShadowMonitor {
    window_size: usize,
    stride: usize,
    observed: usize,
    samples: VecDeque<TdaTokenSample>,
    last_decision: Option<TdaShadowDecision>,
}

impl TdaShadowMonitor {
    pub(crate) fn new(window_size: usize, stride: usize) -> Self {
        let window_size = window_size.clamp(3, MAX_WINDOW_SIZE);
        let stride = stride.max(1);
        Self {
            window_size,
            stride,
            observed: 0,
            samples: VecDeque::with_capacity(window_size),
            last_decision: None,
        }
    }

    pub(crate) fn annotate(&mut self, token: &mut TokenPhysics) -> Option<TdaShadowDecision> {
        let fresh_decision = self.observe(token);
        token.tda_shadow_enabled = true;
        token.tda_shadow_window_size = self.window_size;
        token.tda_shadow_stride = self.stride;
        token.tda_shadow_window_ready = self.samples.len() >= self.window_size;
        token.tda_shadow_decision_fresh = fresh_decision.is_some();

        if let Some(decision) = self.last_decision.as_ref() {
            apply_decision_to_token(token, decision);
        } else {
            token.tda_shadow_action = "warming".to_string();
            token.tda_shadow_reason = "window_not_ready".to_string();
        }

        fresh_decision
    }

    fn observe(&mut self, token: &TokenPhysics) -> Option<TdaShadowDecision> {
        self.observed = self.observed.saturating_add(1);
        self.samples.push_back(TdaTokenSample::from_token(token));
        while self.samples.len() > self.window_size {
            self.samples.pop_front();
        }

        if self.samples.len() < self.window_size {
            return None;
        }

        let ready_index = self.observed.saturating_sub(self.window_size);
        if ready_index % self.stride != 0 {
            return None;
        }

        let window: Vec<TdaTokenSample> = self.samples.iter().cloned().collect();
        let decision = analyze_window(&window);
        self.last_decision = Some(decision.clone());
        Some(decision)
    }
}

fn apply_decision_to_token(token: &mut TokenPhysics, decision: &TdaShadowDecision) {
    token.tda_shadow_action = decision.action.as_str().to_string();
    token.tda_shadow_reason = decision.reason.to_string();
    token.tda_shadow_breath_requested = decision.should_breathe;
    token.tda_shadow_loop_pressure = decision.signals.loop_pressure;
    token.tda_shadow_route_fragmentation = decision.signals.route_fragmentation;
    token.tda_shadow_margin_collapse = decision.signals.margin_collapse;
    token.tda_shadow_force_overfire = decision.signals.force_overfire;
    token.tda_shadow_route_churn = decision.signals.route_churn;
    token.tda_shadow_tag_density = decision.signals.tag_density;
    token.tda_shadow_repetition_pressure = decision.signals.repetition_pressure;
    token.tda_shadow_breath_score = decision.signals.breath_score;
    token.tda_shadow_h0_bars = decision.dimensions[0].bars;
    token.tda_shadow_h0_finite_bars = decision.dimensions[0].finite_bars;
    token.tda_shadow_h0_infinite_bars = decision.dimensions[0].infinite_bars;
    token.tda_shadow_h0_total_persistence = decision.dimensions[0].total_persistence;
    token.tda_shadow_h0_max_persistence = decision.dimensions[0].max_persistence;
    token.tda_shadow_h1_bars = decision.dimensions[1].bars;
    token.tda_shadow_h1_finite_bars = decision.dimensions[1].finite_bars;
    token.tda_shadow_h1_infinite_bars = decision.dimensions[1].infinite_bars;
    token.tda_shadow_h1_total_persistence = decision.dimensions[1].total_persistence;
    token.tda_shadow_h1_max_persistence = decision.dimensions[1].max_persistence;
    token.tda_shadow_involution_residual_max = decision.involution.max_double_apply_residual_l2;
    token.tda_shadow_involution_residual_mean = decision.involution.mean_double_apply_residual_l2;
    token.tda_shadow_involution_valid = decision.involution.valid_with_fixed_axis;
}

#[derive(Debug, Clone)]
struct TdaTokenSample {
    step: usize,
    token: String,
    route_surface_id: Option<String>,
    metrics: [f32; METRIC_COUNT],
}

impl TdaTokenSample {
    fn from_token(record: &TokenPhysics) -> Self {
        Self {
            step: record.step,
            token: record.token.clone(),
            route_surface_id: record.route_surface_id.clone(),
            metrics: [
                record.route_margin,
                record.nearest_ghost_distance,
                record.second_nearest_ghost_distance,
                record.ghost_pull_delta_norm,
                record.total_force,
                record.activation_gate,
                record.live_basin_pressure,
                record.vq_encode_error,
                record.correction_delta_norm,
                record.correction_packet_fire_count as f32,
                record.correction_packet_force_norm,
                record.correction_packet_vq_encode_error,
                record.bridge_motif_count as f32,
                bool_metric(record.lock_detected),
                bool_metric(record.intervention_applied),
                bool_metric(record.surface_heuristic_flag),
            ],
        }
    }
}

fn analyze_window(samples: &[TdaTokenSample]) -> TdaShadowDecision {
    let points: Vec<Vec<f32>> = samples
        .iter()
        .map(|sample| sample.metrics.to_vec())
        .collect();
    let dimensions = persistent_shape(&points);
    let signals = gate_signals(samples, &dimensions);
    let action = gate_action(samples, &signals);
    let reason = gate_reason(action);
    let should_breathe = matches!(
        action,
        TdaShadowAction::WouldPause | TdaShadowAction::WouldFocus
    ) || signals.breath_score >= 0.55;
    let involution = context_reflection_involution_check(&points);

    TdaShadowDecision {
        action,
        should_breathe,
        reason,
        signals,
        dimensions,
        involution,
    }
}

fn gate_action(samples: &[TdaTokenSample], signals: &TdaShadowSignals) -> TdaShadowAction {
    let lock_seen = samples.iter().any(|sample| {
        metric(sample, Metric::LockDetected) > 0.0 || sample.token.contains("[REQUEST: LOCK]")
    });

    if lock_seen
        && signals.loop_pressure < 0.35
        && signals.margin_collapse < 0.60
        && signals.force_overfire < 0.55
    {
        TdaShadowAction::WouldLock
    } else if signals.loop_pressure >= 0.50 && signals.force_overfire >= 0.45 {
        TdaShadowAction::WouldUnfold
    } else if signals.breath_score >= 0.55 {
        TdaShadowAction::WouldPause
    } else if signals.route_fragmentation >= 0.60 || signals.route_churn >= 0.35 {
        TdaShadowAction::WouldFocus
    } else {
        TdaShadowAction::Observe
    }
}

fn gate_reason(action: TdaShadowAction) -> &'static str {
    match action {
        TdaShadowAction::Observe => "signals_below_shadow_gate_threshold",
        TdaShadowAction::WouldPause => "breath_score_high_from_loop_or_margin_collapse",
        TdaShadowAction::WouldFocus => "route_fragmentation_or_surface_churn_high",
        TdaShadowAction::WouldUnfold => "loop_pressure_and_force_overfire_high",
        TdaShadowAction::WouldLock => "lock_surface_seen_with_low_loop_and_low_overfire",
    }
}

fn persistent_shape(points: &[Vec<f32>]) -> [TdaDimSummary; 2] {
    if points.len() < 2 {
        return [TdaDimSummary::empty(0), TdaDimSummary::empty(1)];
    }
    let normalized = zscore_columns(points);
    let distances = distance_matrix(&normalized);
    let simplices = build_vietoris_rips_simplices(&distances);
    summarize_persistence(&simplices)
}

#[derive(Debug, Clone)]
struct Simplex {
    vertices: Vec<usize>,
    dim: usize,
    filtration: f32,
}

fn build_vietoris_rips_simplices(distances: &[Vec<f32>]) -> Vec<Simplex> {
    let n = distances.len();
    let mut simplices = Vec::new();
    for i in 0..n {
        simplices.push(Simplex {
            vertices: vec![i],
            dim: 0,
            filtration: 0.0,
        });
    }
    for i in 0..n {
        for j in (i + 1)..n {
            simplices.push(Simplex {
                vertices: vec![i, j],
                dim: 1,
                filtration: distances[i][j],
            });
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                simplices.push(Simplex {
                    vertices: vec![i, j, k],
                    dim: 2,
                    filtration: distances[i][j].max(distances[i][k]).max(distances[j][k]),
                });
            }
        }
    }

    simplices.sort_by(|a, b| {
        a.filtration
            .partial_cmp(&b.filtration)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.dim.cmp(&b.dim))
            .then_with(|| a.vertices.cmp(&b.vertices))
    });
    simplices
}

fn summarize_persistence(simplices: &[Simplex]) -> [TdaDimSummary; 2] {
    let mut index_by_vertices: HashMap<Vec<usize>, usize> = HashMap::new();
    for (idx, simplex) in simplices.iter().enumerate() {
        index_by_vertices.insert(simplex.vertices.clone(), idx);
    }

    let mut reduced_by_low: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut paired_lows: HashSet<usize> = HashSet::new();
    let mut positive = vec![false; simplices.len()];
    let mut bars: [Vec<(f32, f32)>; 2] = [Vec::new(), Vec::new()];

    for (idx, simplex) in simplices.iter().enumerate() {
        let mut column = boundary_indices(simplex, &index_by_vertices);
        while let Some(low) = column.last().copied() {
            if let Some(previous) = reduced_by_low.get(&low) {
                column = xor_sorted(&column, previous);
            } else {
                break;
            }
        }

        if column.is_empty() {
            positive[idx] = true;
            continue;
        }

        let low = *column.last().expect("non-empty reduced column");
        reduced_by_low.insert(low, column);
        paired_lows.insert(low);
        let birth_dim = simplices[low].dim;
        if birth_dim <= 1 {
            bars[birth_dim].push((simplices[low].filtration, simplex.filtration));
        }
    }

    for (idx, simplex) in simplices.iter().enumerate() {
        if positive[idx] && !paired_lows.contains(&idx) && simplex.dim <= 1 {
            bars[simplex.dim].push((simplex.filtration, f32::INFINITY));
        }
    }

    [summarize_dim(0, &bars[0]), summarize_dim(1, &bars[1])]
}

fn boundary_indices(
    simplex: &Simplex,
    index_by_vertices: &HashMap<Vec<usize>, usize>,
) -> Vec<usize> {
    if simplex.dim == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(simplex.vertices.len());
    for remove_idx in 0..simplex.vertices.len() {
        let mut face = Vec::with_capacity(simplex.vertices.len() - 1);
        for (idx, vertex) in simplex.vertices.iter().enumerate() {
            if idx != remove_idx {
                face.push(*vertex);
            }
        }
        if let Some(face_idx) = index_by_vertices.get(&face) {
            out.push(*face_idx);
        }
    }
    out.sort_unstable();
    out
}

fn xor_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len().max(b.len()));
    let mut i = 0;
    let mut j = 0;
    while i < a.len() || j < b.len() {
        if i >= a.len() {
            out.push(b[j]);
            j += 1;
        } else if j >= b.len() {
            out.push(a[i]);
            i += 1;
        } else if a[i] == b[j] {
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out
}

fn summarize_dim(dim: usize, values: &[(f32, f32)]) -> TdaDimSummary {
    let mut finite_bars = 0usize;
    let mut infinite_bars = 0usize;
    let mut max_persistence = 0.0f32;
    let mut total_persistence = 0.0f32;
    for (birth, death) in values {
        if death.is_infinite() {
            infinite_bars += 1;
        } else {
            finite_bars += 1;
            let persistence = (death - birth).max(0.0);
            max_persistence = max_persistence.max(persistence);
            total_persistence += persistence;
        }
    }
    TdaDimSummary {
        dim,
        bars: values.len(),
        finite_bars,
        infinite_bars,
        max_persistence,
        mean_persistence: if finite_bars > 0 {
            total_persistence / finite_bars as f32
        } else {
            0.0
        },
        total_persistence,
    }
}

fn gate_signals(samples: &[TdaTokenSample], dims: &[TdaDimSummary; 2]) -> TdaShadowSignals {
    let h0 = dims[0];
    let h1 = dims[1];
    let n = samples.len().max(1) as f32;
    let margin = metric_values(samples, Metric::RouteMargin);
    let total_force = metric_values(samples, Metric::TotalForce);
    let packet_fire = metric_values(samples, Metric::CorrectionPacketFireCount);
    let intervention = metric_values(samples, Metric::InterventionApplied);

    let margin_mean = mean(&margin);
    let margin_std = stddev(&margin, margin_mean);
    let margin_slope = slope(&margin);
    let force_mean = mean(&total_force);
    let force_std = stddev(&total_force, force_mean);
    let fire_density = packet_fire.iter().filter(|value| **value > 0.0).count() as f32 / n;
    let intervention_density = intervention.iter().filter(|value| **value > 0.0).count() as f32 / n;
    let loop_pressure = clamp01(h1.total_persistence / (h1.total_persistence + n.sqrt() + 1e-6));
    let route_fragmentation =
        clamp01(h0.finite_bars as f32 / n + h0.max_persistence / (h0.total_persistence + 1.0));
    let margin_collapse = clamp01(
        1.0 / (1.0 + margin_mean.max(0.0) * 20.0)
            + (-margin_slope).max(0.0) / (margin_std + margin_mean.abs() + 1e-3),
    );
    let force_overfire = clamp01(
        fire_density
            + intervention_density * 0.5
            + force_std / (force_mean.abs() + force_std + 1.0),
    );
    let route_churn = route_surface_churn(samples);
    let tag_density = tag_density(samples);
    let repetition_pressure = repetition_pressure(samples);
    let breath_score = clamp01(
        loop_pressure * 0.45
            + margin_collapse * 0.25
            + force_overfire * 0.15
            + repetition_pressure * 0.15,
    );

    TdaShadowSignals {
        loop_pressure,
        route_fragmentation,
        margin_collapse,
        force_overfire,
        route_churn,
        tag_density,
        repetition_pressure,
        breath_score,
    }
}

fn context_reflection_involution_check(points: &[Vec<f32>]) -> TdaInvolutionCheck {
    if points.is_empty() {
        return TdaInvolutionCheck::default();
    }
    let normalized = zscore_columns(points);
    let center = centroid(&normalized);
    let axis = principal_axis_approx(&normalized, &center);
    let mut max_residual = 0.0f32;
    let mut total_residual = 0.0f32;
    for point in &normalized {
        let reflected = reflect_about_center(point, &center, &axis);
        let double_reflected = reflect_about_center(&reflected, &center, &axis);
        let residual = euclidean_distance(point, &double_reflected);
        max_residual = max_residual.max(residual);
        total_residual += residual;
    }
    let mean_residual = total_residual / normalized.len().max(1) as f32;
    TdaInvolutionCheck {
        max_double_apply_residual_l2: max_residual,
        mean_double_apply_residual_l2: mean_residual,
        valid_with_fixed_axis: max_residual < 1e-4,
    }
}

fn zscore_columns(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let dim = points.iter().map(Vec::len).max().unwrap_or(0);
    if dim == 0 {
        return vec![Vec::new(); points.len()];
    }
    let mut means = vec![0.0f32; dim];
    for point in points {
        for idx in 0..dim {
            means[idx] += point.get(idx).copied().unwrap_or(0.0);
        }
    }
    for mean in &mut means {
        *mean /= points.len().max(1) as f32;
    }
    let mut variances = vec![0.0f32; dim];
    for point in points {
        for idx in 0..dim {
            let delta = point.get(idx).copied().unwrap_or(0.0) - means[idx];
            variances[idx] += delta * delta;
        }
    }
    let stddevs: Vec<f32> = variances
        .into_iter()
        .map(|value| (value / points.len().max(1) as f32).sqrt().max(1e-6))
        .collect();
    points
        .iter()
        .map(|point| {
            (0..dim)
                .map(|idx| (point.get(idx).copied().unwrap_or(0.0) - means[idx]) / stddevs[idx])
                .collect()
        })
        .collect()
}

fn distance_matrix(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = points.len();
    let mut distances = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..i {
            let distance = euclidean_distance(&points[i], &points[j]);
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }
    distances
}

fn reflect_about_center(point: &[f32], center: &[f32], axis: &[f32]) -> Vec<f32> {
    let offset: Vec<f32> = point
        .iter()
        .zip(center.iter())
        .map(|(value, mean)| value - mean)
        .collect();
    let projection_scale = dot(&offset, axis);
    point
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let projected = projection_scale * axis[idx];
            let orthogonal = offset[idx] - projected;
            center[idx] - projected + orthogonal
        })
        .collect()
}

fn principal_axis_approx(points: &[Vec<f32>], center: &[f32]) -> Vec<f32> {
    let dim = center.len();
    let mut axis = vec![0.0f32; dim];
    for point in points {
        for idx in 0..dim {
            axis[idx] += point[idx] - center[idx];
        }
    }
    if norm(&axis) <= 1e-6 {
        if let (Some(first), Some(last)) = (points.first(), points.last()) {
            for idx in 0..dim {
                axis[idx] = last[idx] - first[idx];
            }
        }
    }
    let axis_norm = norm(&axis);
    if axis_norm <= 1e-6 {
        axis.fill(0.0);
        if !axis.is_empty() {
            axis[0] = 1.0;
        }
        return axis;
    }
    axis.into_iter().map(|value| value / axis_norm).collect()
}

fn centroid(points: &[Vec<f32>]) -> Vec<f32> {
    let dim = points.iter().map(Vec::len).max().unwrap_or(0);
    let mut center = vec![0.0f32; dim];
    for point in points {
        for idx in 0..dim {
            center[idx] += point.get(idx).copied().unwrap_or(0.0);
        }
    }
    for value in &mut center {
        *value /= points.len().max(1) as f32;
    }
    center
}

#[derive(Debug, Clone, Copy)]
enum Metric {
    RouteMargin = 0,
    TotalForce = 4,
    CorrectionPacketFireCount = 9,
    LockDetected = 13,
    InterventionApplied = 14,
}

fn metric_values(samples: &[TdaTokenSample], field: Metric) -> Vec<f32> {
    samples.iter().map(|sample| metric(sample, field)).collect()
}

fn metric(sample: &TdaTokenSample, field: Metric) -> f32 {
    sample.metrics[field as usize]
}

fn route_surface_churn(samples: &[TdaTokenSample]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let changes = samples
        .windows(2)
        .filter(|pair| pair[0].route_surface_id != pair[1].route_surface_id)
        .count();
    changes as f32 / (samples.len() - 1) as f32
}

fn tag_density(samples: &[TdaTokenSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let hits = samples
        .iter()
        .filter(|sample| {
            let upper = sample.token.to_ascii_uppercase();
            upper.contains("[REQUEST:")
                || upper.contains("FOCUS")
                || upper.contains("RESET")
                || upper.contains("LOCK")
                || upper.contains("SPIKE")
                || upper.contains("EXPLORE")
        })
        .count();
    hits as f32 / samples.len() as f32
}

fn repetition_pressure(samples: &[TdaTokenSample]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut repeats = 0usize;
    for idx in 1..samples.len() {
        if !samples[idx].token.is_empty() && samples[idx].token == samples[idx - 1].token {
            repeats += 1;
        }
    }
    repeats as f32 / samples.len() as f32
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn stddev(values: &[f32], mean: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    (values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32)
        .sqrt()
}

fn slope(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean_x = (n - 1.0) * 0.5;
    let mean_y = mean(values);
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (idx, value) in values.iter().enumerate() {
        let dx = idx as f32 - mean_x;
        numerator += dx * (value - mean_y);
        denominator += dx * dx;
    }
    if denominator <= 1e-6 {
        0.0
    } else {
        numerator / denominator
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let dim = a.len().max(b.len());
    (0..dim)
        .map(|idx| {
            let delta = a.get(idx).copied().unwrap_or(0.0) - b.get(idx).copied().unwrap_or(0.0);
            delta * delta
        })
        .sum::<f32>()
        .sqrt()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let dim = a.len().max(b.len());
    (0..dim)
        .map(|idx| a.get(idx).copied().unwrap_or(0.0) * b.get(idx).copied().unwrap_or(0.0))
        .sum()
}

fn norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn bool_metric(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_points_have_h1_signal() {
        let mut points = Vec::new();
        for idx in 0..24 {
            let theta = idx as f32 * std::f32::consts::TAU / 24.0;
            points.push(vec![theta.cos(), theta.sin()]);
        }
        let dims = persistent_shape(&points);
        assert!(dims[1].finite_bars > 0, "{dims:?}");
        assert!(dims[1].total_persistence > 0.0, "{dims:?}");
    }

    #[test]
    fn context_reflection_is_involutive_with_fixed_axis() {
        let points = vec![
            vec![1.0, 0.0, 0.1],
            vec![0.0, 1.0, 0.2],
            vec![-1.0, 0.0, 0.3],
            vec![0.0, -1.0, 0.4],
        ];
        let check = context_reflection_involution_check(&points);
        assert!(check.valid_with_fixed_axis, "{check:?}");
        assert!(check.max_double_apply_residual_l2 < 1e-4, "{check:?}");
    }

    #[test]
    fn monitor_waits_for_full_window_then_emits_decision() {
        let mut monitor = TdaShadowMonitor::new(8, 4);
        let mut last = None;
        for idx in 0..8 {
            let mut record = TokenPhysics::default();
            record.step = idx;
            record.token = format!("tok{idx}");
            record.route_surface_id = Some(format!("route{}", idx % 2));
            record.route_margin = if idx < 4 { 0.4 } else { 0.01 };
            record.nearest_ghost_distance = idx as f32 * 0.1;
            record.second_nearest_ghost_distance = 1.0 + idx as f32 * 0.1;
            record.total_force = if idx % 3 == 0 { 10.0 } else { 1.0 };
            record.intervention_applied = idx % 3 == 0;
            last = monitor.annotate(&mut record);
        }
        let decision = last.expect("window should emit on eighth token");
        assert!(matches!(
            decision.action,
            TdaShadowAction::Observe
                | TdaShadowAction::WouldPause
                | TdaShadowAction::WouldFocus
                | TdaShadowAction::WouldUnfold
                | TdaShadowAction::WouldLock
        ));
    }
}
