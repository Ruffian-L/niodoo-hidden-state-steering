//! Metric channel print helpers — runtime metric summaries + postmortem.
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).

use crate::principia::PrincipiaEngine;
use crate::{MetricChannelSummary, RuntimeMetricAudit};

fn classify_metric_channel(stats: &MetricChannelSummary, inactive_by_design: bool) -> &'static str {
    if inactive_by_design {
        "inactive_by_design"
    } else if stats.count == 0
        || (stats.max - stats.min).abs() < 1e-6
        || (stats.max.abs() < 1e-6 && stats.min.abs() < 1e-6)
    {
        "flat"
    } else if stats.unique_count() <= 2 {
        "binary/coarse"
    } else {
        "dynamic"
    }
}

pub(crate) fn print_metric_summary_line(step: usize, engine: &PrincipiaEngine) {
    println!(
        "[METRICS] step={} status={} gate={:.2} ghost_pre={:.2} ghost_gain={:.2} ghost_applied={:.2} live_basin={:.2} live_radius={:.3} blend={:.2} dyn_rep={:.2} stress={:.2} boredom={:.2} empathy={:.2} adrenaline={:.2} motif={:.2} recovery={:.2} absence={:.2} trap={:.2} hidden_req={:.3}@{} fired={} guard={}",
        step,
        engine.last_engine_status.as_str(),
        engine.last_activation_gate,
        engine.last_ghost_pre_norm,
        engine.last_ghost_gain,
        engine.last_applied_ghost_mag,
        engine.last_live_basin_pressure,
        engine.last_live_motif_radius,
        engine.physics_blend,
        engine.dynamic_repulsion,
        engine.stress_level,
        engine.boredom_level,
        engine.empathy_spike,
        engine.adrenaline,
        engine.last_motif_mag,
        engine.last_recovery_mag,
        engine.last_absence_signal,
        engine.last_trap_score,
        engine.last_hidden_request_pressure,
        engine
            .hidden_request_candidate
            .map(|req| req.as_str())
            .unwrap_or("-"),
        engine
            .last_hidden_request
            .map(|req| req.as_str())
            .unwrap_or("-"),
        if engine.last_guardrail_active { "on" } else { "off" }
    );
}

pub(crate) fn print_metric_postmortem(audit: &RuntimeMetricAudit, goal_active: bool) {
    let channels = [
        ("gravity", &audit.gravity, false),
        ("ghost_pre_norm", &audit.ghost_pre_norm, false),
        ("ghost_applied", &audit.ghost_applied, false),
        ("goal", &audit.goal, !goal_active),
        ("repulsion", &audit.repulsion, false),
        ("motif", &audit.motif, false),
        ("recovery", &audit.recovery, false),
        ("absence", &audit.absence, false),
        ("trap", &audit.trap, false),
        ("live_basin", &audit.live_basin, false),
        ("guardrail", &audit.guardrail, false),
        ("stress", &audit.stress, false),
        ("boredom", &audit.boredom, false),
        ("adrenaline", &audit.adrenaline, false),
        ("blend", &audit.physics_blend, false),
        ("dynamic_repulsion", &audit.dynamic_repulsion, false),
        ("activation_gate", &audit.activation_gate, false),
        (
            "hidden_request_pressure",
            &audit.hidden_request_pressure,
            false,
        ),
    ];

    println!("[METRIC_SUMMARY] ----");
    for (name, stats, inactive) in channels {
        println!(
            "[METRIC_SUMMARY] {} min={:.3} max={:.3} mean={:.3} unique={} class={}",
            name,
            stats.min,
            stats.max,
            stats.mean(),
            stats.unique_count(),
            classify_metric_channel(stats, inactive)
        );
    }
}
