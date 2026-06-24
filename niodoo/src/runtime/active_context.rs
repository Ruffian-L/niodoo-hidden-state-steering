use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const ACTIVE_CONTEXT_ADAPTER_ID: &str = "active_context_runtime_adapter_check_v1";

const FORBIDDEN_METADATA_KEYS: &[&str] = &[
    "answer",
    "answer_text",
    "assistant_answer",
    "completion",
    "correct_answer",
    "diagnostic_labels",
    "diagnostics",
    "expected_answer",
    "final_answer",
    "ground_truth",
    "model_answer",
    "payload",
    "payload_text",
    "prompt",
    "prompt_excerpt",
    "semantic_query_summary",
    "target_answer",
    "text",
    "text_injection",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveContextRuntimeDecisionKind {
    NoopObserve,
    ContextGuard,
    EvidenceGuard,
    SuppressStalePath,
    RouteSteerShadow,
    PreserveState,
    AskForClarification,
}

impl ActiveContextRuntimeDecisionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoopObserve => "noop_observe",
            Self::ContextGuard => "context_guard",
            Self::EvidenceGuard => "evidence_guard",
            Self::SuppressStalePath => "suppress_stale_path",
            Self::RouteSteerShadow => "route_steer_shadow",
            Self::PreserveState => "preserve_state",
            Self::AskForClarification => "ask_for_clarification",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveContextPacketRef {
    pub packet_id: String,
    pub task_family: String,
    pub score: f32,
    pub may_steer: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveContextTurnStartPacketRef {
    pub active_context_id: String,
    pub input_hash: String,
    pub source_control_action: String,
    pub runtime_decision: ActiveContextRuntimeDecisionKind,
    pub packet_id: String,
    pub task_family: String,
    pub score: f32,
    pub may_steer: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveContextAdapterGates {
    pub control_metadata_only: bool,
    pub text_injection_allowed: bool,
    pub external_embedding_model_allowed: bool,
    pub behavior_integration_applied: bool,
    pub same_model_route_geometry_required_for_shadow_steer: bool,
    pub route_family_compatible_required_for_shadow_steer: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveContextRuntimeDecision {
    pub adapter_id: String,
    pub active_context_id: String,
    pub input_hash: String,
    pub source_control_action: String,
    pub runtime_decision: ActiveContextRuntimeDecisionKind,
    pub confidence: f32,
    #[serde(default)]
    pub reason_codes: Vec<String>,
    pub route_family_ref_count: usize,
    pub memory_control_ref_count: usize,
    pub guard_ref_count: usize,
    #[serde(default)]
    pub selected_packet_refs: Vec<ActiveContextPacketRef>,
    pub adapter_gates: ActiveContextAdapterGates,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextDecisionSummary {
    pub record_count: usize,
    pub decision_counts: BTreeMap<String, usize>,
    pub shadow_steer_count: usize,
    pub selected_packet_ref_count: usize,
    pub all_control_metadata_only: bool,
    pub text_injection_allowed_any: bool,
    pub external_embedding_model_allowed_any: bool,
    pub behavior_integration_applied_any: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextRuntimeMetadataDiagnostic {
    pub adapter_id: &'static str,
    pub record_count: usize,
    pub shadow_steer_count: usize,
    pub selected_packet_ref_count: usize,
    pub all_control_metadata_only: bool,
    pub text_injection_allowed_any: bool,
    pub external_embedding_model_allowed_any: bool,
    pub behavior_integration_applied_any: bool,
    pub model_load: bool,
    pub model_generation: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextRuntimeStartupSummary {
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub adapter_decisions_loaded: bool,
    pub runtime_metadata: ActiveContextRuntimeMetadataDiagnostic,
    pub startup_record_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub model_load: bool,
    pub model_generation: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextRuntimeStartupTelemetryRecord {
    pub record_type: &'static str,
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub adapter_decisions_loaded: bool,
    pub runtime_metadata: ActiveContextRuntimeMetadataDiagnostic,
    pub generation_startup_record: bool,
    pub read_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextRuntimeTurnStartState {
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub active_context_record_count: usize,
    pub runtime_metadata: ActiveContextRuntimeMetadataDiagnostic,
    pub turn_start_record_only: bool,
    pub read_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
    pub route_steer_shadow_available: bool,
    pub selected_packet_refs: Vec<ActiveContextPacketRef>,
    pub selected_packet_envelopes: Vec<ActiveContextTurnStartPacketRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextObserveOnlyPacketReload {
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub source: String,
    pub loaded: bool,
    pub read_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
    pub selected_packet_ref_count: usize,
    pub selected_packet_refs: Vec<ActiveContextTurnStartPacketRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextObserveOnlyDecisionSummary {
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub source: String,
    pub loaded_packet_ref_count: usize,
    pub matched_packet_ref_count: usize,
    pub unmatched_packet_ref_count: usize,
    pub matched_decision_count: usize,
    pub route_steer_shadow_decision_count: usize,
    pub all_reloaded_refs_matched: bool,
    pub read_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
    pub observed_runtime_decisions: Vec<ActiveContextRuntimeDecisionKind>,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextReasonCodeFamilyCount {
    pub family: String,
    pub count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextObserveOnlyTurnAggregate {
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub source: String,
    pub matched_decision_count: usize,
    pub recommended_action_counts: BTreeMap<String, usize>,
    pub runtime_decision_counts: BTreeMap<String, usize>,
    pub top_reason_code_families: Vec<ActiveContextReasonCodeFamilyCount>,
    pub read_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ActiveContextShadowSteeringReadiness {
    pub surface_id: &'static str,
    pub adapter_id: &'static str,
    pub source: String,
    pub shadow_steering_ready: bool,
    pub selected_packet_ref_count: usize,
    pub route_steer_shadow_decision_count: usize,
    pub recommended_steer_count: usize,
    pub safety_gate_count: usize,
    pub failed_gate_count: usize,
    pub read_only: bool,
    pub prompt_text_injected: bool,
    pub final_answer_injected: bool,
    pub answer_scoring: bool,
    pub runtime_steering_applied: bool,
    pub reason_codes: Vec<String>,
}

pub fn load_runtime_adapter_decisions(path: &Path) -> Result<Vec<ActiveContextRuntimeDecision>> {
    let raw = fs::read_to_string(path).with_context(|| {
        format!(
            "Failed to read Active Context runtime adapter decisions {}",
            path.display()
        )
    })?;
    let mut decisions = Vec::new();
    for (line_index, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "Failed to parse Active Context adapter decision JSON at {}:{}",
                path.display(),
                line_index + 1
            )
        })?;
        let forbidden_paths = find_forbidden_metadata_keys(&value);
        if !forbidden_paths.is_empty() {
            anyhow::bail!(
                "Active Context adapter decision {}:{} contains forbidden runtime metadata keys: {}",
                path.display(),
                line_index + 1,
                forbidden_paths.join(", ")
            );
        }
        let decision: ActiveContextRuntimeDecision =
            serde_json::from_value(value).with_context(|| {
                format!(
                    "Failed to decode Active Context adapter decision at {}:{}",
                    path.display(),
                    line_index + 1
                )
            })?;
        validate_runtime_adapter_decision(&decision).with_context(|| {
            format!(
                "Invalid Active Context adapter decision at {}:{}",
                path.display(),
                line_index + 1
            )
        })?;
        decisions.push(decision);
    }
    Ok(decisions)
}

pub fn observe_only_packet_refs_from_metadata_value(
    source: &str,
    value: &Value,
) -> Result<ActiveContextObserveOnlyPacketReload> {
    let payload = value
        .get("payload")
        .and_then(Value::as_object)
        .map(|object| Value::Object(object.clone()))
        .unwrap_or_else(|| value.clone());
    let forbidden_paths = find_forbidden_metadata_keys(&payload);
    if !forbidden_paths.is_empty() {
        anyhow::bail!(
            "Active Context observe-only metadata source '{}' contains forbidden keys: {}",
            source,
            forbidden_paths.join(", ")
        );
    }
    let object = payload
        .as_object()
        .context("Active Context observe-only metadata payload is not an object")?;
    if !object
        .get("active_context_turn_start_record_only")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        anyhow::bail!("Active Context observe-only metadata is not record-only");
    }
    if !object
        .get("active_context_read_only")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        anyhow::bail!("Active Context observe-only metadata is not read-only");
    }
    if object
        .get("active_context_prompt_text_injected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        anyhow::bail!("Active Context observe-only metadata claims prompt text injection");
    }
    if object
        .get("active_context_final_answer_injected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        anyhow::bail!("Active Context observe-only metadata claims final-answer injection");
    }
    if object
        .get("active_context_answer_scoring")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        anyhow::bail!("Active Context observe-only metadata claims answer scoring");
    }
    if object
        .get("active_context_runtime_steering_applied")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        anyhow::bail!("Active Context observe-only metadata claims runtime steering");
    }

    let refs_value = object
        .get("active_context_selected_packet_refs")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    let selected_packet_refs: Vec<ActiveContextTurnStartPacketRef> =
        serde_json::from_value(refs_value)
            .context("Failed to decode Active Context structured packet refs")?;
    for packet in &selected_packet_refs {
        validate_turn_start_packet_ref(packet)?;
    }

    if let Some(declared_count) = object
        .get("active_context_selected_packet_ref_count")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
    {
        if declared_count != selected_packet_refs.len() {
            anyhow::bail!(
                "Active Context selected packet ref count mismatch: declared {}, loaded {}",
                declared_count,
                selected_packet_refs.len()
            );
        }
    }
    if let Some(ids) = object
        .get("active_context_selected_packet_ids")
        .and_then(Value::as_array)
    {
        let envelope_ids = selected_packet_refs
            .iter()
            .map(|packet| packet.packet_id.as_str())
            .collect::<Vec<_>>();
        let declared_ids = ids
            .iter()
            .map(|item| item.as_str().unwrap_or(""))
            .collect::<Vec<_>>();
        if declared_ids != envelope_ids {
            anyhow::bail!("Active Context selected packet ids do not match structured refs");
        }
    }

    Ok(ActiveContextObserveOnlyPacketReload {
        surface_id: "active_context_observe_only_packet_reload_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        source: source.to_string(),
        loaded: true,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        selected_packet_ref_count: selected_packet_refs.len(),
        selected_packet_refs,
    })
}

pub fn observe_only_runtime_decision_summary(
    source: &str,
    reload: &ActiveContextObserveOnlyPacketReload,
    decisions: &[ActiveContextRuntimeDecision],
) -> Result<ActiveContextObserveOnlyDecisionSummary> {
    if !reload.read_only {
        anyhow::bail!(
            "Active Context observe-only decision summary received writable reload state"
        );
    }
    if reload.prompt_text_injected
        || reload.final_answer_injected
        || reload.answer_scoring
        || reload.runtime_steering_applied
    {
        anyhow::bail!("Active Context observe-only decision summary received unsafe reload flags");
    }

    for decision in decisions {
        validate_runtime_adapter_decision(decision)?;
    }
    for packet in &reload.selected_packet_refs {
        validate_turn_start_packet_ref(packet)?;
    }

    let route_steer_shadow_decision_count = decisions
        .iter()
        .filter(|decision| {
            decision.runtime_decision == ActiveContextRuntimeDecisionKind::RouteSteerShadow
        })
        .count();
    let mut observed_runtime_decisions = Vec::new();
    let mut matched_decision_keys = BTreeMap::new();
    let mut matched_packet_ref_count = 0usize;

    for reloaded_packet in &reload.selected_packet_refs {
        let matched_decision = decisions.iter().find(|decision| {
            decision.runtime_decision == ActiveContextRuntimeDecisionKind::RouteSteerShadow
                && decision.active_context_id == reloaded_packet.active_context_id
                && decision.input_hash == reloaded_packet.input_hash
                && decision.source_control_action == reloaded_packet.source_control_action
                && decision.selected_packet_refs.iter().any(|candidate| {
                    candidate.packet_id == reloaded_packet.packet_id
                        && candidate.task_family == reloaded_packet.task_family
                        && candidate.may_steer == reloaded_packet.may_steer
                        && (candidate.score - reloaded_packet.score).abs() <= f32::EPSILON
                })
        });
        if let Some(decision) = matched_decision {
            matched_packet_ref_count += 1;
            observed_runtime_decisions.push(decision.runtime_decision);
            matched_decision_keys.insert(
                format!("{}::{}", decision.active_context_id, decision.input_hash),
                true,
            );
        }
    }

    let unmatched_packet_ref_count = reload
        .selected_packet_refs
        .len()
        .saturating_sub(matched_packet_ref_count);
    let all_reloaded_refs_matched = reload.loaded
        && unmatched_packet_ref_count == 0
        && reload.selected_packet_ref_count == reload.selected_packet_refs.len();
    let mut reason_codes = vec![
        "observe_only".to_string(),
        "decision_lookup_by_context_hash_packet_and_family".to_string(),
        "no_prompt_or_answer_payload".to_string(),
        "no_runtime_steering_applied".to_string(),
    ];
    if all_reloaded_refs_matched {
        reason_codes.push("all_reloaded_packet_refs_matched_adapter_decisions".to_string());
    } else {
        reason_codes.push("some_reloaded_packet_refs_unmatched".to_string());
    }

    Ok(ActiveContextObserveOnlyDecisionSummary {
        surface_id: "active_context_observe_only_decision_summary_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        source: source.to_string(),
        loaded_packet_ref_count: reload.selected_packet_ref_count,
        matched_packet_ref_count,
        unmatched_packet_ref_count,
        matched_decision_count: matched_decision_keys.len(),
        route_steer_shadow_decision_count,
        all_reloaded_refs_matched,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        observed_runtime_decisions,
        reason_codes,
    })
}

pub fn observe_only_turn_aggregate(
    source: &str,
    decisions: &[ActiveContextRuntimeDecision],
) -> Result<ActiveContextObserveOnlyTurnAggregate> {
    for decision in decisions {
        validate_runtime_adapter_decision(decision)?;
    }

    let mut recommended_action_counts = BTreeMap::new();
    let mut runtime_decision_counts = BTreeMap::new();
    let mut reason_family_counts = BTreeMap::new();
    for decision in decisions {
        *recommended_action_counts
            .entry(decision.source_control_action.clone())
            .or_insert(0) += 1;
        *runtime_decision_counts
            .entry(decision.runtime_decision.as_str().to_string())
            .or_insert(0) += 1;
        for code in &decision.reason_codes {
            *reason_family_counts
                .entry(reason_code_family(code).to_string())
                .or_insert(0) += 1;
        }
    }

    let mut top_reason_code_families = reason_family_counts
        .into_iter()
        .map(|(family, count)| ActiveContextReasonCodeFamilyCount { family, count })
        .collect::<Vec<_>>();
    top_reason_code_families.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.family.cmp(&right.family))
    });
    top_reason_code_families.truncate(8);

    let mut reason_codes = vec![
        "observe_only".to_string(),
        "recommended_action_counts_from_adapter_decisions".to_string(),
        "reason_code_families_from_adapter_decisions".to_string(),
        "no_prompt_or_answer_payload".to_string(),
        "no_runtime_steering_applied".to_string(),
    ];
    if top_reason_code_families.is_empty() {
        reason_codes.push("no_adapter_reason_codes_present".to_string());
    }

    Ok(ActiveContextObserveOnlyTurnAggregate {
        surface_id: "active_context_observe_only_turn_aggregate_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        source: source.to_string(),
        matched_decision_count: decisions.len(),
        recommended_action_counts,
        runtime_decision_counts,
        top_reason_code_families,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        reason_codes,
    })
}

pub fn observe_only_shadow_steering_readiness(
    source: &str,
    turn_start: &ActiveContextRuntimeTurnStartState,
    aggregate: &ActiveContextObserveOnlyTurnAggregate,
) -> Result<ActiveContextShadowSteeringReadiness> {
    if !turn_start.read_only || !aggregate.read_only {
        anyhow::bail!("Active Context shadow steering readiness received writable metadata");
    }
    if turn_start.prompt_text_injected
        || turn_start.final_answer_injected
        || turn_start.answer_scoring
        || turn_start.runtime_steering_applied
        || aggregate.prompt_text_injected
        || aggregate.final_answer_injected
        || aggregate.answer_scoring
        || aggregate.runtime_steering_applied
    {
        anyhow::bail!("Active Context shadow steering readiness received unsafe metadata flags");
    }
    for packet in &turn_start.selected_packet_envelopes {
        validate_turn_start_packet_ref(packet)?;
    }

    let selected_packet_ref_count = turn_start.selected_packet_envelopes.len();
    let route_steer_shadow_decision_count = aggregate
        .runtime_decision_counts
        .get(ActiveContextRuntimeDecisionKind::RouteSteerShadow.as_str())
        .copied()
        .unwrap_or(0);
    let recommended_steer_count = aggregate
        .recommended_action_counts
        .get("steer")
        .copied()
        .unwrap_or(0);
    let safety_gates = [
        (
            "turn_start_loaded",
            turn_start.active_context_record_count > 0 && aggregate.matched_decision_count > 0,
        ),
        ("read_only", turn_start.read_only && aggregate.read_only),
        (
            "no_prompt_or_answer_payload",
            !turn_start.prompt_text_injected
                && !turn_start.final_answer_injected
                && !aggregate.prompt_text_injected
                && !aggregate.final_answer_injected,
        ),
        (
            "no_answer_scoring",
            !turn_start.answer_scoring && !aggregate.answer_scoring,
        ),
        (
            "no_runtime_steering_applied",
            !turn_start.runtime_steering_applied && !aggregate.runtime_steering_applied,
        ),
        (
            "route_steer_shadow_available",
            turn_start.route_steer_shadow_available
                && selected_packet_ref_count > 0
                && route_steer_shadow_decision_count > 0,
        ),
        ("recommended_steer_present", recommended_steer_count > 0),
        (
            "same_model_route_packet_refs_present",
            turn_start
                .selected_packet_envelopes
                .iter()
                .all(|packet| packet.may_steer),
        ),
    ];
    let safety_gate_count = safety_gates.len();

    let mut reason_codes = vec![
        "observe_only".to_string(),
        "shadow_steering_readiness_metadata_only".to_string(),
        "no_prompt_or_answer_payload".to_string(),
        "no_runtime_steering_applied".to_string(),
    ];
    let mut failed_gate_count = 0usize;
    for (gate, passed) in safety_gates {
        if passed {
            reason_codes.push(format!("gate_pass:{gate}"));
        } else {
            failed_gate_count += 1;
            reason_codes.push(format!("gate_fail:{gate}"));
        }
    }

    Ok(ActiveContextShadowSteeringReadiness {
        surface_id: "active_context_shadow_steering_readiness_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        source: source.to_string(),
        shadow_steering_ready: failed_gate_count == 0,
        selected_packet_ref_count,
        route_steer_shadow_decision_count,
        recommended_steer_count,
        safety_gate_count,
        failed_gate_count,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        reason_codes,
    })
}

pub fn summarize_runtime_adapter_decisions(
    decisions: &[ActiveContextRuntimeDecision],
) -> ActiveContextDecisionSummary {
    let mut decision_counts = BTreeMap::new();
    let mut selected_packet_ref_count = 0usize;
    for decision in decisions {
        *decision_counts
            .entry(decision.runtime_decision.as_str().to_string())
            .or_insert(0) += 1;
        selected_packet_ref_count += decision.selected_packet_refs.len();
    }
    ActiveContextDecisionSummary {
        record_count: decisions.len(),
        shadow_steer_count: decisions
            .iter()
            .filter(|decision| {
                decision.runtime_decision == ActiveContextRuntimeDecisionKind::RouteSteerShadow
            })
            .count(),
        selected_packet_ref_count,
        decision_counts,
        all_control_metadata_only: decisions
            .iter()
            .all(|decision| decision.adapter_gates.control_metadata_only),
        text_injection_allowed_any: decisions
            .iter()
            .any(|decision| decision.adapter_gates.text_injection_allowed),
        external_embedding_model_allowed_any: decisions
            .iter()
            .any(|decision| decision.adapter_gates.external_embedding_model_allowed),
        behavior_integration_applied_any: decisions
            .iter()
            .any(|decision| decision.adapter_gates.behavior_integration_applied),
    }
}

pub fn runtime_metadata_diagnostic(
    summary: &ActiveContextDecisionSummary,
) -> ActiveContextRuntimeMetadataDiagnostic {
    ActiveContextRuntimeMetadataDiagnostic {
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        record_count: summary.record_count,
        shadow_steer_count: summary.shadow_steer_count,
        selected_packet_ref_count: summary.selected_packet_ref_count,
        all_control_metadata_only: summary.all_control_metadata_only,
        text_injection_allowed_any: summary.text_injection_allowed_any,
        external_embedding_model_allowed_any: summary.external_embedding_model_allowed_any,
        behavior_integration_applied_any: summary.behavior_integration_applied_any,
        model_load: false,
        model_generation: false,
        answer_scoring: false,
        runtime_steering_applied: false,
    }
}

pub fn runtime_startup_summary(
    diagnostic: ActiveContextRuntimeMetadataDiagnostic,
) -> ActiveContextRuntimeStartupSummary {
    ActiveContextRuntimeStartupSummary {
        surface_id: "active_context_runtime_startup_summary_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        adapter_decisions_loaded: true,
        runtime_metadata: diagnostic,
        startup_record_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        model_load: false,
        model_generation: false,
        answer_scoring: false,
        runtime_steering_applied: false,
    }
}

pub fn runtime_startup_telemetry_record(
    diagnostic: ActiveContextRuntimeMetadataDiagnostic,
) -> ActiveContextRuntimeStartupTelemetryRecord {
    ActiveContextRuntimeStartupTelemetryRecord {
        record_type: "active_context_startup",
        surface_id: "active_context_runtime_startup_telemetry_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        adapter_decisions_loaded: true,
        runtime_metadata: diagnostic,
        generation_startup_record: true,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
    }
}

pub fn runtime_turn_start_state(
    decisions: &[ActiveContextRuntimeDecision],
) -> ActiveContextRuntimeTurnStartState {
    let summary = summarize_runtime_adapter_decisions(decisions);
    let diagnostic = runtime_metadata_diagnostic(&summary);
    let mut selected_packet_refs = Vec::new();
    let mut selected_packet_envelopes = Vec::new();
    for decision in decisions {
        if decision.runtime_decision == ActiveContextRuntimeDecisionKind::RouteSteerShadow {
            for packet in &decision.selected_packet_refs {
                selected_packet_refs.push(packet.clone());
                selected_packet_envelopes.push(ActiveContextTurnStartPacketRef {
                    active_context_id: decision.active_context_id.clone(),
                    input_hash: decision.input_hash.clone(),
                    source_control_action: decision.source_control_action.clone(),
                    runtime_decision: decision.runtime_decision,
                    packet_id: packet.packet_id.clone(),
                    task_family: packet.task_family.clone(),
                    score: packet.score,
                    may_steer: packet.may_steer,
                });
            }
        }
    }
    ActiveContextRuntimeTurnStartState {
        surface_id: "active_context_runtime_turn_start_state_v1",
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID,
        active_context_record_count: decisions.len(),
        runtime_metadata: diagnostic,
        turn_start_record_only: true,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        route_steer_shadow_available: !selected_packet_refs.is_empty(),
        selected_packet_refs,
        selected_packet_envelopes,
    }
}

fn validate_runtime_adapter_decision(decision: &ActiveContextRuntimeDecision) -> Result<()> {
    if decision.adapter_id != ACTIVE_CONTEXT_ADAPTER_ID {
        anyhow::bail!("unexpected adapter_id '{}'", decision.adapter_id);
    }
    if decision.active_context_id.trim().is_empty() {
        anyhow::bail!("active_context_id is empty");
    }
    if decision.input_hash.trim().is_empty() {
        anyhow::bail!("input_hash is empty");
    }
    if !(0.0..=1.0).contains(&decision.confidence) {
        anyhow::bail!("confidence outside [0, 1]");
    }
    let gates = &decision.adapter_gates;
    if !gates.control_metadata_only {
        anyhow::bail!("adapter decision is not marked control_metadata_only");
    }
    if gates.text_injection_allowed {
        anyhow::bail!("adapter decision allows text injection");
    }
    if gates.external_embedding_model_allowed {
        anyhow::bail!("adapter decision allows an external embedding model");
    }
    if gates.behavior_integration_applied {
        anyhow::bail!("adapter decision claims behavior integration was applied");
    }
    if !gates.same_model_route_geometry_required_for_shadow_steer {
        anyhow::bail!("shadow steering does not require same-model route geometry");
    }
    if !gates.route_family_compatible_required_for_shadow_steer {
        anyhow::bail!("shadow steering does not require route-family compatibility");
    }

    for packet in &decision.selected_packet_refs {
        if packet.packet_id.trim().is_empty() {
            anyhow::bail!("selected packet ref missing packet_id");
        }
        if packet.task_family.trim().is_empty() {
            anyhow::bail!("selected packet ref missing task_family");
        }
        if !(0.0..=1.0).contains(&packet.score) {
            anyhow::bail!("selected packet score outside [0, 1]");
        }
    }

    if decision.runtime_decision == ActiveContextRuntimeDecisionKind::RouteSteerShadow {
        if decision.selected_packet_refs.is_empty() {
            anyhow::bail!("route_steer_shadow decision has no selected packet refs");
        }
        if decision
            .selected_packet_refs
            .iter()
            .any(|packet| !packet.may_steer)
        {
            anyhow::bail!("route_steer_shadow selected packet has may_steer=false");
        }
    } else if !decision.selected_packet_refs.is_empty() {
        anyhow::bail!("non-route_steer_shadow decision includes selected packet refs");
    }
    Ok(())
}

fn validate_turn_start_packet_ref(packet: &ActiveContextTurnStartPacketRef) -> Result<()> {
    if packet.active_context_id.trim().is_empty() {
        anyhow::bail!("selected packet ref missing active_context_id");
    }
    if packet.input_hash.trim().is_empty() {
        anyhow::bail!("selected packet ref missing input_hash");
    }
    if packet.source_control_action.trim().is_empty() {
        anyhow::bail!("selected packet ref missing source_control_action");
    }
    if packet.packet_id.trim().is_empty() {
        anyhow::bail!("selected packet ref missing packet_id");
    }
    if packet.task_family.trim().is_empty() {
        anyhow::bail!("selected packet ref missing task_family");
    }
    if !(0.0..=1.0).contains(&packet.score) {
        anyhow::bail!("selected packet score outside [0, 1]");
    }
    if packet.runtime_decision != ActiveContextRuntimeDecisionKind::RouteSteerShadow {
        anyhow::bail!("selected packet ref is not from route_steer_shadow");
    }
    if !packet.may_steer {
        anyhow::bail!("selected packet ref has may_steer=false");
    }
    Ok(())
}

fn find_forbidden_metadata_keys(value: &Value) -> Vec<String> {
    let mut paths = Vec::new();
    collect_forbidden_metadata_keys(value, "$", &mut paths);
    paths
}

fn collect_forbidden_metadata_keys(value: &Value, path: &str, out: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let child_path = format!("{path}.{key}");
                if FORBIDDEN_METADATA_KEYS.contains(&key.as_str()) {
                    out.push(child_path.clone());
                }
                collect_forbidden_metadata_keys(child, &child_path, out);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                collect_forbidden_metadata_keys(child, &format!("{path}[{index}]"), out);
            }
        }
        _ => {}
    }
}

fn reason_code_family(code: &str) -> &str {
    code.split_once(':')
        .map(|(family, _)| family)
        .unwrap_or(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_jsonl(name: &str, body: &str) -> std::path::PathBuf {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let path = std::env::temp_dir().join(format!("{name}_{millis}.jsonl"));
        fs::write(&path, body).unwrap();
        path
    }

    fn valid_decision_json(runtime_decision: &str, selected_packet_refs: &str) -> String {
        serde_json::json!({
            "adapter_id": "active_context_runtime_adapter_check_v1",
            "active_context_id": "acl_v1::tool_verification_policy",
            "input_hash": "ef8d51e243bcb3ee",
            "source_control_action": "steer",
            "runtime_decision": runtime_decision,
            "confidence": 0.6178,
            "reason_codes": [
                "memory_slice_candidate_above_threshold",
                "route_packet_candidate_available",
                "hold:family_top1_lift_below_0.20"
            ],
            "route_family_ref_count": 3,
            "memory_control_ref_count": 3,
            "guard_ref_count": 4,
            "selected_packet_refs": serde_json::from_str::<Value>(selected_packet_refs).unwrap(),
            "adapter_gates": {
                "control_metadata_only": true,
                "text_injection_allowed": false,
                "external_embedding_model_allowed": false,
                "behavior_integration_applied": false,
                "same_model_route_geometry_required_for_shadow_steer": true,
                "route_family_compatible_required_for_shadow_steer": true
            }
        })
        .to_string()
    }

    #[test]
    fn loads_control_metadata_only_adapter_decisions() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_valid",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let summary = summarize_runtime_adapter_decisions(&decisions);
        let _ = fs::remove_file(path);

        assert_eq!(decisions.len(), 1);
        assert_eq!(summary.record_count, 1);
        assert_eq!(summary.shadow_steer_count, 1);
        assert_eq!(summary.selected_packet_ref_count, 1);
        assert!(summary.all_control_metadata_only);
        assert!(!summary.behavior_integration_applied_any);
        assert!(!summary.text_injection_allowed_any);
    }

    #[test]
    fn builds_runtime_metadata_diagnostic_without_behavior_claims() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_diagnostic",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let summary = summarize_runtime_adapter_decisions(&decisions);
        let diagnostic = runtime_metadata_diagnostic(&summary);
        let _ = fs::remove_file(path);

        assert_eq!(diagnostic.adapter_id, ACTIVE_CONTEXT_ADAPTER_ID);
        assert_eq!(diagnostic.record_count, 1);
        assert_eq!(diagnostic.shadow_steer_count, 1);
        assert_eq!(diagnostic.selected_packet_ref_count, 1);
        assert!(diagnostic.all_control_metadata_only);
        assert!(!diagnostic.text_injection_allowed_any);
        assert!(!diagnostic.external_embedding_model_allowed_any);
        assert!(!diagnostic.behavior_integration_applied_any);
        assert!(!diagnostic.model_load);
        assert!(!diagnostic.model_generation);
        assert!(!diagnostic.answer_scoring);
        assert!(!diagnostic.runtime_steering_applied);
    }

    #[test]
    fn builds_disabled_startup_summary_without_behavior_claims() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_startup_summary",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let summary = summarize_runtime_adapter_decisions(&decisions);
        let diagnostic = runtime_metadata_diagnostic(&summary);
        let startup = runtime_startup_summary(diagnostic);
        let _ = fs::remove_file(path);

        assert_eq!(
            startup.surface_id,
            "active_context_runtime_startup_summary_v1"
        );
        assert_eq!(startup.adapter_id, ACTIVE_CONTEXT_ADAPTER_ID);
        assert!(startup.adapter_decisions_loaded);
        assert!(startup.startup_record_only);
        assert_eq!(startup.runtime_metadata.record_count, 1);
        assert_eq!(startup.runtime_metadata.shadow_steer_count, 1);
        assert!(!startup.prompt_text_injected);
        assert!(!startup.final_answer_injected);
        assert!(!startup.model_load);
        assert!(!startup.model_generation);
        assert!(!startup.answer_scoring);
        assert!(!startup.runtime_steering_applied);
    }

    #[test]
    fn builds_startup_telemetry_record_without_behavior_claims() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_startup_telemetry",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let summary = summarize_runtime_adapter_decisions(&decisions);
        let diagnostic = runtime_metadata_diagnostic(&summary);
        let telemetry = runtime_startup_telemetry_record(diagnostic);
        let _ = fs::remove_file(path);

        assert_eq!(telemetry.record_type, "active_context_startup");
        assert_eq!(
            telemetry.surface_id,
            "active_context_runtime_startup_telemetry_v1"
        );
        assert_eq!(telemetry.adapter_id, ACTIVE_CONTEXT_ADAPTER_ID);
        assert!(telemetry.adapter_decisions_loaded);
        assert!(telemetry.generation_startup_record);
        assert!(telemetry.read_only);
        assert_eq!(telemetry.runtime_metadata.record_count, 1);
        assert_eq!(telemetry.runtime_metadata.shadow_steer_count, 1);
        assert!(!telemetry.prompt_text_injected);
        assert!(!telemetry.final_answer_injected);
        assert!(!telemetry.answer_scoring);
        assert!(!telemetry.runtime_steering_applied);
        assert!(!telemetry.runtime_metadata.model_generation);
        assert!(!telemetry.runtime_metadata.runtime_steering_applied);
    }

    #[test]
    fn builds_turn_start_state_without_prompt_or_steering_integration() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_turn_start",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let state = runtime_turn_start_state(&decisions);
        let _ = fs::remove_file(path);

        assert_eq!(
            state.surface_id,
            "active_context_runtime_turn_start_state_v1"
        );
        assert_eq!(state.adapter_id, ACTIVE_CONTEXT_ADAPTER_ID);
        assert_eq!(state.active_context_record_count, 1);
        assert_eq!(state.runtime_metadata.record_count, 1);
        assert_eq!(state.runtime_metadata.shadow_steer_count, 1);
        assert_eq!(state.selected_packet_refs.len(), 1);
        assert_eq!(state.selected_packet_envelopes.len(), 1);
        assert_eq!(
            state.selected_packet_envelopes[0].active_context_id,
            "acl_v1::tool_verification_policy"
        );
        assert_eq!(
            state.selected_packet_envelopes[0].input_hash,
            "ef8d51e243bcb3ee"
        );
        assert_eq!(
            state.selected_packet_envelopes[0].packet_id,
            "scalar::scalar_int8_roundtrip::step0000::4919441a91b3"
        );
        assert_eq!(
            state.selected_packet_envelopes[0].runtime_decision,
            ActiveContextRuntimeDecisionKind::RouteSteerShadow
        );
        assert!(state.route_steer_shadow_available);
        assert!(state.turn_start_record_only);
        assert!(state.read_only);
        assert!(!state.prompt_text_injected);
        assert!(!state.final_answer_injected);
        assert!(!state.answer_scoring);
        assert!(!state.runtime_steering_applied);
        assert!(!state.runtime_metadata.model_load);
        assert!(!state.runtime_metadata.model_generation);
        assert!(!state.runtime_metadata.runtime_steering_applied);
    }

    #[test]
    fn builds_turn_start_packet_envelopes_without_prompt_payloads() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_turn_start_envelope",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let state = runtime_turn_start_state(&decisions);
        let value = serde_json::to_value(&state.selected_packet_envelopes).unwrap();
        let forbidden_paths = find_forbidden_metadata_keys(&value);
        let _ = fs::remove_file(path);

        assert_eq!(
            state.selected_packet_refs.len(),
            state.selected_packet_envelopes.len()
        );
        assert!(forbidden_paths.is_empty());
        assert!(state
            .selected_packet_envelopes
            .iter()
            .all(|packet| packet.may_steer));
        assert!(state
            .selected_packet_envelopes
            .iter()
            .all(|packet| packet.runtime_decision
                == ActiveContextRuntimeDecisionKind::RouteSteerShadow));
    }

    #[test]
    fn reloads_turn_start_packet_refs_as_observe_only_state() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_reload_turn_start",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let state = runtime_turn_start_state(&decisions);
        let event = serde_json::json!({
            "event": "turn_start",
            "payload": {
                "active_context_turn_start_record_only": state.turn_start_record_only,
                "active_context_read_only": state.read_only,
                "active_context_prompt_text_injected": state.prompt_text_injected,
                "active_context_final_answer_injected": state.final_answer_injected,
                "active_context_answer_scoring": state.answer_scoring,
                "active_context_runtime_steering_applied": state.runtime_steering_applied,
                "active_context_selected_packet_ref_count": state.selected_packet_envelopes.len(),
                "active_context_selected_packet_ids": state
                    .selected_packet_envelopes
                    .iter()
                    .map(|packet| packet.packet_id.as_str())
                    .collect::<Vec<_>>(),
                "active_context_selected_packet_refs": state.selected_packet_envelopes,
            }
        });
        let reloaded =
            observe_only_packet_refs_from_metadata_value("live_turn_start_event", &event).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(
            reloaded.surface_id,
            "active_context_observe_only_packet_reload_v1"
        );
        assert_eq!(reloaded.adapter_id, ACTIVE_CONTEXT_ADAPTER_ID);
        assert_eq!(reloaded.source, "live_turn_start_event");
        assert!(reloaded.loaded);
        assert!(reloaded.read_only);
        assert_eq!(reloaded.selected_packet_ref_count, 1);
        assert_eq!(
            reloaded.selected_packet_refs[0].packet_id,
            "scalar::scalar_int8_roundtrip::step0000::4919441a91b3"
        );
        assert!(!reloaded.prompt_text_injected);
        assert!(!reloaded.final_answer_injected);
        assert!(!reloaded.answer_scoring);
        assert!(!reloaded.runtime_steering_applied);
    }

    #[test]
    fn summarizes_reloaded_packet_refs_against_adapter_decisions_observe_only() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_reload_decision_summary",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let state = runtime_turn_start_state(&decisions);
        let metadata = serde_json::json!({
            "active_context_turn_start_record_only": state.turn_start_record_only,
            "active_context_read_only": state.read_only,
            "active_context_prompt_text_injected": state.prompt_text_injected,
            "active_context_final_answer_injected": state.final_answer_injected,
            "active_context_answer_scoring": state.answer_scoring,
            "active_context_runtime_steering_applied": state.runtime_steering_applied,
            "active_context_selected_packet_ref_count": state.selected_packet_envelopes.len(),
            "active_context_selected_packet_ids": state
                .selected_packet_envelopes
                .iter()
                .map(|packet| packet.packet_id.as_str())
                .collect::<Vec<_>>(),
            "active_context_selected_packet_refs": state.selected_packet_envelopes,
        });
        let reloaded =
            observe_only_packet_refs_from_metadata_value("live_turn_start_metadata", &metadata)
                .unwrap();
        let summary = observe_only_runtime_decision_summary(
            "live_turn_start_metadata",
            &reloaded,
            &decisions,
        )
        .unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(
            summary.surface_id,
            "active_context_observe_only_decision_summary_v1"
        );
        assert_eq!(summary.loaded_packet_ref_count, 1);
        assert_eq!(summary.matched_packet_ref_count, 1);
        assert_eq!(summary.unmatched_packet_ref_count, 0);
        assert_eq!(summary.matched_decision_count, 1);
        assert_eq!(summary.route_steer_shadow_decision_count, 1);
        assert!(summary.all_reloaded_refs_matched);
        assert_eq!(
            summary.observed_runtime_decisions,
            vec![ActiveContextRuntimeDecisionKind::RouteSteerShadow]
        );
        assert!(summary.read_only);
        assert!(!summary.prompt_text_injected);
        assert!(!summary.final_answer_injected);
        assert!(!summary.answer_scoring);
        assert!(!summary.runtime_steering_applied);
        assert!(summary
            .reason_codes
            .iter()
            .any(|code| { code == "all_reloaded_packet_refs_matched_adapter_decisions" }));
    }

    #[test]
    fn decision_summary_marks_unmatched_reloaded_packet_refs_without_steering() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_reload_unmatched_summary",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let mut state = runtime_turn_start_state(&decisions);
        state.selected_packet_envelopes[0]
            .packet_id
            .push_str("::missing");
        let metadata = serde_json::json!({
            "active_context_turn_start_record_only": state.turn_start_record_only,
            "active_context_read_only": state.read_only,
            "active_context_prompt_text_injected": state.prompt_text_injected,
            "active_context_final_answer_injected": state.final_answer_injected,
            "active_context_answer_scoring": state.answer_scoring,
            "active_context_runtime_steering_applied": state.runtime_steering_applied,
            "active_context_selected_packet_ref_count": state.selected_packet_envelopes.len(),
            "active_context_selected_packet_ids": state
                .selected_packet_envelopes
                .iter()
                .map(|packet| packet.packet_id.as_str())
                .collect::<Vec<_>>(),
            "active_context_selected_packet_refs": state.selected_packet_envelopes,
        });
        let reloaded =
            observe_only_packet_refs_from_metadata_value("live_turn_start_metadata", &metadata)
                .unwrap();
        let summary = observe_only_runtime_decision_summary(
            "live_turn_start_metadata",
            &reloaded,
            &decisions,
        )
        .unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(summary.loaded_packet_ref_count, 1);
        assert_eq!(summary.matched_packet_ref_count, 0);
        assert_eq!(summary.unmatched_packet_ref_count, 1);
        assert!(!summary.all_reloaded_refs_matched);
        assert!(summary.read_only);
        assert!(!summary.runtime_steering_applied);
        assert!(summary
            .reason_codes
            .iter()
            .any(|code| code == "some_reloaded_packet_refs_unmatched"));
    }

    #[test]
    fn builds_observe_only_turn_aggregate_action_and_reason_counts() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let body = format!(
            "{}\n{}",
            valid_decision_json("route_steer_shadow", packets),
            valid_decision_json("context_guard", "[]").replace(
                "\"source_control_action\":\"steer\"",
                "\"source_control_action\":\"context\""
            )
        );
        let path = write_temp_jsonl("active_context_turn_aggregate", &body);
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let aggregate =
            observe_only_turn_aggregate("live_turn_start_metadata", &decisions).unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(
            aggregate.surface_id,
            "active_context_observe_only_turn_aggregate_v1"
        );
        assert_eq!(aggregate.matched_decision_count, 2);
        assert_eq!(aggregate.recommended_action_counts.get("steer"), Some(&1));
        assert_eq!(aggregate.recommended_action_counts.get("context"), Some(&1));
        assert_eq!(
            aggregate.runtime_decision_counts.get("route_steer_shadow"),
            Some(&1)
        );
        assert!(aggregate
            .top_reason_code_families
            .iter()
            .any(|row| row.family == "hold" && row.count == 2));
        assert!(aggregate.read_only);
        assert!(!aggregate.prompt_text_injected);
        assert!(!aggregate.final_answer_injected);
        assert!(!aggregate.answer_scoring);
        assert!(!aggregate.runtime_steering_applied);
        assert!(aggregate
            .reason_codes
            .iter()
            .any(|code| code == "recommended_action_counts_from_adapter_decisions"));
    }

    #[test]
    fn builds_shadow_steering_readiness_without_applying_steering() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_shadow_readiness",
            &valid_decision_json("route_steer_shadow", packets),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let turn_start = runtime_turn_start_state(&decisions);
        let aggregate =
            observe_only_turn_aggregate("live_turn_start_metadata", &decisions).unwrap();
        let readiness = observe_only_shadow_steering_readiness(
            "live_turn_start_metadata",
            &turn_start,
            &aggregate,
        )
        .unwrap();
        let _ = fs::remove_file(path);

        assert_eq!(
            readiness.surface_id,
            "active_context_shadow_steering_readiness_v1"
        );
        assert!(readiness.shadow_steering_ready);
        assert_eq!(readiness.selected_packet_ref_count, 1);
        assert_eq!(readiness.route_steer_shadow_decision_count, 1);
        assert_eq!(readiness.recommended_steer_count, 1);
        assert_eq!(readiness.failed_gate_count, 0);
        assert!(readiness.safety_gate_count > 0);
        assert!(readiness.read_only);
        assert!(!readiness.prompt_text_injected);
        assert!(!readiness.final_answer_injected);
        assert!(!readiness.answer_scoring);
        assert!(!readiness.runtime_steering_applied);
        assert!(readiness
            .reason_codes
            .iter()
            .any(|code| code == "shadow_steering_readiness_metadata_only"));
        assert!(readiness
            .reason_codes
            .iter()
            .any(|code| code == "gate_pass:route_steer_shadow_available"));
    }

    #[test]
    fn shadow_steering_readiness_fails_closed_without_shadow_packets() {
        let path = write_temp_jsonl(
            "active_context_shadow_readiness_fail_closed",
            &valid_decision_json("context_guard", "[]").replace(
                "\"source_control_action\":\"steer\"",
                "\"source_control_action\":\"context\"",
            ),
        );
        let decisions = load_runtime_adapter_decisions(&path).unwrap();
        let turn_start = runtime_turn_start_state(&decisions);
        let aggregate =
            observe_only_turn_aggregate("live_turn_start_metadata", &decisions).unwrap();
        let readiness = observe_only_shadow_steering_readiness(
            "live_turn_start_metadata",
            &turn_start,
            &aggregate,
        )
        .unwrap();
        let _ = fs::remove_file(path);

        assert!(!readiness.shadow_steering_ready);
        assert_eq!(readiness.selected_packet_ref_count, 0);
        assert_eq!(readiness.route_steer_shadow_decision_count, 0);
        assert_eq!(readiness.recommended_steer_count, 0);
        assert!(readiness.failed_gate_count > 0);
        assert!(!readiness.runtime_steering_applied);
        assert!(readiness
            .reason_codes
            .iter()
            .any(|code| code == "gate_fail:route_steer_shadow_available"));
    }

    #[test]
    fn rejects_observe_only_packet_ref_reload_with_unsafe_flags() {
        let metadata = serde_json::json!({
            "active_context_turn_start_record_only": true,
            "active_context_read_only": true,
            "active_context_prompt_text_injected": false,
            "active_context_final_answer_injected": false,
            "active_context_answer_scoring": false,
            "active_context_runtime_steering_applied": true,
            "active_context_selected_packet_ref_count": 1,
            "active_context_selected_packet_ids": ["scalar::scalar_int8_roundtrip::step0000::4919441a91b3"],
            "active_context_selected_packet_refs": [{
                "active_context_id": "acl_v1::tool_verification_policy",
                "input_hash": "ef8d51e243bcb3ee",
                "source_control_action": "steer",
                "runtime_decision": "route_steer_shadow",
                "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
                "task_family": "visible-hands/reflex",
                "score": 0.77,
                "may_steer": true
            }]
        });
        let err =
            observe_only_packet_refs_from_metadata_value("compact_resume_metadata", &metadata)
                .unwrap_err();

        assert!(format!("{err:#}").contains("runtime steering"));
    }

    #[test]
    fn rejects_forbidden_text_payload_keys() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["payload_text"] = Value::String("do not pass answer text".to_string());
        let path = write_temp_jsonl("active_context_forbidden", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(err.to_string().contains("forbidden runtime metadata keys"));
    }

    #[test]
    fn rejects_shadow_steer_without_packet_refs() {
        let path = write_temp_jsonl(
            "active_context_missing_packet",
            &valid_decision_json("route_steer_shadow", "[]"),
        );
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("no selected packet refs"));
    }

    #[test]
    fn rejects_non_shadow_decision_with_packet_refs() {
        let packets = r#"[{
            "packet_id": "scalar::scalar_int8_roundtrip::step0000::4919441a91b3",
            "task_family": "visible-hands/reflex",
            "score": 0.77,
            "may_steer": true
        }]"#;
        let path = write_temp_jsonl(
            "active_context_non_shadow_packet",
            &valid_decision_json("context_guard", packets),
        );
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("non-route_steer_shadow"));
    }

    #[test]
    fn rejects_behavior_integration_claims() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["adapter_gates"]["behavior_integration_applied"] = Value::Bool(true);
        let path = write_temp_jsonl("active_context_behavior_claim", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("behavior integration"));
    }

    #[test]
    fn rejects_non_metadata_only_adapter_decisions() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["adapter_gates"]["control_metadata_only"] = Value::Bool(false);
        let path = write_temp_jsonl("active_context_not_metadata_only", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("not marked control_metadata_only"));
    }

    #[test]
    fn rejects_text_injection_allowed_gate() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["adapter_gates"]["text_injection_allowed"] = Value::Bool(true);
        let path = write_temp_jsonl("active_context_text_injection", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("allows text injection"));
    }

    #[test]
    fn rejects_external_embedding_allowed_gate() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["adapter_gates"]["external_embedding_model_allowed"] = Value::Bool(true);
        let path = write_temp_jsonl("active_context_external_embedding", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("external embedding model"));
    }

    #[test]
    fn rejects_shadow_steer_without_same_model_geometry_requirement() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["adapter_gates"]["same_model_route_geometry_required_for_shadow_steer"] =
            Value::Bool(false);
        let path = write_temp_jsonl("active_context_same_model_gate", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("same-model route geometry"));
    }

    #[test]
    fn rejects_shadow_steer_without_route_family_compatibility_requirement() {
        let mut value: Value =
            serde_json::from_str(&valid_decision_json("noop_observe", "[]")).unwrap();
        value["adapter_gates"]["route_family_compatible_required_for_shadow_steer"] =
            Value::Bool(false);
        let path = write_temp_jsonl("active_context_route_family_gate", &value.to_string());
        let err = load_runtime_adapter_decisions(&path).unwrap_err();
        let _ = fs::remove_file(path);
        assert!(format!("{err:#}").contains("route-family compatibility"));
    }
}
