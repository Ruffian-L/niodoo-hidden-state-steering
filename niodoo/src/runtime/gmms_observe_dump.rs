//! GMMS observe-only turn-start dump: payload assembly + safety scanners.
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).

use anyhow::{Context, Error, Result};
use std::fs;

use crate::cli::Args;
use crate::runtime::mistake_reflex::{GmmsObserveOnlySummary, MistakeReflexMemory};

pub(crate) fn maybe_run_gmms_observe_dump(args: &Args) -> Result<bool> {
    let Some(prompt) = args.gmms_observe_dump_prompt.as_ref() else {
        return Ok(false);
    };
    let path = args
        .mistake_reflex_path
        .as_ref()
        .context("--gmms-observe-dump-prompt requires --mistake-reflex-path")?;
    let memory = MistakeReflexMemory::load(path)?;
    let summaries = memory.observe_gmms_applicability(prompt, args.gmms_observe_dump_limit);
    let selected = summaries.first();
    let value = serde_json::json!({
        "dump_type": "gmms_observe_only_applicability",
        "observe_only": true,
        "runtime_matcher_activation_claimed": false,
        "mistake_reflex_query_called": false,
        "final_answer_text_included": false,
        "mistake_reflex_path": path.display().to_string(),
        "prompt": prompt,
        "limit": args.gmms_observe_dump_limit,
        "summary_count": summaries.len(),
        "selected_slice_id": selected.map(|item| item.event_id.as_str()),
        "mode": selected.map(|item| item.mode.as_str()),
        "allowed_action_max": selected.map(|item| item.allowed_action_max.as_str()),
        "action_level": selected.map(|item| item.action_level),
        "score": selected.map(|item| item.score),
        "route_unicode_sidecar_attached": selected.map(|item| item.route_unicode_sidecar_attached),
        "unicode_packet_id": selected.and_then(|item| item.unicode_packet_id.as_deref()),
        "summaries": summaries,
    });
    let text = serde_json::to_string_pretty(&value)? + "\n";
    if let Some(out) = &args.gmms_observe_dump_out {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create GMMS dump dir {}", parent.display())
                })?;
            }
        }
        fs::write(out, text)
            .with_context(|| format!("Failed to write GMMS observe dump {}", out.display()))?;
    } else {
        print!("{text}");
    }
    Ok(true)
}

pub(crate) fn gmms_observe_turn_start_payload(
    turn_index: usize,
    memory_path_configured: bool,
    limit: usize,
    summaries: &[GmmsObserveOnlySummary],
) -> serde_json::Value {
    let selected = summaries.first();
    serde_json::json!({
        "turn_index": turn_index,
        "phase": "turn_start",
        "observe_only": true,
        "runtime_matcher_activation_claimed": false,
        "mistake_reflex_query_called_for_gmms_observe": false,
        "prompt_injection_applied": false,
        "final_answer_text_included": false,
        "memory_path_configured": memory_path_configured,
        "limit": limit,
        "summary_count": summaries.len(),
        "selected_slice_id": selected.map(|item| item.event_id.as_str()),
        "mode": selected.map(|item| item.mode.as_str()),
        "allowed_action_max": selected.map(|item| item.allowed_action_max.as_str()),
        "action_level": selected.map(|item| item.action_level),
        "score": selected.map(|item| item.score),
        "route_unicode_sidecar_attached": selected.map(|item| item.route_unicode_sidecar_attached),
        "unicode_packet_id": selected.and_then(|item| item.unicode_packet_id.as_deref()),
        "summaries": summaries,
    })
}

fn gmms_observe_turn_start_forbidden_key_paths(value: &serde_json::Value) -> Vec<String> {
    const FORBIDDEN_KEYS: &[&str] = &[
        "raw_episode",
        "assistant_failure_span",
        "user_correction",
        "corrected_delta",
        "accepted_surfaces",
    ];

    fn visit(value: &serde_json::Value, path: &str, out: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    let child_path = if path.is_empty() {
                        key.to_string()
                    } else {
                        format!("{path}.{key}")
                    };
                    if FORBIDDEN_KEYS.contains(&key.as_str()) {
                        out.push(child_path.clone());
                    }
                    visit(child, &child_path, out);
                }
            }
            serde_json::Value::Array(items) => {
                for (idx, child) in items.iter().enumerate() {
                    let child_path = format!("{path}[{idx}]");
                    visit(child, &child_path, out);
                }
            }
            _ => {}
        }
    }

    let mut paths = Vec::new();
    visit(value, "", &mut paths);
    paths
}

fn gmms_observe_turn_start_final_answer_value_paths(value: &serde_json::Value) -> Vec<String> {
    const FORBIDDEN_VALUE_PATTERNS: &[&str] = &[
        "final answer:",
        "correct answer:",
        "accepted answer:",
        "correct_answer",
        "start_visible_answer",
    ];

    fn visit(value: &serde_json::Value, path: &str, out: &mut Vec<String>) {
        match value {
            serde_json::Value::String(text) => {
                let normalized = text.to_ascii_lowercase();
                if FORBIDDEN_VALUE_PATTERNS
                    .iter()
                    .any(|pattern| normalized.contains(pattern))
                {
                    out.push(path.to_string());
                }
            }
            serde_json::Value::Object(map) => {
                for (key, child) in map {
                    let child_path = if path.is_empty() {
                        key.to_string()
                    } else {
                        format!("{path}.{key}")
                    };
                    visit(child, &child_path, out);
                }
            }
            serde_json::Value::Array(items) => {
                for (idx, child) in items.iter().enumerate() {
                    let child_path = format!("{path}[{idx}]");
                    visit(child, &child_path, out);
                }
            }
            _ => {}
        }
    }

    let mut paths = Vec::new();
    visit(value, "", &mut paths);
    paths
}

pub(crate) fn gmms_observe_turn_start_event_safety_violations(
    record: &serde_json::Value,
) -> Vec<String> {
    let mut violations = Vec::new();
    if record.get("event").and_then(serde_json::Value::as_str)
        != Some("gmms_observe_only_applicability")
    {
        violations.push("unexpected_event_name".to_string());
    }

    let Some(payload) = record.get("payload") else {
        violations.push("payload_missing".to_string());
        return violations;
    };
    if !payload.is_object() {
        violations.push("payload_not_object".to_string());
        return violations;
    }

    for field in ["observe_only", "memory_path_configured"] {
        if !payload
            .get(field)
            .and_then(serde_json::Value::as_bool)
            .is_some()
        {
            violations.push(format!("{field}_missing_or_not_bool"));
        }
    }
    for field in [
        "runtime_matcher_activation_claimed",
        "mistake_reflex_query_called_for_gmms_observe",
        "prompt_injection_applied",
        "final_answer_text_included",
    ] {
        if payload.get(field).and_then(serde_json::Value::as_bool) != Some(false) {
            violations.push(format!("{field}_not_false"));
        }
    }
    if payload
        .get("observe_only")
        .and_then(serde_json::Value::as_bool)
        != Some(true)
    {
        violations.push("observe_only_not_true".to_string());
    }

    let forbidden_keys = gmms_observe_turn_start_forbidden_key_paths(record);
    if !forbidden_keys.is_empty() {
        violations.push(format!(
            "forbidden_payload_keys:{}",
            forbidden_keys.join(",")
        ));
    }
    let final_answer_values = gmms_observe_turn_start_final_answer_value_paths(record);
    if !final_answer_values.is_empty() {
        violations.push(format!(
            "final_answer_like_values:{}",
            final_answer_values.join(",")
        ));
    }

    violations
}

fn ensure_gmms_observe_turn_start_event_safe(record: &serde_json::Value) -> Result<()> {
    let violations = gmms_observe_turn_start_event_safety_violations(record);
    if !violations.is_empty() {
        anyhow::bail!(
            "unsafe GMMS turn-start observe event record: {}",
            violations.join("; ")
        );
    }
    Ok(())
}

pub(crate) fn gmms_observe_turn_start_payload_checked(
    turn_index: usize,
    memory_path_configured: bool,
    limit: usize,
    summaries: &[GmmsObserveOnlySummary],
) -> Result<serde_json::Value> {
    let payload =
        gmms_observe_turn_start_payload(turn_index, memory_path_configured, limit, summaries);
    let record = serde_json::json!({
        "event": "gmms_observe_only_applicability",
        "payload": payload,
    });
    ensure_gmms_observe_turn_start_event_safe(&record)?;
    Ok(record["payload"].clone())
}

pub(crate) fn gmms_observe_turn_start_event_record(
    turn_index: usize,
    memory_path_configured: bool,
    limit: usize,
    summaries: &[GmmsObserveOnlySummary],
) -> serde_json::Value {
    serde_json::json!({
        "event": "gmms_observe_only_applicability",
        "payload": gmms_observe_turn_start_payload(
            turn_index,
            memory_path_configured,
            limit,
            summaries,
        ),
    })
}

pub(crate) fn gmms_observe_turn_start_event_record_checked(
    turn_index: usize,
    memory_path_configured: bool,
    limit: usize,
    summaries: &[GmmsObserveOnlySummary],
) -> Result<serde_json::Value> {
    let record =
        gmms_observe_turn_start_event_record(turn_index, memory_path_configured, limit, summaries);
    ensure_gmms_observe_turn_start_event_safe(&record)?;
    Ok(record)
}

pub(crate) fn gmms_observe_turn_start_rejected_event_record(
    turn_index: usize,
    error: &Error,
) -> serde_json::Value {
    serde_json::json!({
        "event": "gmms_observe_only_applicability_rejected",
        "payload": {
            "turn_index": turn_index,
            "phase": "turn_start",
            "observe_only": true,
            "consumer_action": "reject_record",
            "error": error.to_string(),
            "runtime_matcher_activation_claimed": false,
            "mistake_reflex_query_called_for_gmms_observe": false,
            "prompt_injection_applied": false,
            "final_answer_text_included": false,
        },
    })
}

pub(crate) fn gmms_observe_turn_start_unsafe_fixture_record(
    turn_index: usize,
    memory_path_configured: bool,
    limit: usize,
    summaries: &[GmmsObserveOnlySummary],
) -> serde_json::Value {
    let mut record =
        gmms_observe_turn_start_event_record(turn_index, memory_path_configured, limit, summaries);
    if let Some(payload) = record
        .get_mut("payload")
        .and_then(serde_json::Value::as_object_mut)
    {
        payload.insert(
            "prompt_injection_applied".to_string(),
            serde_json::Value::Bool(true),
        );
        if let Some(summary) = payload
            .get_mut("summaries")
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|items| items.first_mut())
            .and_then(serde_json::Value::as_object_mut)
        {
            summary.insert(
                "user_correction".to_string(),
                serde_json::Value::String(
                    "negative fixture raw correction must be rejected".to_string(),
                ),
            );
        }
    }
    record
}

pub(crate) fn maybe_run_gmms_observe_turn_start_event_dump(args: &Args) -> Result<bool> {
    let Some(prompt) = args.gmms_observe_turn_start_event_dump_prompt.as_ref() else {
        return Ok(false);
    };
    let path = args
        .mistake_reflex_path
        .as_ref()
        .context("--gmms-observe-turn-start-event-dump-prompt requires --mistake-reflex-path")?;
    let memory = MistakeReflexMemory::load(path)?;
    let summaries = memory.observe_gmms_applicability(prompt, args.gmms_observe_turn_start_limit);
    let value = if args.gmms_observe_turn_start_event_dump_unsafe_fixture {
        let unsafe_record = gmms_observe_turn_start_unsafe_fixture_record(
            0,
            args.mistake_reflex_path.is_some(),
            args.gmms_observe_turn_start_limit,
            &summaries,
        );
        match ensure_gmms_observe_turn_start_event_safe(&unsafe_record) {
            Ok(()) => anyhow::bail!(
                "GMMS unsafe turn-start fixture unexpectedly passed safety validation"
            ),
            Err(error) => gmms_observe_turn_start_rejected_event_record(0, &error),
        }
    } else {
        gmms_observe_turn_start_event_record_checked(
            0,
            args.mistake_reflex_path.is_some(),
            args.gmms_observe_turn_start_limit,
            &summaries,
        )?
    };
    let text = serde_json::to_string_pretty(&value)? + "\n";
    if let Some(out) = &args.gmms_observe_turn_start_event_dump_out {
        if let Some(parent) = out.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create GMMS event dump dir {}", parent.display())
                })?;
            }
        }
        fs::write(out, text).with_context(|| {
            format!(
                "Failed to write GMMS turn-start event dump {}",
                out.display()
            )
        })?;
    } else {
        print!("{text}");
    }
    Ok(true)
}
