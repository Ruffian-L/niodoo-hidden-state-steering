#![recursion_limit = "512"]

use anyhow::Error;
use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use clap::{Parser, ValueEnum};
use memmap2::MmapOptions;
mod cli;
mod main_helpers;
mod main_helpers2;
mod main_helpers3;
mod physics;
mod principia;
mod rainbow_test;
mod runtime;
mod simulation;
mod visualizer;
use crate::cli::*;
use crate::main_helpers::*;
use crate::main_helpers2::*;
use crate::main_helpers3::*;
use crate::principia::*;
use crate::simulation::run_simulation;

// Bridge module - feature-gated via niodv4_bridge feature
#[cfg(feature = "niodv4_bridge")]
mod bridge;

#[cfg(feature = "niodv4_bridge")]
use crate::bridge::startup_log::log_bridge_startup;
use crate::physics::naked_llama::{ModelKvCacheSnapshot, PhysicsEngine, QuantizedNakedLlama};
use crate::physics::optimizer::PhysicsParams;
use crate::physics::qwen35_hybrid::{
    summarize_qwen35_metadata, QuantizedQwen35Hybrid, Qwen35GgufMetadata,
};
use crate::physics::sensors::Sensor;
use crate::physics::vae::ManifoldVAE;
use crate::physics::websocket::{start_physics_server, PhysicsUpdate};
use crate::runtime::activation::{
    choose_correction_packet_arbitration, clamp_f32, clamp_usize, normalize,
    parse_correction_packet_arbitration_mode, parse_correction_packet_prompt_top_k_map,
    parse_step_window, pressure_activation_gate, resolve_correction_packet_prompt_top_k_match,
    resolve_correction_packet_prompt_top_k_override, should_suppress_correction_packets_for_prompt,
    smoothstep01, tensor_to_vec_f32, vec_to_tensor_f32, visible_request_activation_gate,
    CorrectionPacketArbitrationChoice, CorrectionPacketArbitrationDecision,
    CorrectionPacketArbitrationInput, CorrectionPacketArbitrationMode, GRAVITY_WELL, NIODOO_WOBBLE,
    ORBIT_SPEED, ORBIT_TOP_K,
};
use crate::runtime::active_context::{
    load_runtime_adapter_decisions, observe_only_packet_refs_from_metadata_value,
    observe_only_runtime_decision_summary, observe_only_shadow_steering_readiness,
    observe_only_turn_aggregate, runtime_metadata_diagnostic, runtime_startup_summary,
    runtime_startup_telemetry_record, runtime_turn_start_state,
    summarize_runtime_adapter_decisions, ActiveContextRuntimeStartupTelemetryRecord,
    ActiveContextRuntimeTurnStartState, ActiveContextShadowSteeringReadiness,
    ACTIVE_CONTEXT_ADAPTER_ID,
};
use crate::runtime::control_surface::{
    detect_request, visible_control_surface_active, RequestType,
};
use crate::runtime::finalization::{
    AnswerBoundaryFinalizer, FinalizationController, LockStopPolicy,
};
use crate::runtime::gmms_observe_dump::{
    gmms_observe_turn_start_event_record_checked, gmms_observe_turn_start_event_safety_violations,
    gmms_observe_turn_start_payload, gmms_observe_turn_start_payload_checked,
    maybe_run_gmms_observe_dump, maybe_run_gmms_observe_turn_start_event_dump,
};
use crate::runtime::metric_print::{print_metric_postmortem, print_metric_summary_line};
use crate::runtime::mistake_memory::{MistakeMemory, MistakeMemoryGuard};
use crate::runtime::mistake_reflex::{
    GmmsObserveOnlySummary, MistakeReflexGuard, MistakeReflexMemory,
};
use crate::runtime::secret_sauce_codec::{
    build_state_packet_secret_sauce, compress_hidden_state_to_64d, compress_runtime_hidden_64d,
    compress_slice_to_dim, compress_tensor_to_dim, cosine_similarity_f32, decode_secret_sauce,
    encode_secret_sauce_v1, encode_secret_sauce_v2, encode_secret_sauce_v3, splitmix64,
    SecretSauceDecoded, SecretSauceInputVersion, SecretSauceSegments, SecretSauceVersion,
    SECRET_SAUCE_RESTORE_DECAY_STEPS, SECRET_SAUCE_RESTORE_HIDDEN_WEIGHT,
    SECRET_SAUCE_RESTORE_MOMENTUM_WEIGHT, SECRET_SAUCE_RESTORE_SENTENCE_ALIGNMENT_FLOOR,
    SECRET_SAUCE_RESTORE_SENTENCE_WEIGHT,
};
use crate::runtime::state_types::{
    ControllerCandidateRecord, Gate34CandidateRecord, Gate34Phase, HingeCorrelationSummary,
    HingeWindowArtifact, HingeWindowCandidateSummary, HingeWindowTickRecord, MotifHingeSummary,
    MotifRoutingSummary, RoutingDecisionCache, TaskAnchorSummary,
};
use crate::runtime::telemetry::{ForceEngineStatus, TelemetryProfile, TokenPhysics};
use crate::runtime::timing::{
    elapsed_ms, now as timing_now, run_timing_path, write_run_timing, RunTiming,
};
use crate::visualizer::RenderParticle; // Stubbed struct
use candle_nn::ops::{sigmoid, softmax};
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safetensors::SafeTensors;
use serde_json;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

// =============================================================================
// NIODOO v1.0 GOLD MASTER CONFIGURATION
// Validated: Dec 16, 2025 (Seed 123 Clean / Seed 42 Creative)
// DO NOT MODIFY without full regression testing.
// =============================================================================

// 1. THE FORCE FIELDS
pub const NIODOO_PHYSICS_BLEND: f32 = 0.55; // The "Soul" Strength
pub const NIODOO_GHOST_GRAVITY: f32 = 10.0; // The "Topic" Anchor
pub const NIODOO_REPULSION: f32 = -0.60; // The "Anti-Boring" Field
/// stamp target ids as lowercase route ids such as `tn_count_e`.
fn main() -> Result<()> {
    // Default to GPU 0, but never override a device selection the user already
    // made (multi-GPU hosts set CUDA_VISIBLE_DEVICES themselves). Harmless no-op
    // on CPU-only machines.
    if std::env::var_os("CUDA_VISIBLE_DEVICES").is_none() {
        std::env::set_var("CUDA_VISIBLE_DEVICES", "0");
    }
    let mut args = Args::parse();
    args.gate34_target_source = args.gate34_target_source.to_ascii_lowercase();
    if args.gate34_target_source != "motifs"
        && args.gate34_target_source != "basins"
        && args.gate34_target_source != "specialists"
    {
        anyhow::bail!("--gate34-target-source must be one of: motifs, basins, specialists");
    }

    let active_bridge_modes = [
        args.bridge_influence_smoke,
        args.bridge_influence_selective,
        args.bridge_gate34_latch,
    ]
    .into_iter()
    .filter(|enabled| *enabled)
    .count();
    if active_bridge_modes > 1 {
        anyhow::bail!(
            "--bridge-influence-smoke, --bridge-influence-selective, and --bridge-gate34-latch are mutually exclusive"
        );
    }

    if maybe_run_gmms_observe_dump(&args)? {
        return Ok(());
    }
    if maybe_run_gmms_observe_turn_start_event_dump(&args)? {
        return Ok(());
    }

    let chat_stdout = args.stdout_profile == StdoutProfile::Chat;

    if args.active_context_startup_summary_out.is_some()
        && args.active_context_adapter_decisions.is_none()
    {
        anyhow::bail!(
            "--active-context-startup-summary-out requires --active-context-adapter-decisions"
        );
    }
    if args.active_context_startup_telemetry && args.active_context_adapter_decisions.is_none() {
        anyhow::bail!(
            "--active-context-startup-telemetry requires --active-context-adapter-decisions"
        );
    }

    if let Some(path) = &args.active_context_adapter_decisions {
        let decisions = load_runtime_adapter_decisions(path)?;
        let summary = summarize_runtime_adapter_decisions(&decisions);
        let summary_json = serde_json::to_string(&summary)?;
        let diagnostic = runtime_metadata_diagnostic(&summary);
        let diagnostic_json = serde_json::to_string(&diagnostic)?;
        if let Some(out) = &args.active_context_startup_summary_out {
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    fs::create_dir_all(parent).with_context(|| {
                        format!(
                            "Failed to create Active Context startup summary dir {}",
                            parent.display()
                        )
                    })?;
                }
            }
            let startup = runtime_startup_summary(diagnostic.clone());
            let startup_json = serde_json::to_string_pretty(&startup)? + "\n";
            fs::write(out, startup_json).with_context(|| {
                format!(
                    "Failed to write Active Context startup summary {}",
                    out.display()
                )
            })?;
        }
        if chat_stdout {
            eprintln!(" [ACTIVE_CONTEXT] loaded_adapter_decisions={summary_json}");
            eprintln!(" [ACTIVE_CONTEXT] runtime_metadata_counters={diagnostic_json}");
        } else {
            println!(" [ACTIVE_CONTEXT] loaded_adapter_decisions={summary_json}");
            println!(" [ACTIVE_CONTEXT] runtime_metadata_counters={diagnostic_json}");
        }
    }

    // Bridge startup log - reports exact state of bridge artifacts
    #[cfg(feature = "niodv4_bridge")]
    if !chat_stdout {
        log_bridge_startup();
    }

    let scaling_profile = apply_model_auto_scaling(&mut args);
    if args.print_scaling_profile_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&build_scaling_profile_value(&args, &scaling_profile))?
        );
        return Ok(());
    }

    let ui_events_json = args.ui_events_json;
    let scaling_profile_value = build_scaling_profile_value(&args, &scaling_profile);
    let rt = tokio::runtime::Runtime::new()?;

    if chat_stdout {
        eprintln!("Starting Simulation...");
    } else {
        println!("Starting Simulation...");
    }
    if let Some(profile) = &scaling_profile {
        if chat_stdout {
            eprintln!(
                " [MODEL_SCALE] size={}B archetype={:?} scale={:.3} sigma={:.3} theta={:.2} beta={:.1} repulsion={:.2} temp={:.3} motif={:.3} recovery={:.3} guard={:.2} focus_lock={}",
                profile.params_billions,
                profile.archetype,
                profile.scale,
                profile.sigma,
                profile.theta,
                profile.beta,
                profile.loop_repulsion,
                profile.temperature,
                profile.motif_force_scale,
                profile.recovery_force_scale,
                profile.guardrail_bias_scale,
                profile.focus_lock_ticks
            );
            eprintln!(
                " [MODEL_SCALE] applied sigma={:.3} physics_blend={:.2} repulsion_strength={:.2} temperature={:.3}",
                args.sigma,
                args.physics_blend,
                args.repulsion_strength,
                args.temperature
            );
        } else {
            println!(
                " [MODEL_SCALE] size={}B archetype={:?} scale={:.3} sigma={:.3} theta={:.2} beta={:.1} repulsion={:.2} temp={:.3} motif={:.3} recovery={:.3} guard={:.2} focus_lock={}",
                profile.params_billions,
                profile.archetype,
                profile.scale,
                profile.sigma,
                profile.theta,
                profile.beta,
                profile.loop_repulsion,
                profile.temperature,
                profile.motif_force_scale,
                profile.recovery_force_scale,
                profile.guardrail_bias_scale,
                profile.focus_lock_ticks
            );
            println!(
                " [MODEL_SCALE] applied sigma={:.3} physics_blend={:.2} repulsion_strength={:.2} temperature={:.3}",
                args.sigma,
                args.physics_blend,
                args.repulsion_strength,
                args.temperature
            );
        }
    } else if args.model_auto_scale {
        if chat_stdout {
            eprintln!(
                " [MODEL_SCALE] skipped: could not parse model_size='{}'; using manual parameters",
                args.model_size
            );
        } else {
            println!(
                " [MODEL_SCALE] skipped: could not parse model_size='{}'; using manual parameters",
                args.model_size
            );
        }
    }
    emit_ui_event_value(ui_events_json, "model_scale", scaling_profile_value);

    let run_result = rt.block_on(async {
        if args.rainbow_test {
            rainbow_test::run_rainbow_test(args).await
        } else {
            run_simulation(None, args, scaling_profile.clone()).await
        }
    });
    if let Err(e) = run_result {
        emit_ui_event_value(
            ui_events_json,
            "fatal_error",
            serde_json::json!({
                "message": format!("{:?}", e),
            }),
        );
        if scaling_profile.is_some() {
            eprintln!("Simulation Error: {:?}", e);
        } else {
            eprintln!("Runtime Error: {:?}", e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
