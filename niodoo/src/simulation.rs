//! `run_simulation`: the main async decode-loop entry point.
//! Extracted from main.rs as part of the comprehensive refactor
//! (pre-refactor-main-split-20260508 backup).
//!
//! This module pulls a wildcard `use crate::*;` because the simulation body
//! references the majority of crate-root types/fns. Subsequent refactors
//! should narrow the import surface.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::ops::{sigmoid, softmax};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use tokenizers::Tokenizer;

use crate::cli::*;
use crate::physics::naked_llama::{ModelKvCacheSnapshot, PhysicsEngine, QuantizedNakedLlama};
use crate::physics::optimizer::PhysicsParams;
use crate::physics::qwen35_hybrid::{
    summarize_qwen35_metadata, QuantizedQwen35Hybrid, Qwen35GgufMetadata,
};
use crate::principia::*;
use crate::rainbow_test;
use crate::runtime::activation::*;
use crate::runtime::active_context::*;
use crate::runtime::control_surface::{
    detect_request, visible_control_surface_active, RequestType,
};
use crate::runtime::finalization::{
    detect_answer_boundary_expectation, AnswerBoundaryFinalizer, AnswerBoundaryKind,
    FinalizationController, LockStopPolicy,
};
use crate::runtime::gmms_observe_dump::*;
use crate::runtime::metric_print::*;
use crate::runtime::mistake_memory::{MistakeMemory, MistakeMemoryGuard};
use crate::runtime::mistake_reflex::{
    GmmsObserveOnlySummary, MistakeReflexGuard, MistakeReflexMemory,
};
use crate::runtime::secret_sauce_codec::*;
use crate::runtime::state_types::*;
use crate::runtime::telemetry::{ForceEngineStatus, TelemetryProfile, TokenPhysics};
use crate::runtime::timing::*;
use crate::visualizer::RenderParticle;
use crate::*;

pub(crate) async fn run_simulation(
    _vis_tx: Option<Sender<Vec<RenderParticle>>>,
    args: Args,
    scaling_profile: Option<ModelScalingProfile>,
) -> Result<()> {
    let chat_stdout = args.stdout_profile == StdoutProfile::Chat;
    // Select Device
    let mut device = match Device::new_cuda(0) {
        Ok(cuda_device) => cuda_device,
        Err(err) => {
            if args.require_cuda {
                anyhow::bail!("[DEVICE] cuda_init_failed={} (require_cuda=true)", err);
            }
            if chat_stdout {
                eprintln!("[DEVICE] cuda_init_failed={} ; falling back to CPU", err);
            } else {
                println!("[DEVICE] cuda_init_failed={} ; falling back to CPU", err);
            }
            Device::Cpu
        }
    };
    if chat_stdout {
        eprintln!("DEVICE: {:?}", device);
    } else {
        println!("DEVICE: {:?}", device);
    }
    if format!("{:?}", device).starts_with("Cuda(") {
        if chat_stdout {
            eprintln!("[GPU] cuda_device_selected=true");
        } else {
            println!("[GPU] cuda_device_selected=true");
        }
    }
    if chat_stdout {
        eprintln!("MODE: Naked Llama (Physics Attention) - EVOLVED V4.1");
    } else {
        println!("MODE: Naked Llama (Physics Attention) - EVOLVED V4.1");
    }
    let mut run_timing = RunTiming::start(format!("{:?}", device), args.max_steps);

    #[cfg(feature = "niodv4_bridge")]
    {
        if let Some(codec_path) = args.rave_codec_path.as_ref() {
            if codec_path.exists() {
                match crate::bridge::rave_codec::RaveCodec::load(codec_path, &device) {
                    Ok(codec) => {
                        if install_rave_codec(codec) {
                            println!(
                                " [RAVE_CODEC] Loaded trained hidden-state codec from {}",
                                codec_path.display()
                            );
                        } else {
                            println!(" [RAVE_CODEC] codec already installed; skipping reload");
                        }
                    }
                    Err(err) => {
                        println!(
                            " [RAVE_CODEC] Failed to load codec from {}: {err}",
                            codec_path.display()
                        );
                    }
                }
            } else {
                println!(
                    " [RAVE_CODEC] Path {} does not exist; falling back to bucket-expansion projection",
                    codec_path.display()
                );
            }
        }
    }

    let runtime_bridge_manifest = match load_runtime_bridge(&args.runtime_bridge_path)? {
        Some((path, manifest)) => {
            if chat_stdout {
                eprintln!(" [BRIDGE] Loaded runtime bridge: {}", path.display());
            } else {
                print_runtime_bridge_summary(&path, &manifest);
            }
            Some(manifest)
        }
        None => {
            if chat_stdout {
                eprintln!(
                    " [BRIDGE] No runtime bridge found from '{}'; continuing without motif-memory bootstrap",
                    args.runtime_bridge_path
                );
            } else {
                println!(
                    " [BRIDGE] No runtime bridge found from '{}'; continuing without motif-memory bootstrap",
                    args.runtime_bridge_path
                );
            }
            None
        }
    };

    let (tx_ctrl, _control_rx) = mpsc::channel(100);
    let broadcast_tx = if args.physics_ws {
        if chat_stdout {
            eprintln!(
                " [WS] Starting legacy physics websocket broadcast on 0.0.0.0:{}",
                args.physics_ws_port
            );
        } else {
            println!(
                " [WS] Starting legacy physics websocket broadcast on 0.0.0.0:{}",
                args.physics_ws_port
            );
        }
        Some(start_physics_server(args.physics_ws_port, tx_ctrl).await)
    } else {
        None
    };

    if chat_stdout {
        eprintln!(" [MEMORY] Loading Model...");
    } else {
        println!(" [MEMORY] Loading Model...");
    }
    let model_load_started = timing_now();
    let mut model = match load_model(&args, &device) {
        Ok(model) => model,
        Err(err) if format!("{device:?}").starts_with("Cuda(") => {
            if args.require_cuda {
                anyhow::bail!(
                    "[DEVICE] cuda_model_load_failed={} (require_cuda=true)",
                    err
                );
            }
            if chat_stdout {
                eprintln!(
                    " [DEVICE_FALLBACK] cuda_model_load_failed={} ; retrying on CPU",
                    err
                );
            } else {
                println!(
                    " [DEVICE_FALLBACK] cuda_model_load_failed={} ; retrying on CPU",
                    err
                );
            }
            device = Device::Cpu;
            if chat_stdout {
                eprintln!("DEVICE: {:?}", device);
            } else {
                println!("DEVICE: {:?}", device);
            }
            load_model(&args, &device)?
        }
        Err(err) => return Err(err),
    };
    run_timing.add_model_load_ms(elapsed_ms(model_load_started));
    run_timing.set_gpu_name(format!("{:?}", device));
    if args.metadata_only {
        println!(" [QWEN35] metadata_only=true; exiting before universe/bootstrap/generation");
        return Ok(());
    }
    if args.tokenizer_smoke {
        run_tokenizer_smoke(&args, &model)?;
        println!(" [QWEN35] tokenizer_smoke=true; exiting before universe/bootstrap/generation");
        return Ok(());
    }
    let control_token_ids = build_control_token_shield(model.tokenizer(), args.runtime_mode);
    let mistake_reflex_retry_token_ids = build_parallel_duration_retry_shield(model.tokenizer());
    let hidden_request_profiles = build_request_surface_profiles(model.tokenizer());
    let answer_logit_probe_targets = if args.answer_logit_probe_out.is_some() {
        build_answer_logit_probe_targets(model.tokenizer(), &args.answer_logit_probe_surfaces)?
    } else {
        Vec::new()
    };
    let mut answer_logit_probe_records: Vec<serde_json::Value> = Vec::new();
    if args.answer_logit_probe_out.is_some() && !chat_stdout {
        println!(
            " [ANSWER_LOGIT_PROBE] targets={} surfaces=\"{}\"",
            answer_logit_probe_targets.len(),
            args.answer_logit_probe_surfaces
        );
    }
    if !chat_stdout {
        println!(
            " [MODE] runtime_mode={:?} control_token_shield={}",
            args.runtime_mode,
            if args.runtime_mode.uses_control_shield() {
                control_token_ids.len().to_string()
            } else {
                "disabled".to_string()
            }
        );
    }
    if args.runtime_mode.uses_control_shield() && !chat_stdout {
        print_control_token_shield_summary(
            model.tokenizer(),
            &control_token_ids,
            args.runtime_mode,
        );
    }
    if args.hidden_request_inference && args.runtime_mode.teaches_control_language() && !chat_stdout
    {
        println!(
            " [MODE] hidden_request_profiles={}",
            hidden_request_profiles.len()
        );
    }
    // If GGUF, hidden_size might be different. We rely on Universe adaptation mostly.

    if chat_stdout {
        eprintln!(" [MEMORY] Loading Universe...");
    } else {
        println!(" [MEMORY] Loading Universe...");
    }
    let universe = load_universe_bootstrap(&args, &model, &device)?;
    let hidden_dim = model.hidden_dim();
    let full_vocab_size = universe.full_vocab_size;
    let emb_dim = universe.emb_dim;
    let limit_n = universe.limit_n;
    let charge_tensor = universe.charge_tensor;
    let particle_words = universe.particle_words;
    let runtime_motifs = if let Some(manifest) = &runtime_bridge_manifest {
        build_runtime_motif_bank(manifest, hidden_dim, &device)?
    } else {
        Vec::new()
    };
    let runtime_recovery_ops = if let Some(manifest) = &runtime_bridge_manifest {
        build_runtime_recovery_bank(manifest, hidden_dim, &device)?
    } else {
        Vec::new()
    };
    let specialist_memory_workers = match args.specialist_memory_workers_mode {
        SpecialistMemoryWorkerMode::Off => Vec::new(),
        SpecialistMemoryWorkerMode::Shadow | SpecialistMemoryWorkerMode::Influence => {
            let path = args.specialist_memory_workers_path.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "--specialist-memory-workers-mode shadow|influence requires --specialist-memory-workers-path"
                )
            })?;
            let workers = load_specialist_memory_workers(path, hidden_dim, &device)?;
            if chat_stdout {
                eprintln!(
                    " [SPECIALIST_WORKERS] Loaded {} {} packets from {}",
                    workers.len(),
                    args.specialist_memory_workers_mode.as_str(),
                    path.display()
                );
            } else {
                println!(
                    " [SPECIALIST_WORKERS] Loaded {} {} packets from {}",
                    workers.len(),
                    args.specialist_memory_workers_mode.as_str(),
                    path.display()
                );
            }
            workers
        }
    };
    if let Some(fixed_packet_id) = args.specialist_memory_worker_fixed_packet_id.as_deref() {
        if args.specialist_memory_workers_mode == SpecialistMemoryWorkerMode::Off {
            anyhow::bail!(
                "--specialist-memory-worker-fixed-packet-id requires --specialist-memory-workers-mode shadow|influence"
            );
        }
        if !specialist_memory_workers
            .iter()
            .any(|worker| worker.packet_id == fixed_packet_id)
        {
            anyhow::bail!(
                "--specialist-memory-worker-fixed-packet-id={} did not match any loaded worker packet",
                fixed_packet_id
            );
        }
    }
    let continuity_scale = scaling_profile
        .as_ref()
        .map(|profile| continuity_scale_tuning(profile.params_billions))
        .or_else(|| parse_params_billions(&args.model_size).map(continuity_scale_tuning))
        .unwrap_or(ContinuityScaleTuning {
            support_scale: 1.0,
            release_scale: 1.0,
        });

    if chat_stdout {
        eprintln!(
            " [UNIVERSE] Loaded: source={} tensor_rows={} usable_rows={} dim={} token_map={} (Model hidden={})",
            universe.source_description,
            full_vocab_size,
            limit_n,
            emb_dim,
            universe.token_map_description,
            hidden_dim
        );
    } else {
        println!(
            " [UNIVERSE] Loaded: source={} tensor_rows={} usable_rows={} dim={} token_map={} (Model hidden={})",
            universe.source_description,
            full_vocab_size,
            limit_n,
            emb_dim,
            universe.token_map_description,
            hidden_dim
        );
    }
    let initial_motif_provenance = summarize_runtime_motif_provenance(&runtime_motifs);
    emit_ui_event_value(
        args.ui_events_json,
        "runtime_bootstrap",
        serde_json::json!({
            "embedding_source": universe.source_description,
            "token_map_description": universe.token_map_description,
            "tensor_rows": full_vocab_size,
            "usable_rows": limit_n,
            "embedding_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "runtime_mode": args.runtime_mode.as_str(),
            "motif_provenance": initial_motif_provenance,
            "top_motifs": runtime_motif_briefs(&runtime_motifs, 3),
        }),
    );

    let mass_vec: Vec<f32> = (1..=limit_n)
        .map(|i| {
            let r = i as f32;
            (r.ln().max(0.1) / (limit_n as f32).ln()) * 5000.0
        })
        .collect();
    let mass_tensor = Tensor::from_vec(mass_vec, (limit_n, 1), &device)?;

    let params = PhysicsParams::new(
        args.gravity as f64,
        args.dt as f64,
        args.repulsion_strength,
        0.1, // alpha_info
        0.1, // alpha_sem
        0.1, // alpha_coh
        0.5, // alpha_struct
        0.6, // alpha_quantum
        0.7, // alpha_geometric
        0.5, // alpha_emo
        true,
        0.01,
        args.mu,
        args.sigma,
        0.9,
        args.pinn_enabled,
        args.pinn_stiffness,
        args.ghost_gravity,
        args.gravity_well as f64, // Gravity Well Strength
        args.orbit_speed as f64,  // Orbit Speed
    );

    let mut heap = BinaryHeap::new();
    for _ in 0..10 {
        heap.push(EvoEntry {
            fitness: 0.1,
            params: params.clone(),
        });
    }

    let black_hole_embeddings_vec = {
        let mut black_holes = Vec::new();
        if !args.black_holes.is_empty() {
            let targets: Vec<&str> = args.black_holes.split(',').map(|s| s.trim()).collect();
            if chat_stdout {
                eprintln!(" [REPEL] Initializing Black Holes: {:?}", targets);
            } else {
                println!(" [REPEL] Initializing Black Holes: {:?}", targets);
            }
            for target in targets {
                if let Some(idx) = particle_words
                    .iter()
                    .position(|w| w.eq_ignore_ascii_case(target))
                {
                    if let Ok(emb) = charge_tensor.i(idx) {
                        if let Ok(emb_f32) = emb.to_dtype(DType::F32) {
                            // Project if needed
                            let projected = if emb_f32.dim(0).unwrap_or(0) != hidden_dim {
                                if let Some(proj) =
                                    &Tensor::randn(0.0f32, 0.02, (emb_dim, hidden_dim), &device)
                                        .ok()
                                {
                                    emb_f32.matmul(proj).ok()
                                } else {
                                    None
                                }
                            } else {
                                Some(emb_f32)
                            };

                            if let Some(p) = projected {
                                black_holes.push(p.detach());
                            }
                        }
                    }
                }
            }
        }
        black_holes
    };

    let kv_prefix_vocab_n = particle_words.len().max(1);
    let kv_prefix_token_ids: Vec<u32> = (0..kv_prefix_vocab_n as u32).collect();
    let kv_prefix_token_tensor =
        Tensor::from_vec(kv_prefix_token_ids, (kv_prefix_vocab_n,), &device)?;
    let kv_prefix_emb = model
        .embed_tokens_forward(&kv_prefix_token_tensor)?
        .to_dtype(DType::F32)?;
    let kv_prefix_charge_tensor =
        kv_prefix_emb.broadcast_div(&kv_prefix_emb.sqr()?.sum_keepdim(1)?.sqrt()?)?;

    let runtime_speed_profile = args.runtime_speed_profile;
    let stdout_profile =
        if runtime_speed_profile.is_eval_fast() && args.stdout_profile == StdoutProfile::Debug {
            StdoutProfile::Quiet
        } else {
            args.stdout_profile
        };
    let bridge_force_layer_policy = if runtime_speed_profile.is_eval_fast()
        && args.bridge_force_layer_policy == BridgeForceLayerPolicy::All
    {
        BridgeForceLayerPolicy::Single
    } else {
        args.bridge_force_layer_policy
    };
    let secret_sauce_capture_policy = if runtime_speed_profile.is_eval_fast()
        && args.secret_sauce_capture_policy == SecretSauceCapturePolicy::PerToken
    {
        SecretSauceCapturePolicy::Off
    } else {
        args.secret_sauce_capture_policy
    };

    let parsed_worker_scope = parse_specialist_memory_worker_influence_scope_arg(
        args.specialist_memory_worker_influence_scope.trim(),
    )?;
    let specialist_worker_influence_layers_tuple = match args
        .specialist_memory_worker_influence_layers
        .as_deref()
        .map(|s| s.trim())
    {
        Some(s) if !s.is_empty() => Some(parse_specialist_memory_worker_influence_layers_arg(s)?),
        _ => None,
    };
    #[cfg(feature = "niodv4_bridge")]
    let correction_packet_arbitration_mode =
        parse_correction_packet_arbitration_mode(&args.correction_packet_arbitration)?;
    #[cfg(not(feature = "niodv4_bridge"))]
    let correction_packet_arbitration_mode = CorrectionPacketArbitrationMode::Disabled;
    #[cfg(feature = "niodv4_bridge")]
    let correction_packet_arbitration_healthy_factor_threshold =
        args.correction_packet_arbitration_healthy_factor_threshold;
    #[cfg(not(feature = "niodv4_bridge"))]
    let correction_packet_arbitration_healthy_factor_threshold = 0.999;
    #[cfg(feature = "niodv4_bridge")]
    let correction_packet_arbitration_stale_distance_threshold =
        args.correction_packet_arbitration_stale_distance_threshold;
    #[cfg(not(feature = "niodv4_bridge"))]
    let correction_packet_arbitration_stale_distance_threshold = 0.0;

    let mut phys_engine = PrincipiaEngine {
        mass_tensor,
        charge_tensor: charge_tensor.clone(),
        kv_prefix_charge_tensor,
        particle_words,
        sensors: Vec::new(),
        vae: None,
        sigma: None,
        attractors: Vec::new(),
        vad_head: VADHead::new(hidden_dim, &device).ok(),
        sentence_history: VecDeque::new(),
        params: params.clone(),
        evo_population: heap,
        symbolic_solver: Some(SymbolicModule {}),
        pinn_loss: None,
        lpm_collaborator: Some(LPMInterface {}),
        geometric_dl: Some(GraphConv {}),
        deepmd_kit: Some(DeePMDKit {}),
        nvidia_physicsnemo: Some(PhysicsNeMo {}),
        current_step: 0,
        current_sentence_embeddings: Vec::new(),
        current_surprisals: Vec::new(),
        current_sentence_tokens: Vec::new(),
        start_logits: None,
        graviton_proj: Tensor::ones((8, hidden_dim), DType::F32, &device)
            .map(|t| {
                let scale = Tensor::new(0.001f32, t.device()).unwrap();
                t.broadcast_mul(&scale).unwrap()
            })
            .ok(),
        layer_norms: std::collections::HashMap::new(),
        last_deltas: std::collections::HashMap::new(),
        goal_embedding: None,
        momentum_buffer: None,
        secret_sauce_hidden_prior: None,
        secret_sauce_sentence_prior: None,
        secret_sauce_momentum_prior: None,
        secret_sauce_version: None,
        secret_sauce_decay_steps: SECRET_SAUCE_RESTORE_DECAY_STEPS,
        secret_sauce_steps_remaining: 0,
        hidden_dim,
        emb_dim,
        proj_matrix: if emb_dim != hidden_dim {
            Some(Tensor::randn(0.0f32, 0.02, (emb_dim, hidden_dim), &device)?)
        } else {
            None
        },
        physics_blend: args.physics_blend,
        physics_start_layer: args.physics_start_layer,
        physics_end_layer: args.physics_end_layer,
        multiplicative_blend: args.multiplicative_blend,
        runtime_mode: args.runtime_mode,
        hidden_request_inference: args.hidden_request_inference,
        ui_events_json: args.ui_events_json,
        runtime_speed_profile,
        stdout_profile,
        bridge_force_layer_policy,
        bridge_force_layer: args.bridge_force_layer,
        bridge_force_selection: args.bridge_force_selection,
        bridge_force_trajectory_schedule: args.bridge_force_trajectory_schedule,
        bridge_force_role_filter: args.bridge_force_role_filter,
        bridge_force_min_margin: args.bridge_force_min_margin,
        secret_sauce_capture_policy,
        specialist_memory_workers,
        specialist_memory_workers_mode: args.specialist_memory_workers_mode,
        specialist_memory_worker_top_k: args.specialist_memory_worker_top_k.max(1),
        specialist_memory_worker_influence_clamp: args
            .specialist_memory_worker_influence_clamp
            .clamp(0.0, 0.03),
        specialist_memory_worker_influence_sign: args
            .specialist_memory_worker_influence_sign
            .clamp(-1.0, 1.0),
        specialist_memory_worker_influence_scope: parsed_worker_scope,
        specialist_memory_worker_influence_direction: args
            .specialist_memory_worker_influence_direction,
        specialist_memory_worker_influence_layers: specialist_worker_influence_layers_tuple,
        specialist_memory_worker_answer_window_active: false,
        specialist_memory_worker_pre_answer_active: true,
        specialist_memory_worker_pre_earned_active: true,
        specialist_memory_worker_was_pre_answer_active: true,
        specialist_memory_worker_at_boundary_active: false,
        specialist_memory_worker_fixed_packet_id: args
            .specialist_memory_worker_fixed_packet_id
            .clone(),
        runtime_motifs,
        runtime_recovery_ops,
        control_token_ids,
        hidden_request_profiles,
        motif_force_scale: scaling_profile
            .as_ref()
            .map(|profile| profile.motif_force_scale)
            .unwrap_or(0.35),
        bridge_motif_gate_floor: args.bridge_motif_gate_floor,
        recovery_force_scale: scaling_profile
            .as_ref()
            .map(|profile| profile.recovery_force_scale)
            .unwrap_or(0.45),
        guardrail_bias_scale: scaling_profile
            .as_ref()
            .map(|profile| profile.guardrail_bias_scale)
            .unwrap_or(1.5),
        black_hole_embeddings: black_hole_embeddings_vec,
        // Phase 1 Telemetry defaults
        last_force_trace: None,
        last_gravity_mag: 0.0,
        last_ghost_pre_norm: 0.0,
        last_ghost_gain: args.ghost_gravity as f32,
        last_applied_ghost_mag: 0.0,
        last_applied_ghost_vector: None,
        last_goal_mag: 0.0,
        last_repulsion_mag: 0.0,
        last_activation_gate: 0.0,
        last_motif_mag: 0.0,
        last_bridge_force_selection: args.bridge_force_selection.as_str().to_string(),
        last_bridge_force_selected_count: 0,
        last_bridge_force_selected_ids: Vec::new(),
        last_bridge_force_selection_source: "none".to_string(),
        last_bridge_force_selected_score_max: None,
        last_bridge_force_selected_role: None,
        last_bridge_force_second_score: None,
        last_bridge_force_selected_margin: None,
        last_bridge_force_role_filter: args.bridge_force_role_filter.as_str().to_string(),
        last_bridge_force_min_margin: args.bridge_force_min_margin,
        last_recovery_mag: 0.0,
        last_absence_signal: 0.0,
        last_trap_score: 0.0,
        last_live_motif_count: 0,
        last_live_motif_distance: 0.0,
        last_live_motif_radius: 0.0,
        last_live_basin_pressure: 0.0,
        last_guardrail_active: false,
        last_forces_applied: false,
        last_engine_status: ForceEngineStatus::Idle,
        last_wobble_pressure_crossing: false,
        last_task_anchor_clamp: None,
        orbital_active: args.mode_orbital,
        momentum: vec![0.0; hidden_dim],
        braking: false,
        dynamic_gravity: args.gravity,
        dynamic_repulsion: args.repulsion_strength as f32,
        // Phase 4: Heartbeat + Defibrillator
        stress_buffer: VecDeque::with_capacity(10),
        heartbeat_blend: args.physics_blend,
        heartbeat_gravity: args.gravity_well,
        heartbeat_repulsion: args.repulsion_strength as f32,
        stress_level: 0.0,
        boredom_level: 0.0,
        empathy_spike: 0.0,
        defibrillator_active: false,
        defib_cooldown: 0,
        adrenaline: 0.0,
        // Phase 3: The Mirror
        pending_insight: None,
        last_insight_step: 0,
        insight_persistence: 0,
        // Phase 4: Autonomic Override
        request_count: 0,
        last_request_token: 0,
        visible_request_gate: args.visible_request_gate,
        request_buffer: String::new(),
        surface_buffer: String::new(),
        hidden_request_candidate: None,
        hidden_request_streak: 0,
        last_hidden_request: None,
        last_hidden_request_pressure: 0.0,
        hidden_request_activations: 0,
        current_turn_structure_bias: 0.0,
        current_task_anchor_signature: None,
        task_anchor_similarity_start: 0.0,
        task_anchor_similarity_hinge: 0.0,
        task_anchor_similarity_24tok: 0.0,
        task_anchor_drift: 0.0,
        task_anchor_window_tokens_seen: 0,
        first_promotion_attempt_step: None,
        structured_streak: 0,
        max_structured_streak: 0,
        promotion_attempt_count: 0,
        promotion_failure_count: 0,
        first_organic_promoted_step: None,
        first_recovered_promoted_step: None,
        motif_restore_bias_steps_remaining: 0,
        motif_restore_bias_strength: 0.0,
        reentry_clamp_steps_remaining: 0,
        reentry_clamp_strength: 0.0,
        reentry_temp_scale: 1.0,
        motif_regression_assist_steps_remaining: 0,
        motif_regression_assist_strength: 0.0,
        restored_run_active: false,
        current_run_id: build_run_id(&args),
        routing_cache: None,
        controller_tick_count: 0,
        controller_selected_structured_count: 0,
        controller_selected_structured_candidate_count: 0,
        controller_selected_conversational_count: 0,
        conflict_tie_break_count: 0,
        structured_basin_lock_count: 0,
        neutral_basin_penalty_applied: 0,
        task_utility_bonus_applied: 0,
        structured_candidate_escalation_attempts: 0,
        structured_candidate_escalation_wins: 0,
        structured_candidate_loss_reason_counts: BTreeMap::new(),
        structured_resume_window_remaining: 0,
        structured_resume_conversational_hits: 0,
        last_routed_motif_id: None,
        last_routed_motif_role: None,
        last_routed_motif_score: f32::INFINITY,
        last_controller_candidates: Vec::new(),
        hinge_window_records: Vec::new(),
        continuity_support_scale: continuity_scale.support_scale,
        continuity_release_scale: continuity_scale.release_scale,
        ablate_periodic_controller: args.ablate_periodic_controller,
        ablate_live_motifs: args.ablate_live_motifs,
        ablate_conflict_routing: args.ablate_conflict_routing,
        ablate_reentry_clamp: args.ablate_reentry_clamp,
        ablate_crystal_ratchet: args.ablate_crystal_ratchet,
        ablate_promotion_override: args.ablate_promotion_override,
        dev_structured_candidate_task_sim: args.dev_structured_candidate_task_sim,
        dev_structured_candidate_bonus_scale: args.dev_structured_candidate_bonus_scale,
        dev_neutral_basin_penalty_scale: args.dev_neutral_basin_penalty_scale,
        dev_task_utility_bonus_scale: args.dev_task_utility_bonus_scale,
        dev_fragmentation_discount: args.dev_fragmentation_discount,
        dev_restored_topology_floor_signal: args.dev_restored_topology_floor_signal,
        dev_restored_topology_floor_tightness: args.dev_restored_topology_floor_tightness,
        dev_structured_candidate_escalation_topology: args
            .dev_structured_candidate_escalation_topology,
        dev_structured_candidate_escalation_task: args.dev_structured_candidate_escalation_task,
        dev_routing_stickiness_bonus: args.dev_routing_stickiness_bonus,
        dev_routing_stickiness_ticks: args.dev_routing_stickiness_ticks,
        routing_stickiness_motif_id: None,
        routing_stickiness_remaining_ticks: 0,
        focus_lock_remaining_ticks: 0,
        focus_lock_max_ticks: scaling_profile
            .as_ref()
            .map(|profile| profile.focus_lock_ticks)
            .unwrap_or(30),
        tda_shadow_monitor_enabled: args.tda_shadow_monitor,
        tda_shadow_breath_apply: args.tda_shadow_breath_apply,
        tda_shadow_monitor: crate::runtime::tda_monitor::TdaShadowMonitor::new(
            args.tda_shadow_window,
            args.tda_shadow_stride,
        ),

        // Bridge Telemetry (niodv4_bridge)
        bridge_enabled: cfg!(feature = "niodv4_bridge") && !args.bridge_off,
        bridge_influence_smoke: cfg!(feature = "niodv4_bridge")
            && !args.bridge_off
            && args.bridge_influence_smoke,
        bridge_influence_smoke_clamp: args.bridge_influence_smoke_clamp,
        bridge_influence_selective: cfg!(feature = "niodv4_bridge")
            && !args.bridge_off
            && args.bridge_influence_selective,
        bridge_gate34_latch: cfg!(feature = "niodv4_bridge")
            && !args.bridge_off
            && args.bridge_gate34_latch,
        gate34_warmup_steps: args.gate34_warmup_steps,
        gate34_hold_steps: args.gate34_hold_steps,
        gate34_release_margin_floor: args.gate34_release_margin_floor,
        gate34_release_patience: args.gate34_release_patience,
        gate34_release_distance_mult: args.gate34_release_distance_mult,
        gate34_acquire_top_k: args.gate34_acquire_top_k.max(1),
        bridge_prompt_weight: args.bridge_prompt_weight,
        gate34_target_source: args.gate34_target_source.clone(),
        gate34_motif_routing_safety_floor: args.gate34_motif_routing_safety_floor,
        prompt_vec: None,
        prompt_vec_norm: 0.0,
        prompt_embedding_source: "unavailable:init".to_string(),
        prompt_similarity_unavailable: true,
        gate34_acquisition_candidates: Vec::new(),
        gate34_phase: if cfg!(feature = "niodv4_bridge")
            && !args.bridge_off
            && args.bridge_gate34_latch
        {
            Gate34Phase::Warmup
        } else {
            Gate34Phase::Inactive
        },
        gate34_target_ghost_id: None,
        gate34_target_specialist_id: None,
        gate34_target_motif_id: None,
        gate34_target_vector: None,
        gate34_target_acquired_step: -1,
        gate34_target_margin_at_acquire: 0.0,
        gate34_target_distance_at_acquire: 0.0,
        gate34_current_target_distance: 0.0,
        gate34_warmup_step_count: 0,
        gate34_held_step_count: 0,
        gate34_last_step: -1,
        gate34_bad_margin_count: 0,
        gate34_bad_distance_count: 0,
        gate34_release_reason: None,
        gate34_intervention_count: 0,
        gate34_target_switch_count: 0,
        gate34_candidate_counts: HashMap::new(),
        gate34_candidate_margin_sum: HashMap::new(),
        gate34_candidate_best_margin: HashMap::new(),
        gate34_candidate_distance_sum: HashMap::new(),
        gate34_candidate_distance_sq_sum: HashMap::new(),
        gate34_candidate_distance_min: HashMap::new(),
        gate34_candidate_distance_max: HashMap::new(),
        gate34_target_warmup_distance_min: 0.0,
        gate34_target_warmup_distance_mean: 0.0,
        gate34_target_warmup_distance_max: 0.0,
        gate34_target_warmup_distance_std: 0.0,
        gate34_last_distance_drift_score: 0.0,
        gate34_last_distance_limit_ratio: 0.0,
        gate34_last_distance_limit_warmup: 0.0,
        gate34_last_distance_gate_mode: "ratio_only".to_string(),
        bridge_margin_threshold: args.bridge_margin_threshold,
        bridge_stability_k: args.bridge_stability_k,
        bridge_cooldown_after_switch: args.bridge_cooldown_after_switch,
        bridge_scale_by_margin: args.bridge_scale_by_margin,
        last_ghost_id_run_length: 0,
        last_ghost_switch_cooldown_remaining: 0,
        last_bridge_counter_step: -1,
        last_bridge_cooldown_step: -1,
        ghost_basins_loaded: 0,
        last_prompt_hash: crate::main_helpers::sha256_hex(args.prompt.as_bytes()),
        last_nearest_ghost_id: None,
        last_nearest_ghost_distance: 0.0,
        last_second_nearest_ghost_distance: 0.0,
        last_route_margin: 0.0,
        last_bridge_route_probe_64d: Vec::new(),
        last_projection_strategy: "none".to_string(),
        last_ghost_pull_delta_norm: 0.0,
        last_intervention_applied: false,
        active_recovery_specialist_id: None,
        active_recovery_weight: 0.0,
        specialist_run_length: 0,
        last_recovery_specialist_id: None,
        last_recovery_counter_step: -1,
        last_specialist_worker_enabled: false,
        last_specialist_worker_selected_id: None,
        last_specialist_worker_packet_id: None,
        last_specialist_worker_unicode_escape: None,
        last_specialist_worker_original_route_id: None,
        last_specialist_worker_decoded_route_id: None,
        last_specialist_worker_route_preserved: None,
        last_specialist_worker_topk_hit: None,
        last_specialist_worker_score: None,
        last_specialist_worker_source_prompt_id: None,
        last_specialist_worker_direction_source: None,
        last_specialist_worker_delta_norm_64d: None,
        last_specialist_worker_hidden_delta_norm: None,
        last_specialist_worker_influence_clamp: None,
        last_specialist_worker_influence_scale: None,
        last_specialist_worker_probe_signature_64d: None,
        last_specialist_worker_target_signature_64d: None,

        // Ghost Registry for niodv4_bridge
        #[cfg(feature = "niodv4_bridge")]
        ghost_registry: None,

        // VQ codec + phase2 specialist
        #[cfg(feature = "niodv4_bridge")]
        vq_codebook: None,
        #[cfg(feature = "niodv4_bridge")]
        vq_specialist: None,
        last_vq_code_assigned: None,
        last_vq_encode_error: 0.0,
        last_correction_delta_norm: 0.0,
        last_specialist_activated: false,
        specialist_correction_apply: args.specialist_correction_apply,
        specialist_correction_clamp: args.specialist_correction_clamp,
        last_specialist_force_applied: false,
        last_specialist_force_norm: 0.0,
        last_correction_packet_vq_code: None,
        #[cfg(feature = "niodv4_bridge")]
        correction_packets: None,
        correction_packet_clamp: args.correction_packet_clamp,
        correction_packet_payload_blend: args.correction_packet_payload_blend,
        last_correction_packet_fire_count: 0,
        last_correction_packet_force_norm: 0.0,
        last_correction_packet_ids: Vec::new(),
        last_probe_bucket_mean_64: None,

        // REMEMBER vault tether (quiet queue + 64D probe search + self-save for "read own creation")
        vault_client: None, // populated below if qdrant url present
        vault_collection: "niodoo-4096-vault".to_string(),
        correction_packets_out: args.correction_packets_out.clone(),
        correction_packet_out_unicode_v3: args.correction_packet_out_unicode_v3,
        correction_packet_out_pull_strength: args.correction_packet_out_pull_strength,
        correction_packet_out_distance_threshold: args.correction_packet_out_distance_threshold,
        correction_packet_lock_pull_strength: args.correction_packet_lock_pull_strength,
        correction_packet_lock_contradiction_multiplier: args
            .correction_packet_lock_contradiction_multiplier,
        correction_packet_invalidate_on_contradiction: args
            .correction_packet_invalidate_on_contradiction,
        correction_packet_revalidate_on_affirmation: args
            .correction_packet_revalidate_on_affirmation,
        correction_packet_adaptive_contradiction_cap: args
            .correction_packet_adaptive_contradiction_cap,
        contradiction_counts: std::collections::HashMap::new(),
        correction_contradiction_counts_path: args.correction_contradiction_counts_path.clone(),
        correction_packet_unfold_on_retry_count: args.correction_packet_unfold_on_retry_count,
        last_mistake_reflex_retry_count: 0,
        correction_packet_unfold_retry_factor: args.correction_packet_unfold_retry_factor,
        correction_packet_eviction_floor: args.correction_packet_eviction_floor,
        correction_packet_decay_rate: args.correction_packet_decay_rate,
        correction_packet_unfold_encode_error_threshold: args
            .correction_packet_unfold_encode_error_threshold,
        correction_packet_unfold_factor: args.correction_packet_unfold_factor,
        correction_packet_competence_suppress_factor: args
            .correction_packet_competence_suppress_factor,
        correction_packet_competence_entropy_threshold: args
            .correction_packet_competence_entropy_threshold,
        correction_packet_competence_density_threshold: args
            .correction_packet_competence_density_threshold,
        correction_packet_competence_distance_threshold: args
            .correction_packet_competence_distance_threshold,
        correction_packet_competence_combine_mode: args
            .correction_packet_competence_combine_mode
            .clone(),
        correction_packet_trajectory_routing: args.correction_packet_trajectory_routing,
        correction_packet_trajectory_classify_step: args.correction_packet_trajectory_classify_step,
        correction_packet_trajectory_fire_count_threshold: args
            .correction_packet_trajectory_fire_count_threshold,
        correction_packet_trajectory_top_k_competent: args
            .correction_packet_trajectory_top_k_competent,
        correction_packet_trajectory_top_k_drifting: args
            .correction_packet_trajectory_top_k_drifting,
        trajectory_fire_count_sum: 0.0,
        trajectory_fire_count_samples: 0,
        trajectory_classified: None,
        trajectory_turn_step: 0,
        trajectory_last_classified_step: usize::MAX,
        trajectory_pending_step_fires: 0,
        correction_packet_suppress_when_bridge_force_above: args
            .correction_packet_suppress_when_bridge_force_above,
        prev_step_max_ghost_mag: 0.0,
        correction_packet_post_bridge_mode: args.correction_packet_post_bridge_mode,
        correction_packet_readiness_lock_threshold: args.correction_packet_readiness_lock_threshold,
        readiness_lock_skip_count: 0,
        last_readiness_lock_source: String::new(),
        last_readiness_lock_score: 0.0,
        physics_blend_deep_layer_from: args.physics_blend_deep_layer_from,
        physics_blend_deep_layer_multiplier: args.physics_blend_deep_layer_multiplier,
        physics_blend_deep_layer_mask_count: 0,
        motif_routing_consensus_weight: args.motif_routing_consensus_weight,
        autonomic_physics_force_threshold: args.autonomic_physics_force_threshold,
        autonomic_physics_window_size: args.autonomic_physics_window,
        autonomic_physics_force_window: VecDeque::with_capacity(args.autonomic_physics_window),
        autonomic_physics_motif_scale_origin: scaling_profile
            .as_ref()
            .map(|profile| profile.motif_force_scale)
            .unwrap_or(0.35),
        autonomic_physics_recovery_scale_origin: scaling_profile
            .as_ref()
            .map(|profile| profile.recovery_force_scale)
            .unwrap_or(0.45),
        autonomic_physics_scale_down_count: 0,
        autonomic_physics_scale_up_count: 0,
        correction_packet_arbitration_mode,
        correction_packet_arbitration_healthy_factor_threshold,
        correction_packet_arbitration_stale_distance_threshold,
        last_correction_packet_arbitration_mode: "disabled".to_string(),
        last_correction_packet_arbitration_reason: "not_evaluated".to_string(),
        last_correction_packet_arbitration_candidate_count: 0,
        last_correction_packet_arbitration_min_target_distance: f32::INFINITY,
        last_correction_packet_arbitration_force_norm_estimate: 0.0,
        codec_active_prompt_substrings: args
            .codec_active_prompt_substrings
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .filter(|s| !s.is_empty())
            .collect(),
        // Default true so unconfigured runs are unchanged. The per-turn
        // hook overrides this when the gate is configured.
        codec_active_for_current_prompt: true,
        correction_packet_prompt_top_k_map: parse_correction_packet_prompt_top_k_map(
            &args.correction_packet_prompt_top_k_map,
        ),
        current_prompt_top_k_override: None,
        current_prompt_top_k_match_substring: None,
        correction_packet_suppress_when_no_prompt_match: args
            .correction_packet_suppress_when_no_prompt_match,
        correction_packet_suppress_for_current_prompt: false,
        correction_packet_prompt_source_target_map:
            parse_correction_packet_prompt_source_target_map(
                &args.correction_packet_prompt_source_target_map,
            ),
        current_prompt_source_target_override: None,
        last_correction_packet_effective_fire_top_k: args.correction_packet_fire_top_k,
        correction_packet_total_clamp: args.correction_packet_total_clamp,
        correction_packet_fire_max_distance: args.correction_packet_fire_max_distance,
        correction_packet_fire_top_k: args.correction_packet_fire_top_k,
        correction_packet_mint_bucket_cap: args.correction_packet_mint_bucket_cap,
        mint_bucket_counts: std::collections::HashMap::new(),
        correction_packet_capture_every_turn: args.correction_packet_capture_every_turn,
        correction_packet_fire_step_window: parse_step_window(
            &args.correction_packet_fire_step_window,
        ),
        correction_packet_fire_match_prompt_hash: args.correction_packet_fire_match_prompt_hash,
        current_prompt_hash: String::new(),
        correction_packet_authority_mode: args.correction_packet_authority_mode,
        last_packet_authority_score: 0.0,
        last_packet_authority_allowed: false,
        last_packet_authority_reason: "not_evaluated".to_string(),
        last_packet_authority_blocked_reason: "not_evaluated".to_string(),
        last_correction_packet_min_target_distance: f32::INFINITY,
        last_sampling_entropy_norm: 0.0,
        last_correction_packet_effective_pull_avg: 0.0,
        last_correction_packet_unfold_active: false,
        last_correction_packet_vq_encode_error: 0.0,
        last_correction_packet_unfold_factor_applied: 1.0,
        last_correction_packet_competence_factor: 1.0,
        correction_packets_state_out: args.correction_packets_state_out.clone(),
    };

    #[cfg(feature = "niodv4_bridge")]
    {
        if !args.bridge_off {
            let registry_path = "niodv4/data/results/summaries/ghost_candidate_registry.json";
            if let Ok(registry) =
                crate::bridge::registry::GhostRegistry::load_from_path(registry_path)
            {
                phys_engine.ghost_basins_loaded = registry.basin_count();
                phys_engine.last_projection_strategy = "simple".to_string();
                phys_engine.ghost_registry = Some(registry);
                if chat_stdout {
                    eprintln!(
                        " [BRIDGE] Loaded {} ghost basins from {}",
                        phys_engine.ghost_basins_loaded, registry_path
                    );
                } else {
                    println!(
                        " [BRIDGE] Loaded {} ghost basins from {}",
                        phys_engine.ghost_basins_loaded, registry_path
                    );
                }
            } else {
                if chat_stdout {
                    eprintln!(
                        " [BRIDGE] No ghost basins loaded (registry not found at {})",
                        registry_path
                    );
                } else {
                    println!(
                        " [BRIDGE] No ghost basins loaded (registry not found at {})",
                        registry_path
                    );
                }
            }
        } else {
            if chat_stdout {
                eprintln!(" [BRIDGE] Runtime bridge disabled via --bridge-off");
            } else {
                println!(" [BRIDGE] Runtime bridge disabled via --bridge-off");
            }
        }

        // Load VQ codebook if --codebook-path provided.
        if let Some(ref cb_path) = args.codebook_path {
            match crate::bridge::CodebookVQ::load_from_json(cb_path) {
                Ok(cb) => {
                    let msg = format!(
                        " [VQ] Loaded codebook: {} entries × 64D from {}",
                        cb.entries.len(),
                        cb_path.display()
                    );
                    if chat_stdout {
                        eprintln!("{}", msg);
                    } else {
                        println!("{}", msg);
                    }
                    phys_engine.vq_codebook = Some(cb);
                }
                Err(e) => {
                    let msg = format!(
                        " [VQ] Failed to load codebook from {}: {}",
                        cb_path.display(),
                        e
                    );
                    if chat_stdout {
                        eprintln!("{}", msg);
                    } else {
                        println!("{}", msg);
                    }
                }
            }
        }

        // Load rule-based specialist if --specialist-params-path provided.
        if let Some(ref sp_path) = args.specialist_params_path {
            match crate::bridge::RuleBasedSpecialist::load_from_json(sp_path) {
                Ok(sp) => {
                    let msg = format!(
                        " [VQ] Loaded specialist: target=[{:.6},{:.6}] pull={} threshold={} from {}",
                        sp.target_coords[0], sp.target_coords[1],
                        sp.pull_strength, sp.distance_threshold,
                        sp_path.display()
                    );
                    if chat_stdout {
                        eprintln!("{}", msg);
                    } else {
                        println!("{}", msg);
                    }
                    phys_engine.vq_specialist = Some(sp);
                }
                Err(e) => {
                    let msg = format!(
                        " [VQ] Failed to load specialist from {}: {}",
                        sp_path.display(),
                        e
                    );
                    if chat_stdout {
                        eprintln!("{}", msg);
                    } else {
                        println!("{}", msg);
                    }
                }
            }
        }

        // Load VQ-keyed correction packet store if --correction-packets-path provided.
        if let Some(ref cp_path) = args.correction_packets_path {
            match crate::bridge::CorrectionPacketStore::load_from_jsonl(cp_path) {
                Ok(store) => {
                    let msg = format!(
                        " [VQ] Loaded {} correction packets from {}",
                        store.total(),
                        cp_path.display()
                    );
                    if chat_stdout {
                        eprintln!("{}", msg);
                    } else {
                        println!("{}", msg);
                    }
                    phys_engine.correction_packets = Some(store);
                }
                Err(e) => {
                    let msg = format!(
                        " [VQ] Failed to load correction packets from {}: {}",
                        cp_path.display(),
                        e
                    );
                    if chat_stdout {
                        eprintln!("{}", msg);
                    } else {
                        println!("{}", msg);
                    }
                }
            }
        }

        // Report combined load status for smoke verification.
        let specialists_loaded = if phys_engine.vq_specialist.is_some() {
            1
        } else {
            0
        };
        let codebook_loaded = phys_engine.vq_codebook.is_some();
        let correction_packets_loaded = phys_engine
            .correction_packets
            .as_ref()
            .map(|s| s.total())
            .unwrap_or(0);
        let msg = format!(
            " [VQ] specialists_loaded={} codebook_loaded={} correction_packets={}",
            specialists_loaded, codebook_loaded, correction_packets_loaded
        );
        if chat_stdout {
            eprintln!("{}", msg);
        } else {
            println!("{}", msg);
        }
    }

    // REMEMBER vault tether client (64D probe live query + self-save).
    // Uses env so the py runners (read_aloud, mint scripts) can control the "MCP" / 6360 endpoint
    // without new CLI flags in the first wiring. The 64D probe from the exact REMEMBER step is used.
    {
        let vault_url = std::env::var("NIODOO_VAULT_URL")
            .ok()
            .map(|value| value.trim().to_string())
            .unwrap_or_default();
        let vault_coll = std::env::var("NIODOO_VAULT_COLLECTION")
            .unwrap_or_else(|_| "niodoo-4096-vault".to_string());
        if !vault_url.is_empty() && !vault_url.eq_ignore_ascii_case("off") {
            phys_engine.vault_client = Some(crate::runtime::vault_retrieval::VaultClient::new(
                &vault_url,
                &vault_coll,
            ));
            phys_engine.vault_collection = vault_coll.clone();
            let msg = format!(
                " [REMEMBER-VAULT] client ready url={} collection={}",
                vault_url, vault_coll
            );
            if chat_stdout {
                eprintln!("{}", msg);
            } else {
                println!("{}", msg);
            }
        }
    }

    // Load persisted contradiction counts so escalation accumulates across sessions
    // (§10ad). Missing file is fine for first-run; malformed lines are skipped
    // with a warning rather than aborting startup.
    if let Some(path) = phys_engine.correction_contradiction_counts_path.as_ref() {
        match load_contradiction_counts(path) {
            Ok(counts) => {
                let n = counts.len();
                phys_engine.contradiction_counts = counts;
                let msg = format!(
                    " [CONTRADICTION_COUNTS] Loaded {} key(s) from {}",
                    n,
                    path.display()
                );
                if chat_stdout {
                    eprintln!("{}", msg);
                } else {
                    println!("{}", msg);
                }
            }
            Err(e) => {
                let msg = format!(
                    " [CONTRADICTION_COUNTS] Failed to load {}: {} — starting fresh",
                    path.display(),
                    e
                );
                if chat_stdout {
                    eprintln!("{}", msg);
                } else {
                    println!("{}", msg);
                }
            }
        }
    }

    phys_engine.refresh_runtime_motif_metadata()?;

    if phys_engine.bridge_gate34_latch_active() {
        let _ = phys_engine.compute_user_prompt_vec(&model, &args.prompt, &device);
    }

    if let Some(secret_sauce) = &args.secret_sauce {
        let decoded = decode_secret_sauce(secret_sauce, args.secret_sauce_version)?;
        let hidden_projected =
            project_bridge_vector_to_hidden(&decoded.segments.hidden_64, hidden_dim, &device)?;
        phys_engine.secret_sauce_version = Some(decoded.version);
        phys_engine.secret_sauce_steps_remaining = phys_engine.secret_sauce_decay_steps;
        let mut compact_motif_signal = 0.0f32;
        let mut sentence_restored = false;
        let mut momentum_restored = false;
        let mut hidden_restored = false;

        if decoded.version == SecretSauceVersion::V3 {
            phys_engine.secret_sauce_sentence_prior = Some(hidden_projected.detach());
            phys_engine.inject_sentence_context_motif(&hidden_projected, 0.32)?;
            sentence_restored = true;
        } else {
            phys_engine.goal_embedding = Some(hidden_projected.detach());
            phys_engine.secret_sauce_hidden_prior = Some(hidden_projected.detach());
            phys_engine.inject_sentence_context_motif(&hidden_projected, 0.22)?;
            hidden_restored = true;
        }

        if decoded.version == SecretSauceVersion::V2 {
            if decoded.segments.sentence_32.len() == 32 {
                let sentence_projected = project_bridge_vector_to_hidden(
                    &decoded.segments.sentence_32,
                    hidden_dim,
                    &device,
                )?;
                phys_engine.secret_sauce_sentence_prior = Some(sentence_projected.detach());
                if decoded.segments.control_8.len() == 8 {
                    compact_motif_signal = decoded.segments.control_8[7].clamp(0.0, 1.0);
                    phys_engine
                        .inject_compact_runtime_motif(&sentence_projected, compact_motif_signal)?;
                }
                sentence_restored = true;
            }
            if decoded.segments.momentum_16.len() == 16 {
                let momentum_projected = project_bridge_vector_to_hidden(
                    &decoded.segments.momentum_16,
                    hidden_dim,
                    &device,
                )?;
                phys_engine.momentum_buffer = Some(momentum_projected.detach());
                phys_engine.secret_sauce_momentum_prior = Some(momentum_projected.detach());
                momentum_restored = true;
            }
            if decoded.segments.scalar_8.len() == 8 {
                phys_engine.last_motif_mag = decoded.segments.scalar_8[0].clamp(0.0, 4.0);
                phys_engine.last_recovery_mag = decoded.segments.scalar_8[1].clamp(0.0, 4.0);
                phys_engine.last_absence_signal = decoded.segments.scalar_8[2].clamp(0.0, 3.0);
                phys_engine.last_trap_score = decoded.segments.scalar_8[3].clamp(0.0, 3.0);
                phys_engine.stress_level = decoded.segments.scalar_8[4].clamp(0.0, 15.0);
                phys_engine.boredom_level = decoded.segments.scalar_8[5].clamp(0.0, 4.0);
                phys_engine.dynamic_gravity = decoded.segments.scalar_8[6].clamp(0.0, 4.0);
                phys_engine.dynamic_repulsion = decoded.segments.scalar_8[7].clamp(-6.0, 6.0);
            }
            if decoded.segments.control_8.len() == 8 {
                phys_engine.physics_blend = decoded.segments.control_8[0].clamp(0.0, 6.5);
                phys_engine.last_guardrail_active = decoded.segments.control_8[1] > 0.0;
                phys_engine.orbital_active = decoded.segments.control_8[2] > 0.0;
                phys_engine.request_count =
                    decoded.segments.control_8[3].round().clamp(0.0, 16.0) as usize;
                phys_engine.runtime_mode = if decoded.segments.control_8[4] > 0.5 {
                    RuntimeMode::Clean
                } else if decoded.segments.control_8[4] < -0.5 {
                    RuntimeMode::Research
                } else {
                    RuntimeMode::Agency
                };
                phys_engine.insight_persistence =
                    decoded.segments.control_8[5].round().clamp(0.0, 32.0) as usize;
                phys_engine.empathy_spike = decoded.segments.control_8[6].clamp(0.0, 2.0);
            }
        }

        if compact_motif_signal > 0.05 {
            let assist_steps = (16.0 + compact_motif_signal * 24.0).round() as usize;
            phys_engine.motif_restore_bias_steps_remaining = phys_engine
                .motif_restore_bias_steps_remaining
                .max(assist_steps);
            phys_engine.motif_restore_bias_strength = phys_engine
                .motif_restore_bias_strength
                .max((0.12 + compact_motif_signal * 0.32).clamp(0.12, 0.48));
        }

        println!(
            " [SECRET_SAUCE] restored version={} hidden={} sentence={} momentum={} decay_steps={} state=\"{}\"",
            decoded.version.as_str(),
            hidden_restored,
            sentence_restored,
            momentum_restored,
            phys_engine.secret_sauce_steps_remaining,
            secret_sauce
        );
    }

    let mut restored_index_pos: Option<usize> = None;
    let mut previous_motif_continuity_artifact: Option<MotifContinuityArtifact> = None;
    let mut restored_reference_motifs: Option<Vec<RuntimeMotifSnapshot>> = None;
    let mut initial_restored_motif_provenance: Option<MotifProvenanceSummary> = None;
    if let Some(path) = &args.kv_state_load_file {
        previous_motif_continuity_artifact = load_motif_continuity_artifact(path)?;
        let file = File::open(path)
            .with_context(|| format!("Failed to open kv state {}", path.display()))?;
        let snapshot: KvStateRecord = serde_json::from_reader(std::io::BufReader::new(file))
            .with_context(|| format!("Failed to parse kv state {}", path.display()))?;
        model.import_kv_cache_snapshot(&snapshot.kv_cache, &device)?;
        if let Some(state_packet) = &snapshot.state_packet {
            restored_reference_motifs = Some(state_packet.motif_state.runtime_motifs.clone());
            state_packet.restore_into(&mut phys_engine, hidden_dim, &device)?;
        } else if let Some(engine_state) = &snapshot.engine_state {
            restored_reference_motifs = Some(engine_state.runtime_motifs.clone());
            engine_state.restore_into(&mut phys_engine, hidden_dim, &device)?;
        }
        if let Some(summary) = &snapshot.motif_carry_forward {
            phys_engine.apply_restore_continuity_assist(summary);
        }
        if let Some(previous_artifact) = &previous_motif_continuity_artifact {
            if let Some(comparison) = &previous_artifact.comparison_to_previous {
                phys_engine.apply_prior_continuity_policy(comparison);
            }
        }
        phys_engine.restored_run_active = true;
        phys_engine.refresh_runtime_motif_metadata()?;
        initial_restored_motif_provenance = Some(summarize_runtime_motif_provenance(
            &phys_engine.runtime_motifs,
        ));
        restored_index_pos = Some(snapshot.index_pos);
        println!(
            " [KV_STATE] restored={} index_pos={} layers={} state_packet={} engine_state={} preview=\"{}\"",
            path.display(),
            snapshot.index_pos,
            snapshot.kv_cache.layers.len(),
            if snapshot.state_packet.is_some() {
                "yes"
            } else {
                "no"
            },
            if snapshot.engine_state.is_some() {
                "yes"
            } else {
                "no"
            },
            snapshot.assistant_preview
        );
    }

    if let Some(dir) = &args.turn_capture_dir {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create turn capture dir {}", dir.display()))?;
    }
    if let Some(dir) = &args.runtime_hidden_capture_dir {
        std::fs::create_dir_all(dir).with_context(|| {
            format!(
                "Failed to create runtime hidden capture dir {}",
                dir.display()
            )
        })?;
    }

    let active_context_adapter_decisions_for_run =
        if let Some(path) = &args.active_context_adapter_decisions {
            Some(load_runtime_adapter_decisions(path)?)
        } else {
            None
        };
    let active_context_turn_start_state_for_run: Option<ActiveContextRuntimeTurnStartState> =
        active_context_adapter_decisions_for_run
            .as_ref()
            .map(|decisions| runtime_turn_start_state(decisions));

    if !args.goal.is_empty() {
        if let Some(idx) = phys_engine
            .particle_words
            .iter()
            .position(|w| w.eq_ignore_ascii_case(&args.goal))
        {
            if let Ok(emb) = phys_engine.charge_tensor.i(idx) {
                let emb_f32 = emb.to_dtype(DType::F32)?;
                let goal_dim = emb_f32.dim(0)?;

                let projected = if goal_dim != hidden_dim {
                    if let Some(proj) = &phys_engine.proj_matrix {
                        emb_f32.matmul(proj)?
                    } else {
                        // Fallback padding if no projection matrix (should exist if dims differ)
                        println!(
                            " [WARN] Goal dim {} != Hidden {}, but no projection matrix found!",
                            goal_dim, hidden_dim
                        );
                        // Zero-pad or crop
                        if goal_dim < hidden_dim {
                            let pad = Tensor::zeros((hidden_dim - goal_dim,), DType::F32, &device)?;
                            Tensor::cat(&[&emb_f32, &pad], 0)?
                        } else {
                            emb_f32.narrow(0, 0, hidden_dim)?
                        }
                    }
                } else {
                    emb_f32
                };

                phys_engine.goal_embedding = Some(projected.detach());
                println!(" [GOAL] Attractor set to '{}'", args.goal);
            }
        }
    }

    // PROMPT FORMATTING
    let default_system_prompt = default_runtime_system_prompt();
    let system_prompt = if let Some(path) = &args.system_prompt_file {
        std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read system prompt {}", path.display()))?
    } else {
        default_system_prompt.to_string()
    };
    let session_prompts = if let Some(path) = &args.session_script {
        let prompts = load_session_prompts(path)?;
        if chat_stdout {
            eprintln!(
                " [SESSION] source={} turns={} persistent_kv=true",
                path.display(),
                prompts.len()
            );
        } else {
            println!(
                " [SESSION] source={} turns={} persistent_kv=true",
                path.display(),
                prompts.len()
            );
        }
        prompts
    } else {
        vec![args.prompt.clone()]
    };
    let mut compact_resume_state = if let Some(path) = &args.compact_resume_state_load_file {
        let state = load_compact_resume_state(path)?;
        if chat_stdout {
            eprintln!(
                " [COMPACT_RESUME] loaded={} anchors={} turns={}",
                path.display(),
                state.has_anchors(),
                state.turn_count
            );
        } else {
            println!(
                " [COMPACT_RESUME] loaded={} anchors={} turns={}",
                path.display(),
                state.has_anchors(),
                state.turn_count
            );
        }
        state
    } else {
        CompactResumeState::new()
    };
    let mut agency_hands_state = AgencyHandsState::new();
    let mut mistake_memory = if let Some(path) = &args.mistake_memory_path {
        let memory = MistakeMemory::load(path)?;
        if chat_stdout {
            eprintln!(
                " [MISTAKE_MEMORY] loaded={} events={}",
                path.display(),
                memory.len()
            );
        }
        memory
    } else {
        MistakeMemory::default()
    };
    let mut mistake_reflex_memory = if let Some(path) = &args.mistake_reflex_path {
        let mut memory = MistakeReflexMemory::load(path)?;
        if let Some(packet_index) = &args.mistake_reflex_packet_index {
            let attached = memory.attach_vector_slices_from_packet_index(packet_index)?;
            if attached > 0 {
                memory.save(path)?;
            }
            if chat_stdout {
                eprintln!(
                    " [MISTAKE_REFLEX] packet_index={} attached={}",
                    packet_index.display(),
                    attached
                );
            }
        }
        if chat_stdout {
            eprintln!(
                " [MISTAKE_REFLEX] loaded={} events={} mode={:?} action_mode={}",
                path.display(),
                memory.len(),
                args.mistake_reflex_mode,
                args.mistake_reflex_action_mode.as_str()
            );
        }
        memory
    } else {
        MistakeReflexMemory::default()
    };

    // Seeded RNG for reproducibility
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut index_pos = restored_index_pos.unwrap_or(0); // Track position for KV cache
    let restored_kv_active = restored_index_pos.is_some();
    let mut session_records: Vec<(
        String,
        String,
        Vec<TokenPhysics>,
        Option<Vec<f32>>,
        Option<SecretSauceSegments>,
        Option<SecretSauceVersion>,
        Option<String>,
    )> = Vec::new();
    let mut last_assistant_output: Option<String> = None;
    let mut turn_previous_motif_continuity_artifact = previous_motif_continuity_artifact.clone();
    let index_pos_ptr: *const usize = &index_pos;
    let model_ptr: *const ModelWrapper = &model;
    let phys_engine_ptr: *const PrincipiaEngine = &phys_engine;

    let session_prompt_count = session_prompts.len();
    let chat_output = stdout_profile.chat_enabled();
    let mut run_assistant_turn = |turn_index: usize,
                                  user_prompt: &str,
                                  initial_turn: bool,
                                  previous_assistant: Option<&str>|
     -> Result<(
        String,
        Vec<TokenPhysics>,
        Option<Vec<f32>>,
        Option<SecretSauceSegments>,
        Option<SecretSauceVersion>,
        Option<String>,
    )> {
        let prompt_build_started = timing_now();
        // --reset-kv-cache-per-turn: drop accumulated KV cache between turns so each
        // session-script prompt starts from a fresh model state. Skipped on turn 0
        // (no cache to drop yet). Without this, multi-prompt eval runs silently emit
        // empty strings after the model's context window fills (~prompt 27-30 in
        // agency mode at 4096 GGUF tokens). Must also reset `index_pos` to 0 so
        // rotary positional embeddings + attention masks line up with the empty cache.
        if args.reset_kv_cache_per_turn && turn_index > 0 {
            model.reset_kv_cache()?;
            index_pos = 0;
        }
        // §10bd: stamp current prompt hash on engine so the prompt-hash
        // filter (when enabled) compares against the right value.
        phys_engine.current_prompt_hash = hash_str(user_prompt);
        // §10bd Track 2 v11: reset trajectory-routing state per turn so
        // the entropy classifier evaluates this turn's first N steps.
        phys_engine.reset_trajectory_routing_state();
        // §10bf prompt-level codec activation gate: classify the prompt
        // once at turn-start. When the gate is unconfigured (substrings
        // list empty), this leaves codec_active_for_current_prompt=true.
        phys_engine.apply_codec_prompt_gate(user_prompt);
        // §10ck per-prompt → top-K override gate. When the map is empty
        // this leaves current_prompt_top_k_override=None (legacy fire_top_k).
        phys_engine.apply_correction_packet_prompt_top_k_gate(user_prompt);
        let resolved_output_contract_mode =
            resolve_output_contract_mode(args.output_contract_mode, user_prompt);
        let captured_mistakes =
            if args.mistake_memory_learning && args.mistake_memory_path.is_some() {
                mistake_memory.capture_from_correction_turn(user_prompt, previous_assistant)
            } else {
                Vec::new()
            };
        if !captured_mistakes.is_empty() {
            if let Some(path) = &args.mistake_memory_path {
                mistake_memory.save(path)?;
            }
            eprintln!(
                " [MISTAKE_MEMORY] captured={} turn={} events={}",
                captured_mistakes.len(),
                turn_index,
                mistake_memory.len()
            );
            emit_ui_event_value(
                args.ui_events_json,
                "mistake_memory",
                serde_json::json!({
                    "turn_index": turn_index,
                    "phase": "capture",
                    "captured_count": captured_mistakes.len(),
                    "event_ids": captured_mistakes.iter().map(|event| event.id.as_str()).collect::<Vec<_>>(),
                    "task_keys": captured_mistakes.iter().map(|event| event.task_key.as_str()).collect::<Vec<_>>(),
                }),
            );
        }
        let captured_reflexes = if args.mistake_reflex_learning
            && args.mistake_reflex_path.is_some()
            && args.mistake_reflex_mode != MistakeReflexMode::Off
        {
            mistake_reflex_memory.capture_from_correction_turn(user_prompt, previous_assistant)
        } else {
            Vec::new()
        };
        if !captured_reflexes.is_empty() {
            if let Some(packet_index) = &args.mistake_reflex_packet_index {
                let attached =
                    mistake_reflex_memory.attach_vector_slices_from_packet_index(packet_index)?;
                if attached > 0 {
                    eprintln!(
                        " [MISTAKE_REFLEX] attached_vector_slices={} packet_index={}",
                        attached,
                        packet_index.display()
                    );
                }
            }
            if let Some(path) = &args.mistake_reflex_path {
                mistake_reflex_memory.save(path)?;
            }
            eprintln!(
                " [MISTAKE_REFLEX] captured={} turn={} events={}",
                captured_reflexes.len(),
                turn_index,
                mistake_reflex_memory.len()
            );
            emit_ui_event_value(
                args.ui_events_json,
                "mistake_reflex",
                serde_json::json!({
                    "turn_index": turn_index,
                    "phase": "capture",
                    "captured_count": captured_reflexes.len(),
                    "event_ids": captured_reflexes.iter().map(|event| event.id.as_str()).collect::<Vec<_>>(),
                    "domains": captured_reflexes.iter().map(|event| event.domain.as_str()).collect::<Vec<_>>(),
                    "packet_index": args.mistake_reflex_packet_index.as_ref().map(|path| path.display().to_string()),
                }),
            );
        }
        let mistake_memory_matches = if args.mistake_memory_injection
            && args.mistake_memory_path.is_some()
            && !args.raw_prompt
        {
            mistake_memory.query(user_prompt, 3)
        } else {
            Vec::new()
        };
        let compact_resume_prompt_applied = !args.raw_prompt
            && compact_resume_state_should_inject(
                args.compact_resume_state_injection,
                &compact_resume_state,
                user_prompt,
                initial_turn,
                restored_kv_active,
            );
        let prompt_with_resume_state = if compact_resume_prompt_applied {
            apply_compact_resume_state_prompt(user_prompt, &compact_resume_state)
        } else {
            user_prompt.to_string()
        };
        let agency_state_prompt = if !args.raw_prompt
            && resolved_output_contract_mode == OutputContractMode::CollaborativeTransparency
        {
            agency_hands_state.reinjection_prompt(user_prompt)
        } else {
            None
        };
        let prompt_with_agency_state = if let Some(agency_state_prompt) = &agency_state_prompt {
            format!("{agency_state_prompt}\n\n{}", prompt_with_resume_state)
        } else {
            prompt_with_resume_state
        };
        let mistake_memory_prompt_applied = !mistake_memory_matches.is_empty();
        let prompt_with_mistake_memory = if mistake_memory_prompt_applied {
            MistakeMemory::apply_prompt(prompt_with_agency_state.as_str(), &mistake_memory_matches)
        } else {
            prompt_with_agency_state
        };
        let mistake_reflex_matches = if args.mistake_reflex_mode != MistakeReflexMode::Off
            && args.mistake_reflex_path.is_some()
            && !args.raw_prompt
        {
            mistake_reflex_memory.query(user_prompt, 3)
        } else {
            Vec::new()
        };
        let mistake_reflex_prompt_applied = args.mistake_reflex_mode
            == MistakeReflexMode::Influence
            && args.mistake_reflex_action_mode != MistakeReflexActionMode::HiddenControl
            && !mistake_reflex_matches.is_empty();
        let prompt_with_mistake_reflex = if mistake_reflex_prompt_applied {
            MistakeReflexMemory::apply_prompt(
                prompt_with_mistake_memory.as_str(),
                &mistake_reflex_matches,
                args.mistake_reflex_action_mode.as_str(),
            )
        } else {
            prompt_with_mistake_memory
        };
        let mistake_reflex_prompt_hint_text = if mistake_reflex_prompt_applied {
            prompt_with_mistake_reflex
                .split_once("\n\nUSER TURN:\n")
                .map(|(hint, _)| hint.to_string())
        } else {
            None
        };
        let mistake_reflex_prompt_injection_timing = if mistake_reflex_prompt_applied {
            Some("turn_start_preamble_once".to_string())
        } else {
            None
        };
        let mistake_reflex_prompt_injection_repeated = false;
        let gmms_observe_summaries =
            if args.gmms_observe_turn_start && args.mistake_reflex_path.is_some() {
                mistake_reflex_memory
                    .observe_gmms_applicability(user_prompt, args.gmms_observe_turn_start_limit)
            } else {
                Vec::new()
            };
        let gmms_observe_selected = gmms_observe_summaries.first();
        if args.gmms_observe_turn_start {
            match gmms_observe_turn_start_payload_checked(
                turn_index,
                args.mistake_reflex_path.is_some(),
                args.gmms_observe_turn_start_limit,
                &gmms_observe_summaries,
            ) {
                Ok(payload) => emit_ui_event_value(
                    args.ui_events_json,
                    "gmms_observe_only_applicability",
                    payload,
                ),
                Err(error) => emit_ui_event_value(
                    args.ui_events_json,
                    "gmms_observe_only_applicability_rejected",
                    serde_json::json!({
                        "turn_index": turn_index,
                        "phase": "turn_start",
                        "observe_only": true,
                        "consumer_action": "reject_record",
                        "error": error.to_string(),
                        "runtime_matcher_activation_claimed": false,
                        "mistake_reflex_query_called_for_gmms_observe": false,
                        "prompt_injection_applied": false,
                        "final_answer_text_included": false,
                    }),
                ),
            }
        }
        let output_contract_prompt_applied = !args.raw_prompt
            && resolved_output_contract_mode == OutputContractMode::ExactFormDelivery;
        let runtime_user_prompt = if args.raw_prompt {
            // TUI/external caller already built the full chat template.
            user_prompt.to_string()
        } else if resolved_output_contract_mode == OutputContractMode::CollaborativeTransparency {
            apply_collaborative_transparency_prompt(
                prompt_with_mistake_reflex.as_str(),
                resolved_output_contract_mode,
            )
        } else {
            apply_output_contract_prompt(
                prompt_with_mistake_reflex.as_str(),
                resolved_output_contract_mode,
            )
        };
        if resolved_output_contract_mode == OutputContractMode::ExactFormDelivery {
            if let Some(scaffold) = exact_form_scaffold(user_prompt, &compact_resume_state) {
                let assistant_text = repaired_exact_output(&scaffold);
                let output_contract_violation_reason =
                    output_contract_violation(resolved_output_contract_mode, assistant_text.trim());
                eprintln!(
                    " [OUTPUT_CONTRACT] repair_applied=task_scaffold_fast_path mode={} turn={}",
                    resolved_output_contract_mode.as_str(),
                    turn_index
                );
                emit_ui_event_value(
                    args.ui_events_json,
                    "output_contract",
                    serde_json::json!({
                        "turn_index": turn_index,
                        "configured_mode": args.output_contract_mode.as_str(),
                        "resolved_mode": resolved_output_contract_mode.as_str(),
                        "prompt_applied": output_contract_prompt_applied,
                        "effective_max_steps": 0,
                        "repair_applied": true,
                        "repair_source": "task_scaffold_fast_path",
                        "exact_output_marker_count": exact_output_marker_count(assistant_text.trim()),
                        "violation": output_contract_violation_reason,
                        "exact_block_clean": output_contract_violation_reason.is_none(),
                    }),
                );
                update_compact_resume_state_from_turn(
                    &mut compact_resume_state,
                    user_prompt,
                    assistant_text.trim(),
                    resolved_output_contract_mode,
                );
                emit_ui_event_value(
                    args.ui_events_json,
                    "compact_resume_state",
                    serde_json::json!({
                        "turn_index": turn_index,
                        "turn_count": compact_resume_state.turn_count,
                        "anchor_count": compact_resume_state.anchor_count(),
                        "names": compact_resume_state.names.len(),
                        "constraints": compact_resume_state.constraints.len(),
                        "deadlines": compact_resume_state.deadlines.len(),
                        "preference_flags": compact_resume_state.preference_flags.len(),
                        "unresolved_questions": compact_resume_state.unresolved_questions.len(),
                        "requested_output_shape": compact_resume_state.requested_output_shape.len(),
                        "prior_results": compact_resume_state.prior_results.len(),
                        "corrections": compact_resume_state.corrections.len(),
                    }),
                );
                emit_ui_event_value(
                    args.ui_events_json,
                    "turn_end",
                    serde_json::json!({
                        "turn_index": turn_index,
                        "assistant_text": assistant_text.trim(),
                        "token_count": 0,
                        "secret_sauce_version": serde_json::Value::Null,
                        "resolved_output_contract_mode": resolved_output_contract_mode.as_str(),
                        "output_contract_violation": output_contract_violation_reason,
                    }),
                );
                run_timing.add_prompt_build_ms(elapsed_ms(prompt_build_started));
                run_timing.set_stop_reason("output_contract_fast_path");
                return Ok((assistant_text, Vec::new(), None, None, None, None));
            }
        }
        let chat_template = resolve_chat_template(&args, model.arch());
        let formatted_prompt = if args.raw_prompt {
            runtime_user_prompt
        } else if initial_turn && !restored_kv_active {
            format_initial_chat_prompt(
                chat_template,
                args.qwen_thinking,
                &system_prompt,
                runtime_user_prompt.as_str(),
            )
        } else {
            format_followup_chat_prompt(
                chat_template,
                args.qwen_thinking,
                runtime_user_prompt.as_str(),
            )
        };
        run_timing.add_prompt_build_ms(elapsed_ms(prompt_build_started));
        let empathy_signal =
            empathy_signal_from_turn_context(user_prompt, previous_assistant, restored_kv_active);
        let structured_prompt_signal = structured_reasoning_signal(user_prompt);
        phys_engine.current_turn_structure_bias = structured_prompt_signal;
        phys_engine.current_task_anchor_signature =
            if !compact_resume_state.task_anchor_vector.is_empty() {
                // DEEP_DIVE_ROADMAP P1-C: persist task anchor across compact
                // resumes. When the loaded state has a saved task vector,
                // use it directly so hinge similarity, drift, and routing
                // scores reference the persisted anchor instead of re-hashing
                // the new turn's prompt text. Bypasses the structured-prompt
                // gate — the saved anchor itself is the signal that this
                // turn participates in an ongoing structured task,
                // regardless of whether the new prompt's text scores high
                // on `structured_reasoning_signal`.
                Some(compact_resume_state.task_anchor_vector.clone())
            } else if structured_prompt_signal >= STRUCTURED_REENTRY_PROMPT_THRESHOLD {
                Some(task_anchor_signature(user_prompt))
            } else {
                None
            };
        phys_engine.task_anchor_similarity_start = 0.0;
        phys_engine.task_anchor_similarity_hinge = 0.0;
        phys_engine.task_anchor_similarity_24tok = 0.0;
        phys_engine.task_anchor_drift = 0.0;
        phys_engine.task_anchor_window_tokens_seen = 0;
        phys_engine.first_promotion_attempt_step = None;
        phys_engine.routing_cache = None;
        phys_engine.last_routed_motif_id = None;
        phys_engine.last_routed_motif_role = None;
        phys_engine.last_routed_motif_score = f32::INFINITY;
        phys_engine.last_controller_candidates.clear();
        phys_engine.hinge_window_records.clear();
        phys_engine.structured_resume_window_remaining = 0;
        phys_engine.structured_resume_conversational_hits = 0;
        // Reset per-turn controller/autonomic transients while preserving
        // semantic carry-forward from restored motifs and sentence history.
        phys_engine.request_count = 0;
        phys_engine.last_request_token = 0;
        phys_engine.pending_insight = None;
        phys_engine.insight_persistence = 0;
        phys_engine.focus_lock_remaining_ticks = 0;
        phys_engine.hidden_request_candidate = None;
        phys_engine.hidden_request_streak = 0;
        phys_engine.last_hidden_request = None;
        phys_engine.last_hidden_request_pressure = 0.0;
        phys_engine.hidden_request_activations = 0;
        phys_engine.stress_buffer.clear();
        phys_engine.stress_level = 0.0;
        phys_engine.boredom_level = 0.0;
        phys_engine.adrenaline = 0.0;
        phys_engine.defibrillator_active = false;
        phys_engine.defib_cooldown = 0;
        phys_engine.physics_blend = phys_engine.heartbeat_blend;
        phys_engine.dynamic_gravity = phys_engine.heartbeat_gravity;
        phys_engine.dynamic_repulsion = phys_engine.heartbeat_repulsion;
        if empathy_signal > 0.0 {
            phys_engine.reward_empathy(empathy_signal);
            if phys_engine.stdout_debug() {
                println!(
                    " [EMPATHY] user_signal={:.2} -> empathy_spike={:.2}",
                    empathy_signal, phys_engine.empathy_spike
                );
            }
        }
        if structured_prompt_signal > 0.0 {
            if phys_engine.stdout_debug() {
                println!(
                    " [STRUCTURE] user_signal={:.3} restored={} motif_restore_bias={:.3}",
                    structured_prompt_signal,
                    restored_kv_active,
                    phys_engine.motif_restore_bias_strength
                );
            }
        }
        if structured_prompt_signal >= STRUCTURED_REENTRY_PROMPT_THRESHOLD {
            phys_engine.structured_streak = phys_engine.structured_streak.max(1);
            phys_engine.max_structured_streak = phys_engine
                .max_structured_streak
                .max(phys_engine.structured_streak);
        }
        if !phys_engine.ablate_reentry_clamp
            && restored_kv_active
            && structured_prompt_signal >= STRUCTURED_REENTRY_PROMPT_THRESHOLD
        {
            if let Some((
                anchor,
                motif_kind,
                promotion_status,
                promotion_score,
                structure,
                tightness,
            )) = phys_engine.strongest_structured_reentry_target()
            {
                phys_engine.structured_resume_window_remaining = STRUCTURED_RESUME_LOCK_WINDOW;
                phys_engine.secret_sauce_sentence_prior = Some(anchor.detach());
                phys_engine.motif_restore_bias_steps_remaining = phys_engine
                    .motif_restore_bias_steps_remaining
                    .max((18.0 + structured_prompt_signal * 28.0).round() as usize);
                phys_engine.motif_restore_bias_strength = phys_engine
                    .motif_restore_bias_strength
                    .max((0.18 + structured_prompt_signal * 0.34).clamp(0.18, 0.58));
                phys_engine.apply_reentry_clamp(
                    &motif_kind,
                    &promotion_status,
                    structure.max(structured_prompt_signal),
                    tightness,
                );
                if phys_engine.stdout_debug() {
                    println!(
                        " [STRUCTURED_REENTRY] prompt_signal={:.3} steps={} strength={:.3} clamp_steps={} clamp_strength={:.3} temp_scale={:.3} motif_kind={} status={} score={:.2}",
                        structured_prompt_signal,
                        phys_engine.motif_restore_bias_steps_remaining,
                        phys_engine.motif_restore_bias_strength,
                        phys_engine.reentry_clamp_steps_remaining,
                        phys_engine.reentry_clamp_strength,
                        phys_engine.reentry_temp_scale,
                        motif_kind,
                        promotion_status,
                        promotion_score
                    );
                }
                emit_ui_event_value(
                    args.ui_events_json,
                    "structured_reentry",
                    serde_json::json!({
                        "turn_index": turn_index,
                        "prompt_signal": structured_prompt_signal,
                        "motif_restore_bias_steps_remaining": phys_engine.motif_restore_bias_steps_remaining,
                        "motif_restore_bias_strength": phys_engine.motif_restore_bias_strength,
                        "reentry_clamp_steps_remaining": phys_engine.reentry_clamp_steps_remaining,
                        "reentry_clamp_strength": phys_engine.reentry_clamp_strength,
                        "reentry_temp_scale": phys_engine.reentry_temp_scale,
                        "motif_kind": motif_kind,
                        "promotion_status": promotion_status,
                        "promotion_score": promotion_score,
                    }),
                );
            }
        }

        if phys_engine.stdout_debug() {
            println!("\n=== TURN {} USER ===\n{}", turn_index, user_prompt);
            println!(
                " [OUTPUT_CONTRACT] configured={} resolved={} prompt_applied={} compact_resume_applied={}",
                args.output_contract_mode.as_str(),
                resolved_output_contract_mode.as_str(),
                output_contract_prompt_applied,
                compact_resume_prompt_applied
            );
            println!(" [PROMPT] Formatted Llama 3 Input:\n{}", formatted_prompt);
        }
        let active_context_turn_start = active_context_turn_start_state_for_run.as_ref();
        let active_context_packet_ids = active_context_turn_start
            .map(|state| {
                state
                    .selected_packet_refs
                    .iter()
                    .map(|packet| packet.packet_id.as_str())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let active_context_packet_envelopes = active_context_turn_start
            .map(|state| {
                state
                    .selected_packet_envelopes
                    .iter()
                    .map(|packet| {
                        serde_json::json!({
                            "active_context_id": packet.active_context_id.as_str(),
                            "input_hash": packet.input_hash.as_str(),
                            "source_control_action": packet.source_control_action.as_str(),
                            "runtime_decision": packet.runtime_decision.as_str(),
                            "packet_id": packet.packet_id.as_str(),
                            "task_family": packet.task_family.as_str(),
                            "score": packet.score,
                            "may_steer": packet.may_steer,
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let active_context_observe_only_reload = active_context_turn_start.and_then(|state| {
            let packet_ids = state
                .selected_packet_envelopes
                .iter()
                .map(|packet| packet.packet_id.as_str())
                .collect::<Vec<_>>();
            let metadata = serde_json::json!({
                "active_context_turn_start_record_only": state.turn_start_record_only,
                "active_context_read_only": state.read_only,
                "active_context_prompt_text_injected": state.prompt_text_injected,
                "active_context_final_answer_injected": state.final_answer_injected,
                "active_context_answer_scoring": state.answer_scoring,
                "active_context_runtime_steering_applied": state.runtime_steering_applied,
                "active_context_selected_packet_ref_count": state.selected_packet_envelopes.len(),
                "active_context_selected_packet_ids": packet_ids,
                "active_context_selected_packet_refs": state.selected_packet_envelopes.clone(),
            });
            observe_only_packet_refs_from_metadata_value("live_turn_start_metadata", &metadata).ok()
        });
        let active_context_observe_only_decision_summary = active_context_observe_only_reload
            .as_ref()
            .and_then(|reload| {
                active_context_adapter_decisions_for_run
                    .as_ref()
                    .and_then(|decisions| {
                        observe_only_runtime_decision_summary(
                            "live_turn_start_metadata",
                            reload,
                            decisions,
                        )
                        .ok()
                    })
            });
        let active_context_observe_only_turn_aggregate = active_context_adapter_decisions_for_run
            .as_ref()
            .and_then(|decisions| {
                observe_only_turn_aggregate("live_turn_start_metadata", decisions).ok()
            });
        let active_context_shadow_steering_readiness =
            active_context_turn_start.and_then(|state| {
                active_context_observe_only_turn_aggregate
                    .as_ref()
                    .and_then(|aggregate| {
                        observe_only_shadow_steering_readiness(
                            "live_turn_start_metadata",
                            state,
                            aggregate,
                        )
                        .ok()
                    })
            });
        let compact_resume_reloaded_active_context_shadow_steering_readiness = compact_resume_state
            .active_context_shadow_steering_readiness
            .clone();
        if let Some(readiness) = &active_context_shadow_steering_readiness {
            compact_resume_state.active_context_shadow_steering_readiness = Some(
                compact_resume_active_context_shadow_steering_readiness(readiness),
            );
        }
        emit_ui_event_value(
            args.ui_events_json,
            "turn_start",
            serde_json::json!({
                "turn_index": turn_index,
                "initial_turn": initial_turn,
                "restored_kv_active": restored_kv_active,
                "runtime_mode": args.runtime_mode.as_str(),
                "output_contract_mode": args.output_contract_mode.as_str(),
                "resolved_output_contract_mode": resolved_output_contract_mode.as_str(),
                "output_contract_prompt_applied": output_contract_prompt_applied,
                "compact_resume_prompt_applied": compact_resume_prompt_applied,
                "agency_state_prompt_applied": agency_state_prompt.is_some(),
                "mistake_memory_prompt_applied": mistake_memory_prompt_applied,
                "mistake_memory_match_count": mistake_memory_matches.len(),
                "mistake_memory_event_ids": mistake_memory_matches.iter().map(|item| item.event_id.as_str()).collect::<Vec<_>>(),
                "mistake_reflex_mode": format!("{:?}", args.mistake_reflex_mode).to_ascii_lowercase(),
                "mistake_reflex_action_mode": args.mistake_reflex_action_mode.as_str(),
                "mistake_reflex_prompt_applied": mistake_reflex_prompt_applied,
                "mistake_reflex_match_count": mistake_reflex_matches.len(),
                "mistake_reflex_event_ids": mistake_reflex_matches.iter().map(|item| item.event_id.as_str()).collect::<Vec<_>>(),
                "mistake_reflex_domains": mistake_reflex_matches.iter().map(|item| item.domain.as_str()).collect::<Vec<_>>(),
                "mistake_reflex_resolution_levels": mistake_reflex_matches.iter().map(|item| item.current_resolution_level).collect::<Vec<_>>(),
                "mistake_reflex_unicode_packet_ids": mistake_reflex_matches.iter().filter_map(|item| item.unicode_packet_id.as_deref()).collect::<Vec<_>>(),
                "gmms_observe_turn_start": args.gmms_observe_turn_start,
                "gmms_observe_only": args.gmms_observe_turn_start,
                "gmms_observe_summary_count": gmms_observe_summaries.len(),
                "gmms_observe_selected_slice_id": gmms_observe_selected.map(|item| item.event_id.as_str()),
                "gmms_observe_mode": gmms_observe_selected.map(|item| item.mode.as_str()),
                "gmms_observe_allowed_action_max": gmms_observe_selected.map(|item| item.allowed_action_max.as_str()),
                "gmms_observe_action_level": gmms_observe_selected.map(|item| item.action_level),
                "gmms_observe_score": gmms_observe_selected.map(|item| item.score),
                "gmms_observe_route_unicode_sidecar_attached": gmms_observe_selected.map(|item| item.route_unicode_sidecar_attached),
                "gmms_observe_runtime_matcher_activation_claimed": false,
                "gmms_observe_mistake_reflex_query_called": false,
                "gmms_observe_prompt_injection_applied": false,
                "gmms_observe_final_answer_text_included": false,
                "active_context_turn_start_loaded": active_context_turn_start.is_some(),
                "active_context_surface_id": active_context_turn_start.map(|state| state.surface_id),
                "active_context_adapter_id": active_context_turn_start.map(|state| state.adapter_id),
                "active_context_record_count": active_context_turn_start.map(|state| state.active_context_record_count),
                "active_context_turn_start_record_only": active_context_turn_start.map(|state| state.turn_start_record_only).unwrap_or(false),
                "active_context_read_only": active_context_turn_start.map(|state| state.read_only).unwrap_or(false),
                "active_context_prompt_text_injected": active_context_turn_start.map(|state| state.prompt_text_injected).unwrap_or(false),
                "active_context_final_answer_injected": active_context_turn_start.map(|state| state.final_answer_injected).unwrap_or(false),
                "active_context_answer_scoring": active_context_turn_start.map(|state| state.answer_scoring).unwrap_or(false),
                "active_context_runtime_steering_applied": active_context_turn_start.map(|state| state.runtime_steering_applied).unwrap_or(false),
                "active_context_route_steer_shadow_available": active_context_turn_start.map(|state| state.route_steer_shadow_available).unwrap_or(false),
                "active_context_selected_packet_ref_count": active_context_turn_start.map(|state| state.selected_packet_refs.len()).unwrap_or(0),
                "active_context_selected_packet_ids": active_context_packet_ids,
                "active_context_selected_packet_refs": active_context_packet_envelopes,
                "active_context_observe_only_reload_loaded": active_context_observe_only_reload.as_ref().map(|state| state.loaded).unwrap_or(false),
                "active_context_observe_only_reload_read_only": active_context_observe_only_reload.as_ref().map(|state| state.read_only).unwrap_or(false),
                "active_context_observe_only_reload_packet_ref_count": active_context_observe_only_reload.as_ref().map(|state| state.selected_packet_ref_count).unwrap_or(0),
                "active_context_observe_only_reload_prompt_text_injected": active_context_observe_only_reload.as_ref().map(|state| state.prompt_text_injected).unwrap_or(false),
                "active_context_observe_only_reload_final_answer_injected": active_context_observe_only_reload.as_ref().map(|state| state.final_answer_injected).unwrap_or(false),
                "active_context_observe_only_reload_answer_scoring": active_context_observe_only_reload.as_ref().map(|state| state.answer_scoring).unwrap_or(false),
                "active_context_observe_only_reload_runtime_steering_applied": active_context_observe_only_reload.as_ref().map(|state| state.runtime_steering_applied).unwrap_or(false),
                "active_context_observe_only_decision_summary_loaded": active_context_observe_only_decision_summary.is_some(),
                "active_context_observe_only_decision_summary_surface_id": active_context_observe_only_decision_summary.as_ref().map(|state| state.surface_id),
                "active_context_observe_only_decision_summary_matched_packet_ref_count": active_context_observe_only_decision_summary.as_ref().map(|state| state.matched_packet_ref_count).unwrap_or(0),
                "active_context_observe_only_decision_summary_unmatched_packet_ref_count": active_context_observe_only_decision_summary.as_ref().map(|state| state.unmatched_packet_ref_count).unwrap_or(0),
                "active_context_observe_only_decision_summary_matched_decision_count": active_context_observe_only_decision_summary.as_ref().map(|state| state.matched_decision_count).unwrap_or(0),
                "active_context_observe_only_decision_summary_route_steer_shadow_decision_count": active_context_observe_only_decision_summary.as_ref().map(|state| state.route_steer_shadow_decision_count).unwrap_or(0),
                "active_context_observe_only_decision_summary_all_refs_matched": active_context_observe_only_decision_summary.as_ref().map(|state| state.all_reloaded_refs_matched).unwrap_or(false),
                "active_context_observe_only_decision_summary_read_only": active_context_observe_only_decision_summary.as_ref().map(|state| state.read_only).unwrap_or(false),
                "active_context_observe_only_decision_summary_prompt_text_injected": active_context_observe_only_decision_summary.as_ref().map(|state| state.prompt_text_injected).unwrap_or(false),
                "active_context_observe_only_decision_summary_final_answer_injected": active_context_observe_only_decision_summary.as_ref().map(|state| state.final_answer_injected).unwrap_or(false),
                "active_context_observe_only_decision_summary_answer_scoring": active_context_observe_only_decision_summary.as_ref().map(|state| state.answer_scoring).unwrap_or(false),
                "active_context_observe_only_decision_summary_runtime_steering_applied": active_context_observe_only_decision_summary.as_ref().map(|state| state.runtime_steering_applied).unwrap_or(false),
                "active_context_observe_only_decision_summary_reason_codes": active_context_observe_only_decision_summary.as_ref().map(|state| state.reason_codes.clone()).unwrap_or_default(),
                "active_context_observe_only_turn_aggregate_loaded": active_context_observe_only_turn_aggregate.is_some(),
                "active_context_observe_only_turn_aggregate_surface_id": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.surface_id),
                "active_context_observe_only_turn_aggregate_matched_decision_count": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.matched_decision_count).unwrap_or(0),
                "active_context_observe_only_turn_aggregate_recommended_action_counts": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.recommended_action_counts.clone()).unwrap_or_default(),
                "active_context_observe_only_turn_aggregate_runtime_decision_counts": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.runtime_decision_counts.clone()).unwrap_or_default(),
                "active_context_observe_only_turn_aggregate_top_reason_code_families": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.top_reason_code_families.clone()).unwrap_or_default(),
                "active_context_observe_only_turn_aggregate_read_only": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.read_only).unwrap_or(false),
                "active_context_observe_only_turn_aggregate_prompt_text_injected": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.prompt_text_injected).unwrap_or(false),
                "active_context_observe_only_turn_aggregate_final_answer_injected": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.final_answer_injected).unwrap_or(false),
                "active_context_observe_only_turn_aggregate_answer_scoring": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.answer_scoring).unwrap_or(false),
                "active_context_observe_only_turn_aggregate_runtime_steering_applied": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.runtime_steering_applied).unwrap_or(false),
                "active_context_observe_only_turn_aggregate_reason_codes": active_context_observe_only_turn_aggregate.as_ref().map(|state| state.reason_codes.clone()).unwrap_or_default(),
                "active_context_shadow_steering_readiness_loaded": active_context_shadow_steering_readiness.is_some(),
                "active_context_shadow_steering_readiness_surface_id": active_context_shadow_steering_readiness.as_ref().map(|state| state.surface_id),
                "active_context_shadow_steering_ready": active_context_shadow_steering_readiness.as_ref().map(|state| state.shadow_steering_ready).unwrap_or(false),
                "active_context_shadow_steering_readiness_selected_packet_ref_count": active_context_shadow_steering_readiness.as_ref().map(|state| state.selected_packet_ref_count).unwrap_or(0),
                "active_context_shadow_steering_readiness_route_steer_shadow_decision_count": active_context_shadow_steering_readiness.as_ref().map(|state| state.route_steer_shadow_decision_count).unwrap_or(0),
                "active_context_shadow_steering_readiness_recommended_steer_count": active_context_shadow_steering_readiness.as_ref().map(|state| state.recommended_steer_count).unwrap_or(0),
                "active_context_shadow_steering_readiness_safety_gate_count": active_context_shadow_steering_readiness.as_ref().map(|state| state.safety_gate_count).unwrap_or(0),
                "active_context_shadow_steering_readiness_failed_gate_count": active_context_shadow_steering_readiness.as_ref().map(|state| state.failed_gate_count).unwrap_or(0),
                "active_context_shadow_steering_readiness_read_only": active_context_shadow_steering_readiness.as_ref().map(|state| state.read_only).unwrap_or(false),
                "active_context_shadow_steering_readiness_prompt_text_injected": active_context_shadow_steering_readiness.as_ref().map(|state| state.prompt_text_injected).unwrap_or(false),
                "active_context_shadow_steering_readiness_final_answer_injected": active_context_shadow_steering_readiness.as_ref().map(|state| state.final_answer_injected).unwrap_or(false),
                "active_context_shadow_steering_readiness_answer_scoring": active_context_shadow_steering_readiness.as_ref().map(|state| state.answer_scoring).unwrap_or(false),
                "active_context_shadow_steering_readiness_runtime_steering_applied": active_context_shadow_steering_readiness.as_ref().map(|state| state.runtime_steering_applied).unwrap_or(false),
                "active_context_shadow_steering_readiness_reason_codes": active_context_shadow_steering_readiness.as_ref().map(|state| state.reason_codes.clone()).unwrap_or_default(),
                "compact_resume_active_context_shadow_steering_readiness_loaded": compact_resume_reloaded_active_context_shadow_steering_readiness.is_some(),
                "compact_resume_active_context_shadow_steering_ready": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.shadow_steering_ready).unwrap_or(false),
                "compact_resume_active_context_shadow_steering_readiness_selected_packet_ref_count": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.selected_packet_ref_count).unwrap_or(0),
                "compact_resume_active_context_shadow_steering_readiness_failed_gate_count": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.failed_gate_count).unwrap_or(0),
                "compact_resume_active_context_shadow_steering_readiness_read_only": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.read_only).unwrap_or(false),
                "compact_resume_active_context_shadow_steering_readiness_prompt_text_injected": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.prompt_text_injected).unwrap_or(false),
                "compact_resume_active_context_shadow_steering_readiness_final_answer_injected": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.final_answer_injected).unwrap_or(false),
                "compact_resume_active_context_shadow_steering_readiness_answer_scoring": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.answer_scoring).unwrap_or(false),
                "compact_resume_active_context_shadow_steering_readiness_runtime_steering_applied": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.runtime_steering_applied).unwrap_or(false),
                "compact_resume_active_context_shadow_steering_readiness_reason_codes": compact_resume_reloaded_active_context_shadow_steering_readiness.as_ref().map(|state| state.reason_codes.clone()).unwrap_or_default(),
                "compact_resume_anchor_count": compact_resume_state.anchor_count(),
                "agency_remember_count": agency_hands_state.remembers.len(),
                "agency_active_lock": agency_hands_state.active_lock.as_deref(),
                "user_prompt": user_prompt,
                "structured_prompt_signal": structured_prompt_signal,
                "empathy_signal": empathy_signal,
            }),
        );

        let tokens = model
            .tokenizer()
            .encode(formatted_prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();
        if initial_turn
            && args.secret_sauce_kv_prefix
            && phys_engine.secret_sauce_version == Some(SecretSauceVersion::V3)
            && args.secret_sauce_kv_prefix_len > 0
        {
            if let Some(anchor) = &phys_engine.secret_sauce_sentence_prior {
                let prefix_ids = build_secret_sauce_warm_start_tokens(
                    model.tokenizer(),
                    &phys_engine.kv_prefix_charge_tensor,
                    anchor,
                    &phys_engine.control_token_ids,
                    args.secret_sauce_kv_prefix_len,
                )?;
                if !prefix_ids.is_empty() {
                    let prefix_surfaces = prefix_ids
                        .iter()
                        .map(|id| model.tokenizer().decode(&[*id], true).unwrap_or_default())
                        .collect::<Vec<_>>();
                    println!(
                        " [KV_PREFIX] version=v3 len={} ids={:?} surfaces={:?}",
                        prefix_ids.len(),
                        prefix_ids,
                        prefix_surfaces
                    );
                    let prefix_input = Tensor::new(prefix_ids.as_slice(), &device)?.unsqueeze(0)?;
                    let mut noop_physics = NoopPhysicsEngine;
                    let dummy_ghost = Tensor::zeros((hidden_dim,), DType::F32, &device)?;
                    let _ = model.forward_physics(
                        &prefix_input,
                        index_pos,
                        &mut noop_physics,
                        Some(&dummy_ghost),
                    )?;
                    index_pos += prefix_ids.len();
                    phys_engine.secret_sauce_sentence_prior = None;
                    phys_engine.secret_sauce_steps_remaining = 0;
                }
            }
        }
        let mut input = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;

        let mut cognitive_log: Vec<TokenPhysics> = Vec::new();
        let mut assistant_text = String::new();
        let mut final_hidden_capture: Option<Vec<f32>> = None;
        let mut final_secret_sauce_segments: Option<SecretSauceSegments> = None;
        let mut final_secret_sauce_version: Option<SecretSauceVersion> = None;
        let mut final_secret_sauce: Option<String> = None;
        let mut reflex_preserve_byte_len: Option<usize> = None;
        let mut mistake_memory_preserve_byte_len: Option<usize> = None;
        let mut pending_vault_remember: Option<([f32; 64], String, usize)> = None;
        let mut metric_audit = RuntimeMetricAudit::default();
        let mut finalizer = FinalizationController::new(
            args.lock_stop_policy,
            args.lock_taper_tokens,
            args.lock_stop_on_final_answer,
        );
        let mut answer_boundary_finalizer =
            AnswerBoundaryFinalizer::from_prompt(args.answer_boundary_finalization, user_prompt);
        let arithmetic_expectation = detect_answer_boundary_expectation(user_prompt)
            .filter(|expectation| expectation.kind == AnswerBoundaryKind::Arithmetic);
        let math_governor_relief_active =
            args.math_governor_relief && arithmetic_expectation.is_some();
        if math_governor_relief_active {
            phys_engine.braking = true;
            phys_engine.physics_blend = 0.0;
            phys_engine.dynamic_repulsion = 0.0;
            phys_engine.adrenaline = 0.0;
            phys_engine.defibrillator_active = false;
            phys_engine.focus_lock_remaining_ticks = 0;
            emit_ui_event_value(
                args.ui_events_json,
                "governor_relief",
                serde_json::json!({
                    "turn_index": turn_index,
                    "active": true,
                    "reason": "arithmetic_prompt",
                    "expected_answer": arithmetic_expectation
                        .as_ref()
                        .map(|expectation| expectation.expected_answer.clone()),
                    "source_term": arithmetic_expectation
                        .as_ref()
                        .map(|expectation| expectation.source_term.clone()),
                    "physics_braked": true,
                    "guardrail_resampling_disabled": true,
                }),
            );
        } else {
            phys_engine.braking = false;
        }
        let mut mistake_guard = MistakeMemoryGuard::new(mistake_memory_matches.clone());
        let mistake_memory_surface_suppression_terms =
            MistakeMemory::surface_suppression_terms(&mistake_memory_matches);
        let mistake_memory_surface_suppression_token_ids: HashSet<u32> =
            if mistake_memory_surface_suppression_terms.is_empty() {
                HashSet::new()
            } else {
                let mut ids = HashSet::new();
                for surface in &mistake_memory_surface_suppression_terms {
                    for variant in [
                        surface.to_string(),
                        format!(" {surface}"),
                        format!("\n{surface}"),
                    ] {
                        if let Ok(encoding) = model.tokenizer().encode(variant, false) {
                            ids.extend(encoding.get_ids().iter().copied());
                        }
                    }
                }
                ids
            };
        let mut mistake_reflex_guard = MistakeReflexGuard::new(mistake_reflex_matches.clone());
        let mut mistake_reflex_retry_count = 0usize;
        let mut mistake_reflex_retry_tokens_remaining = 0usize;
        let mut mistake_reflex_retry_reason: Option<String> = None;
        let mut mistake_reflex_forced_retry_tokens: VecDeque<u32> = VecDeque::new();
        let count_route_memory_finalization_candidate = if args
            .count_route_memory_finalization_candidate_telemetry
            || args.count_route_memory_finalization_replacement_action
            || args.count_route_memory_finalization_natural_v2_replacement_action
            || args.count_route_memory_finalization_enumeration_aggregation_action
            || args.count_route_memory_finalization_enumeration_preserve_stop
            || args.count_route_memory_finalization_protected_lock_surface
        {
            parse_count_route_memory_finalization_candidate(user_prompt)
        } else {
            None
        };

        let configured_max_steps = turn_configured_max_steps(&args, initial_turn);
        if configured_max_steps != args.max_steps {
            println!(
                " [TURN_STEPS] turn={} base={} configured={}",
                turn_index, args.max_steps, configured_max_steps
            );
        }
        let effective_max_steps =
            exact_form_effective_max_steps(configured_max_steps, resolved_output_contract_mode);
        if effective_max_steps != configured_max_steps {
            println!(
                " [OUTPUT_CONTRACT] exact_form_budget configured={} effective={}",
                configured_max_steps, effective_max_steps
            );
        }

        let decode_started = timing_now();
        let mut turn_stop_reason = "max_steps".to_string();
        // Hoisted out of the per-step decode loop: integration constants don't change
        // across decode steps. Was allocating two 1-elem GPU tensors per token.
        let nbody_dt_t = Tensor::new(args.dt, &device)?;
        let nbody_friction_t = Tensor::new(0.95f32, &device)?;
        for step in 0..effective_max_steps {
            phys_engine.current_step = step;
            if math_governor_relief_active {
                phys_engine.braking = true;
                phys_engine.physics_blend = 0.0;
                phys_engine.dynamic_repulsion = 0.0;
                phys_engine.adrenaline = 0.0;
                phys_engine.defibrillator_active = false;
                phys_engine.last_guardrail_active = false;
            }
            phys_engine.specialist_memory_worker_answer_window_active =
                specialist_worker_answer_window_active(&assistant_text);
            phys_engine.specialist_memory_worker_pre_answer_active =
                specialist_worker_pre_answer_active(&assistant_text);
            phys_engine.specialist_memory_worker_pre_earned_active =
                !mistake_reflex_guard.snapshot().earned_answer_seen;
            phys_engine.specialist_memory_worker_at_boundary_active = phys_engine
                .specialist_memory_worker_was_pre_answer_active
                && phys_engine.specialist_memory_worker_answer_window_active;
            phys_engine.specialist_memory_worker_was_pre_answer_active =
                phys_engine.specialist_memory_worker_pre_answer_active;
            let mut finalization_stop_reason: Option<String> = None;
            if phys_engine.stdout_debug() {
                println!("--- Step {} ---", step);
                std::io::stdout().flush().unwrap();
            }

            // NIODOO PROTOCOL GHOST CALCULATION
            // 1. Get embedding of current input for "Query"
            let input_embed = model.embed_tokens_forward(&input)?.to_dtype(DType::F32)?;
            let input_query: Tensor = if input_embed.rank() == 3 {
                input_embed.i((0, 0))?
            } else {
                input_embed.flatten_all()?
            };

            if !phys_engine.eval_fast() {
                phys_engine.refresh_live_hidden_bridge_vectors()?;
            }

            // Make the current turn's mistake-reflex retry count visible to apply_forces
            // so the multi-source relapse trigger (§10ae) can fire when the model just
            // had to retry. mistake_reflex_retry_count starts 0 each user-prompt cycle
            // and increments on each retry; what apply_forces sees is "how many retries
            // have already happened this cycle when this generation began."
            phys_engine.last_mistake_reflex_retry_count = mistake_reflex_retry_count;

            let ghost_vector = if phys_engine.sentence_history.len() > 0
                || phys_engine.goal_embedding.is_some()
                || phys_engine.vad_head.is_some()
            {
                phys_engine.compute_ghost_vector(&input_query, &device)?
            } else {
                phys_engine.clear_ghost_pressure_telemetry();
                None
            };

            let seq_len = input.dim(1)?;

            let forward_started = timing_now();
            let (logits, current_hidden_raw) = model.forward_physics(
                &input,
                index_pos,
                &mut phys_engine,
                ghost_vector.as_ref(),
            )?;
            let forward_ms = elapsed_ms(forward_started);
            if step == 0 {
                run_timing.add_prefill_ms(forward_ms);
            }

            // === HIDDEN STATE CAPTURE FOR ENCODING TESTING ===
            // Save hidden state to file for encoding/decoding testing
            if !phys_engine.eval_fast() && (step == 0 || step % 20 == 0) {
                let hidden_f32 = current_hidden_raw.to_dtype(DType::F32)?;
                let hidden_vec = hidden_f32.flatten_all()?.to_vec1::<f32>()?;

                // Create req_id-specific directory for hidden states (parallel-safe).
                let req_id_safe: String = args
                    .req_id
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                            c
                        } else {
                            '_'
                        }
                    })
                    .collect();
                let hidden_dir = format!("/tmp/niodoo_hidden_states_{}", req_id_safe);
                std::fs::create_dir_all(&hidden_dir)?;

                // Save as numpy-compatible binary
                let hidden_path = format!("{}/hidden_state_step_{:04}.bin", hidden_dir, step);
                std::fs::write(&hidden_path, bytemuck::cast_slice(&hidden_vec))?;

                // Save sidecar metadata
                let norm = hidden_f32.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                let sidecar = serde_json::json!({
                    "step": step,
                    "norm": norm,
                    "shape": current_hidden_raw.dims(),
                    "drift_score": phys_engine.calculate_drift(&current_hidden_raw).unwrap_or(0.0),
                    "dynamic_gravity": phys_engine.dynamic_gravity,
                    "dynamic_repulsion": phys_engine.dynamic_repulsion,
                    "ghost_vector_present": ghost_vector.is_some(),
                    "basin_pressure": phys_engine.last_live_basin_pressure,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                let sidecar_path = format!("{}/hidden_state_step_{:04}.json", hidden_dir, step);
                std::fs::write(&sidecar_path, serde_json::to_string_pretty(&sidecar)?)?;

                if phys_engine.stdout_debug() {
                    println!(
                        "[HIDDEN_STATE] Saved hidden state to {} (shape: {:?}, norm: {:.2})",
                        hidden_path,
                        current_hidden_raw.shape(),
                        norm
                    );
                }
            }

            let current_hidden = current_hidden_raw.to_dtype(DType::F32)?;
            let live_hidden_1d = if current_hidden.rank() == 3 {
                current_hidden
                    .i((.., current_hidden.dim(1)? - 1, ..))?
                    .squeeze(0)?
                    .flatten_all()?
            } else {
                current_hidden.flatten_all()?
            };
            let logits_f32 = logits.to_dtype(DType::F32)?;

            let drift_score = phys_engine.calculate_drift(&current_hidden).unwrap_or(0.0);
            let base_gravity = args.gravity_well;
            let elastic_gravity = base_gravity * (1.0 + (drift_score * 4.0).powf(2.0));
            let base_repulsion = args.repulsion_strength as f32;
            let elastic_repulsion = if drift_score < 0.15 {
                base_repulsion * 2.0
            } else {
                base_repulsion
            };
            phys_engine.dynamic_gravity = elastic_gravity;
            phys_engine.dynamic_repulsion = if math_governor_relief_active {
                0.0
            } else {
                elastic_repulsion
            };

            if phys_engine.stdout_debug() && step % 10 == 0 {
                println!(
                    " [ELASTIC v3] Drift: {:.3} | G: {:.3} | R: {:.3}",
                    drift_score, elastic_gravity, elastic_repulsion
                );
            }

            let mut latent = if logits_f32.rank() >= 2 {
                logits_f32.mean(0)?
            } else {
                logits.clone()
            };
            if let Some(deepmd) = &phys_engine.deepmd_kit {
                latent = deepmd.atomic_mean(&latent)?;
            }

            if args.mind_state_every > 0 && step % args.mind_state_every == 0 {
                let _ = phys_engine.print_mind_state(step, &current_hidden, 8);
            }

            if phys_engine.sentence_history.len() > 1 {
                // BARNES-HUT: Center of Mass Aggregation - preserves topology, reduces O(n²) to O(n)
                const RECENT_COUNT: usize = 10;
                const AGGREGATE_MASS: f32 = 50.0;

                if phys_engine.sentence_history.len() > RECENT_COUNT + 1 {
                    let history_len = phys_engine.sentence_history.len();
                    let aggregate_end = history_len.saturating_sub(RECENT_COUNT);

                    let mut com_pos: Option<Tensor> = None;
                    let mut com_mass = 0.0f32;

                    for i in 0..aggregate_end {
                        if let Some(p) = phys_engine.sentence_history.get(i) {
                            let mass = (p.m_info
                                + p.m_sem
                                + p.m_coh
                                + p.m_struct
                                + p.m_quantum
                                + p.m_geometric
                                + p.m_emo)
                                .max(0.1);

                            if let Some(pos) = &com_pos {
                                let new_mass = com_mass + mass;
                                // Use Tensor::affine (scalar mul on GPU without allocating
                                // a 1-elem tensor each call). Saves 3 GPU allocations per
                                // particle per token in the COM aggregation loop.
                                let weighted_pos = pos.affine(com_mass as f64, 0.0)?;
                                let p_weighted = p.position.affine(mass as f64, 0.0)?;
                                let sum_pos = (weighted_pos + p_weighted)?;
                                com_pos = Some(sum_pos.affine(1.0 / new_mass as f64, 0.0)?);
                                com_mass = new_mass;
                            } else {
                                com_pos = Some(p.position.clone());
                                com_mass = mass;
                            }
                        }
                    }

                    if let Some(anchor_pos) = com_pos {
                        if com_mass > 0.0 {
                            for _ in 0..aggregate_end {
                                phys_engine.sentence_history.pop_front();
                            }

                            if let Some(last_p) = phys_engine.sentence_history.back() {
                                // Get owned clone by re-borrowing and manually cloning the relevant fields
                                let last_vel = last_p.velocity.clone();
                                let anchor_particle = SentenceParticle {
                                    position: anchor_pos,
                                    velocity: last_vel,
                                    mass: AGGREGATE_MASS,
                                    radius: last_p.radius,
                                    birth_step: last_p.birth_step,
                                    token_count: last_p.token_count,
                                    vad: last_p.vad,
                                    surprisal: last_p.surprisal,
                                    delta: last_p.delta,
                                    m_info: AGGREGATE_MASS,
                                    m_sem: AGGREGATE_MASS * 0.5,
                                    m_coh: AGGREGATE_MASS * 0.3,
                                    m_struct: last_p.m_struct,
                                    m_quantum: last_p.m_quantum,
                                    m_geometric: last_p.m_geometric,
                                    m_emo: last_p.m_emo,
                                    kl_delta: last_p.kl_delta,
                                    text: last_p.text.clone(),
                                    entangled_with: BTreeMap::new(),
                                    quantum_state: last_p.quantum_state.clone(),
                                    fitness: last_p.fitness,
                                    latent_thought: None,
                                    sub_particles: Vec::new(),
                                    is_lpm_active: false,
                                    is_attractor: false,
                                    is_repulsor: false,
                                };
                                phys_engine.sentence_history.push_back(anchor_particle);
                            }
                        }
                    }
                }

                if let Some(lpm) = phys_engine.lpm_collaborator.clone() {
                    lpm.simulate_quantum_step(&mut phys_engine)?;
                }

                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(" [DBG] Len > 10. Starting pos_vec");
                }
                let pos_vec: Vec<Tensor> = phys_engine
                    .sentence_history
                    .iter()
                    .map(|p| {
                        if !p.entangled_with.is_empty() {
                            phys_engine
                                .update_entangled_state(p)
                                .unwrap_or(p.position.clone())
                        } else {
                            p.position.clone()
                        }
                    })
                    .collect();

                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(" [DBG] Step 1: pos_vec collected. Len: {}", pos_vec.len());
                }
                let mut processed_pos = Vec::new();
                if let Some(gdl) = &phys_engine.geometric_dl {
                    for p in pos_vec {
                        processed_pos.push(gdl.process_mesh(&p)?);
                    }
                } else {
                    processed_pos = pos_vec;
                }
                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(" [DBG] Step 2: GDL done");
                }
                // Use stack to combine [hidden] vectors into [N, hidden] matrix
                let all_pos = Tensor::stack(&processed_pos, 0)?;
                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(
                        " [DBG] Step 3: all_pos stack done. shape: {:?}",
                        all_pos.dims()
                    );
                }

                let mass_vec: Vec<f32> = phys_engine
                    .sentence_history
                    .iter()
                    .map(|p| {
                        phys_engine.compute_total_mass(
                            p.m_info,
                            p.m_sem,
                            p.m_coh,
                            p.m_struct,
                            p.m_quantum,
                            p.m_geometric,
                            p.m_emo,
                            p.kl_delta,
                        )
                    })
                    .collect();

                let all_mass =
                    Tensor::from_vec(mass_vec, (phys_engine.sentence_history.len(), 1), &device)?;
                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(" [DBG] Step 4: all_mass done");
                }

                let vel_vec: Vec<Tensor> = phys_engine
                    .sentence_history
                    .iter()
                    .map(|p| p.velocity.clone())
                    .collect();
                // Use stack to combine [hidden] velocity vectors into [N, hidden] matrix
                let all_vel = Tensor::stack(&vel_vec, 0)?;
                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(" [DBG] Step 5: all_vel stack done");
                }

                // Physics Calculation — N-body pairwise forces.
                //
                // Previously this materialized [N, N, hidden_dim] intermediates via
                // candle broadcasts (~250M floats at N=200, hidden=4096 ≈ 1 GB).
                // Replaced with a fused CUDA kernel that streams pairs in shared memory
                // and writes accel[N, hidden_dim] directly. See niodoo/src/gpu/nbody.rs
                // and the `nbody_pairwise_accel` kernel in niodoo/src/kernels.cu.
                //
                // Math: accel[i][k] = G * sum_{j != i} m_j * (pos[j][k] - pos[i][k]) / dist^3.
                let g_val = phys_engine.params.gravity as f32;
                let softening = 1e-6f32;

                // all_mass is [N, 1] from the Tensor::from_vec above; kernel wants [N].
                let mass_1d = all_mass.squeeze(1)?;

                let accel =
                    niodoo::gpu::nbody_pairwise_accel(&all_pos, &mass_1d, g_val, softening)
                        .map_err(|e| {
                            candle_core::Error::Msg(format!("nbody_pairwise_accel: {e}"))
                        })?;

                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(
                        " [DBG] Steps 6-10 (fused nbody kernel): accel done. shape: {:?}",
                        accel.dims()
                    );
                }
                // Use the hoisted constants from before the decode loop —
                // dt and friction don't change across steps.
                let term1 = all_vel.broadcast_mul(&nbody_friction_t)?;
                let term2 = accel.broadcast_mul(&nbody_dt_t)?;
                let new_vel = (term1 + term2)?;

                let pos_delta = new_vel.broadcast_mul(&nbody_dt_t)?;
                let new_pos = (all_pos + pos_delta)?;
                if phys_engine.stdout_debug() && phys_engine.sentence_history.len() > 10 {
                    println!(" [DBG] Step 11: integration done");
                }

                for i in 0..phys_engine.sentence_history.len() {
                    phys_engine.sentence_history[i].velocity = new_vel.i(i)?.detach();
                    phys_engine.sentence_history[i].position = new_pos.i(i)?.detach();
                }

                if phys_engine.stdout_debug() && step % 50 == 0 {
                    println!(
                        " ⚛ PARTICLES EVOLVED: {} particles",
                        phys_engine.sentence_history.len()
                    );
                }
            }

            // === PHASE 2: ORBITAL STEERING ===
            // === PHASE 2: ORBITAL STEERING ===
            // FIX: Activate immediately (len >= 1) so prompt counts as the first "Sun"
            if args.mode_orbital && !phys_engine.sentence_history.is_empty() {
                phys_engine.orbital_active = true;

                // A. Calculate "Sun" (Context Centroid)
                // Average the embeddings of the last N tokens (particles)
                // NOTE: We rely on sentence_history which stores SentenceParticle
                let history_len = phys_engine.sentence_history.len();
                let lookback = 20.min(history_len);

                // We need to retrieve the vectors. SentenceParticle stores `position` tensor.
                let mut center_of_mass_t = Tensor::zeros((hidden_dim,), DType::F32, &device)?;
                let mut count = 0.0f32;

                for i in 0..lookback {
                    let idx = history_len - 1 - i;
                    if let Some(p) = phys_engine.sentence_history.get(idx) {
                        // Ensure on device and flat
                        let pos = p
                            .position
                            .to_device(&device)?
                            .to_dtype(DType::F32)?
                            .flatten_all()?;
                        center_of_mass_t = (center_of_mass_t + pos)?;
                        count += 1.0;
                    }
                }

                // PHASE 2.1: SUN ANCHOR
                // Add the initial prompt embedding to the mass to anchor the orbit.
                // We use `input_query` which is the prompt input in the first step,
                // but we need the ORIGINAL prompt.
                // Hack: We'll assume the FIRST particle in history is the prompt anchor or close enough.
                if let Some(first_particle) = phys_engine.sentence_history.front() {
                    let anchor_pos = first_particle
                        .position
                        .to_device(&device)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?;

                    // Weight the anchor heavily (e.g., 5x mass) to prevent drift
                    // Weight the anchor heavily (e.g., 5x mass) to prevent drift
                    let anchor_weight = args.gravity_well;
                    let weighted_anchor =
                        anchor_pos.broadcast_mul(&Tensor::new(anchor_weight, &device)?)?;
                    center_of_mass_t = (center_of_mass_t + weighted_anchor)?;
                    count += anchor_weight;
                }

                if count > 0.0 {
                    let scale = 1.0 / count;
                    let scale_t = Tensor::new(scale, &device)?;
                    center_of_mass_t = center_of_mass_t.broadcast_mul(&scale_t)?;
                }

                // B. Get Current Position (Last generated token / particle)
                // FIX: Use current LIVE sentence embeddings if available, otherwise last particle
                let current_pos_t = if !phys_engine.current_sentence_embeddings.is_empty() {
                    let stack = Tensor::stack(&phys_engine.current_sentence_embeddings, 0)?;
                    let mean = stack.mean(0)?;
                    // Normalize to keep scale consistent with particles
                    let norm = mean.sqr()?.sum_all()?.sqrt()?;
                    mean.broadcast_div(&norm)?
                } else if let Some(last) = phys_engine.sentence_history.back() {
                    last.position
                        .to_device(&device)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                } else {
                    Tensor::zeros((hidden_dim,), DType::F32, &device)?
                };

                // C. ORBITAL MATH (Tensor-based for speed)
                // Gravity Vector: Sun - Current
                let gravity_t = (&center_of_mass_t - &current_pos_t)?;

                // Tangential Constraint (Gram-Schmidt)
                // momentum is Vec<f32>, convert to Tensor
                let mut momentum_t =
                    Tensor::from_vec(phys_engine.momentum.clone(), (hidden_dim,), &device)?;

                // Project
                let g_sq = gravity_t.sqr()?.sum_all()?;
                let g_dot = (gravity_t.clone() * momentum_t.clone())?.sum_all()?;

                // Avoid div by zero
                let g_sq_scalar = g_sq.to_scalar::<f32>()?;
                if g_sq_scalar > 1e-6 {
                    let proj_scalar = g_dot.to_scalar::<f32>()? / g_sq_scalar;
                    let proj_scalar_t = Tensor::new(proj_scalar, &device)?;
                    let radial_comp = gravity_t.broadcast_mul(&proj_scalar_t)?;
                    momentum_t = (momentum_t - radial_comp)?;
                }

                // Symplectic Kick: Normalize and Scale
                // momentum = momentum.normalize() * ORBIT_SPEED
                let m_sq = momentum_t.sqr()?.sum_all()?.to_scalar::<f32>()?;
                let mut kick = momentum_t.clone();
                if m_sq > 1e-9 {
                    let scale = args.orbit_speed / m_sq.sqrt();
                    let scale_t = Tensor::new(scale, &device)?;
                    kick = kick.broadcast_mul(&scale_t)?;
                }

                // Update Momentum State
                // We just add a fraction of this "ideal orbit" to the persistent momentum?
                // Or typically Symplectic is: p_new = p_old + Force * dt.
                // Here we are synthesizing the p_new directly from the geometry.
                // Let's adopt the geometric momentum directly as the steering force.
                phys_engine.momentum = kick.to_vec1::<f32>()?;

                // D. PROJECT TARGET POSITION (The "Carrot")
                // Where does the orbit want us to go next?
                let target_pos_t = (&current_pos_t + &kick)?;

                // E. UPDATE ENGINE GOAL
                // We set the "Goal Embedding" to this orbital target.
                // The PrincipiaEngine will treat this as a gravity well in the NEXT forward pass.
                phys_engine.goal_embedding = Some(target_pos_t.detach());
            }

            let mut next_token_id: u32;
            let mut sampling_snapshot_for_probe: Option<SamplingLogitSnapshot> = None;
            let mut sampling_branch = "forced".to_string();
            let mut logits_filtered = logits_f32.clone();
            let debris_guard_active = phys_engine.last_absence_signal > 0.85
                || phys_engine.last_trap_score > 0.9
                || phys_engine.defibrillator_active;
            let collapse_guard_active = phys_engine.last_recovery_mag > 1.0
                && phys_engine.last_absence_signal > 0.65
                && phys_engine.last_trap_score > 0.6;
            phys_engine.last_guardrail_active =
                !math_governor_relief_active && (debris_guard_active || collapse_guard_active);

            let garbage_patterns = [
                "://",
                ".Forms",
                "_REF",
                "php",
                ".swing",
                "http",
                "www",
                ".com",
                "html",
                "function(",
                "return ",
                "var ",
                "const ",
                "async ",
                "Angeles",
                "assistant",
            ];
            let hidden_request_signal = if phys_engine.hidden_request_inference
                && phys_engine.runtime_mode.is_agency()
                && !visible_control_surface_active(
                    &phys_engine.request_buffer,
                    &phys_engine.surface_buffer,
                ) {
                inspect_hidden_request_signal(
                    &logits_f32,
                    model.tokenizer(),
                    &phys_engine.hidden_request_profiles,
                )?
            } else {
                None
            };
            phys_engine.maybe_apply_hidden_request(hidden_request_signal.as_ref(), step);
            phys_engine.maybe_release_reentry_clamp();
            phys_engine.run_periodic_controller(&current_hidden)?;
            if let Some(capture_dir) = &args.runtime_hidden_capture_dir {
                if args.runtime_hidden_capture_every > 0
                    && step % args.runtime_hidden_capture_every == 0
                {
                    write_runtime_hidden_capture(
                        &args,
                        model.arch(),
                        capture_dir,
                        turn_index,
                        step,
                        user_prompt,
                        current_hidden.dims().to_vec(),
                        tensor_to_vec_f32(&live_hidden_1d)?,
                        drift_score,
                        ghost_vector.is_some(),
                        &phys_engine,
                    )?;
                }
            }
            // Temperature modulation: use reentry clamp + task-anchor clamp
            // When probe drifts from task geometry, lower temp for deterministic output
            let task_anchor_temp_modulation = if let (Some(ref task_anchor), Some(cache)) = (
                &phys_engine.current_task_anchor_signature,
                phys_engine.active_routing_cache(),
            ) {
                if let Some(routed) = phys_engine
                    .runtime_motifs
                    .iter()
                    .find(|m| m.motif_id == cache.motif_id)
                {
                    let is_structured = routed.motif_kind == "promoted"
                        || routed.motif_role == "structured"
                        || routed.motif_role == "structured_candidate";
                    if is_structured {
                        if let Ok(probe_64d) =
                            compress_hidden_state_to_64d(&current_hidden.flatten_all()?)
                        {
                            let sim = cosine_similarity_f32(&probe_64d, task_anchor);
                            // Lower temperature when probe drifts from task geometry
                            if sim < 0.35 {
                                let drift = (0.35 - sim).clamp(0.0, 1.0);
                                // Aggressive temp reduction: down to 0.35 at max drift
                                (1.0 - drift * 0.65).clamp(0.35, 1.0)
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    }
                } else {
                    1.0
                }
            } else {
                1.0
            };
            let clamp_temp_scale = if phys_engine.reentry_clamp_steps_remaining > 0 {
                (phys_engine.reentry_temp_scale * task_anchor_temp_modulation).clamp(0.3, 1.0)
            } else {
                task_anchor_temp_modulation
            };
            let clamp_viscosity_floor = if phys_engine.reentry_clamp_steps_remaining > 0 {
                (phys_engine.reentry_clamp_strength * 18.0).clamp(0.0, 18.0)
            } else {
                0.0
            };

            let retry_shield_active = mistake_reflex_retry_tokens_remaining > 0;
            let mistake_memory_surface_shield_active =
                !mistake_memory_surface_suppression_terms.is_empty();
            let filtered_sampling_active = phys_engine.last_guardrail_active
                || phys_engine.runtime_mode.uses_control_shield()
                || retry_shield_active
                || mistake_memory_surface_shield_active;

            if let Some(forced_token_id) = mistake_reflex_forced_retry_tokens.pop_front() {
                next_token_id = forced_token_id;
            } else if filtered_sampling_active {
                let mut attempts = 0;
                const MAX_ATTEMPTS: usize = 12;
                let mut combined_blocked_ids: Option<HashSet<u32>> = None;
                if phys_engine.runtime_mode.uses_control_shield()
                    || retry_shield_active
                    || mistake_memory_surface_shield_active
                {
                    let mut ids = HashSet::new();
                    if phys_engine.runtime_mode.uses_control_shield() {
                        ids.extend(phys_engine.control_token_ids.iter().copied());
                    }
                    if retry_shield_active {
                        ids.extend(mistake_reflex_retry_token_ids.iter().copied());
                    }
                    if mistake_memory_surface_shield_active {
                        ids.extend(mistake_memory_surface_suppression_token_ids.iter().copied());
                    }
                    combined_blocked_ids = Some(ids);
                }

                loop {
                    if args.answer_logit_probe_out.is_some() {
                        let (token_id, snapshot) = sample_token_with_snapshot(
                            &logits_filtered,
                            args.temperature,
                            clamp_temp_scale,
                            clamp_viscosity_floor,
                            &mut rng,
                            combined_blocked_ids.as_ref(),
                            phys_engine.stdout_debug(),
                        )?;
                        next_token_id = token_id;
                        sampling_snapshot_for_probe = Some(snapshot);
                        sampling_branch = format!("filtered_attempt_{attempts}");
                    } else {
                        next_token_id = sample_token(
                            &logits_filtered,
                            args.temperature,
                            clamp_temp_scale,
                            clamp_viscosity_floor,
                            &mut rng,
                            combined_blocked_ids.as_ref(),
                            phys_engine.stdout_debug(),
                        )?;
                    }

                    let decoded_candidate = model
                        .tokenizer()
                        .decode(&[next_token_id], true)
                        .unwrap_or_default();
                    let is_garbage = phys_engine.last_guardrail_active
                        && garbage_patterns
                            .iter()
                            .any(|pattern| decoded_candidate.contains(pattern));
                    let is_surface_violation = match phys_engine.runtime_mode {
                        RuntimeMode::Research | RuntimeMode::Agency => false,
                        RuntimeMode::Clean => clean_mode_surface_violation(
                            &phys_engine.surface_buffer,
                            &decoded_candidate,
                        ),
                    };
                    let is_retry_violation = retry_shield_active
                        && mistake_reflex_retry_token_ids.contains(&next_token_id);
                    let is_mistake_memory_surface_violation = mistake_memory_surface_shield_active
                        && MistakeMemory::surface_suppression_violation(
                            &assistant_text,
                            &decoded_candidate,
                            &mistake_memory_surface_suppression_terms,
                        );

                    if (!is_garbage
                        && !is_surface_violation
                        && !is_retry_violation
                        && !is_mistake_memory_surface_violation)
                        || attempts >= MAX_ATTEMPTS
                    {
                        if attempts > 0
                            && (is_garbage
                                || is_surface_violation
                                || is_retry_violation
                                || is_mistake_memory_surface_violation)
                        {
                            if phys_engine.stdout_debug() {
                                println!(
                                    " [FILTER] Failed to escape blocked surface after {} attempts",
                                    attempts
                                );
                            }
                        }
                        break;
                    }

                    if is_garbage {
                        if phys_engine.stdout_debug() {
                            println!(
                                " [DEBRIS] Blocked '{}', re-sampling (attempt {})",
                                decoded_candidate.trim(),
                                attempts + 1
                            );
                        }
                    } else if is_surface_violation {
                        if phys_engine.stdout_debug() {
                            println!(
                                " [SHIELD] Blocked '{}', re-sampling (attempt {})",
                                decoded_candidate.trim(),
                                attempts + 1
                            );
                        }
                    } else if is_retry_violation && phys_engine.stdout_debug() {
                        println!(
                            " [MISTAKE_REFLEX_RETRY] Blocked '{}', re-sampling (attempt {})",
                            decoded_candidate.trim(),
                            attempts + 1
                        );
                    } else if is_mistake_memory_surface_violation && phys_engine.stdout_debug() {
                        println!(
                            " [MISTAKE_MEMORY_SURFACE] Blocked stale surface '{}', re-sampling (attempt {})",
                            decoded_candidate.trim(),
                            attempts + 1
                        );
                    }

                    let vocab_size = logits_filtered.dim(D::Minus1)?;
                    if (next_token_id as usize) < vocab_size {
                        let mut logits_vec = logits_filtered.flatten_all()?.to_vec1::<f32>()?;
                        let guard_penalty = -100.0 * phys_engine.guardrail_bias_scale.max(1.0);
                        logits_vec[next_token_id as usize] = guard_penalty;
                        logits_filtered = Tensor::from_vec(
                            logits_vec,
                            logits_filtered.shape(),
                            logits_filtered.device(),
                        )?;
                    }

                    attempts += 1;
                }
            } else {
                if args.answer_logit_probe_out.is_some() {
                    let (token_id, snapshot) = sample_token_with_snapshot(
                        &logits_f32,
                        args.temperature,
                        clamp_temp_scale,
                        clamp_viscosity_floor,
                        &mut rng,
                        None,
                        phys_engine.stdout_debug(),
                    )?;
                    next_token_id = token_id;
                    sampling_snapshot_for_probe = Some(snapshot);
                    sampling_branch = "unfiltered".to_string();
                } else {
                    // Always use snapshot variant when competence-aware modulation
                    // is engaged, so entropy_norm flows into the next step's force
                    // application (§10aw). When threshold == 0.0, modulation is
                    // disabled and we fall back to the cheaper sample_token path.
                    if phys_engine.correction_packet_competence_entropy_threshold > 0.0 {
                        let (token_id, snapshot) = sample_token_with_snapshot(
                            &logits_f32,
                            args.temperature,
                            clamp_temp_scale,
                            clamp_viscosity_floor,
                            &mut rng,
                            None,
                            phys_engine.stdout_debug(),
                        )?;
                        next_token_id = token_id;
                        sampling_snapshot_for_probe = Some(snapshot);
                        sampling_branch = "competence_modulation".to_string();
                    } else {
                        next_token_id = sample_token(
                            &logits_f32,
                            args.temperature,
                            clamp_temp_scale,
                            clamp_viscosity_floor,
                            &mut rng,
                            None,
                            phys_engine.stdout_debug(),
                        )?;
                    }
                }
            }

            // Update engine's last_sampling_entropy_norm so the next step's
            // correction-packet force application can apply competence-aware
            // suppression (§10aw). Only updated when a snapshot is available.
            if let Some(snapshot) = sampling_snapshot_for_probe.as_ref() {
                phys_engine.last_sampling_entropy_norm = snapshot.entropy_norm;
            }

            if let Some(snapshot) = sampling_snapshot_for_probe.as_ref() {
                answer_logit_probe_records.push(answer_logit_probe_record(
                    model.tokenizer(),
                    &answer_logit_probe_targets,
                    snapshot,
                    turn_index,
                    step,
                    user_prompt,
                    &assistant_text,
                    next_token_id,
                    args.answer_logit_probe_top_k.max(1),
                    &sampling_branch,
                ));
            }

            if let Some(broadcast_tx) = &broadcast_tx {
                if let Ok(pos) = phys_engine.get_positions() {
                    if let Ok(pos_flat) = pos.flatten_all()?.to_vec1::<f32>() {
                        let positions: Vec<Vec<f32>> =
                            pos_flat.chunks(3).map(|c| c.to_vec()).take(1000).collect();
                        if let Err(e) = broadcast_tx.send(PhysicsUpdate {
                            step,
                            positions,
                            colors: vec![],
                        }) {
                            // broadcast::SendError means no receivers - this is common at startup
                            if step % 100 == 0 {
                                eprintln!("[WARN] Broadcast has no receivers: {}", e);
                            }
                        }
                    }
                }
            }

            index_pos += seq_len;

            if let Ok(txt) = model.tokenizer().decode(&[next_token_id], true) {
                // =========================================
                // PHASE 4: AUTONOMIC OVERRIDE (Model Self-Request)
                // Buffer-based detection for multi-token REQUEST tags
                // =========================================

                push_window_text(&mut phys_engine.request_buffer, &txt, 50);
                push_window_text(&mut phys_engine.surface_buffer, &txt, 96);

                // Check buffer for complete request tags
                if let Some(req) = detect_request(&phys_engine.request_buffer) {
                    let request_surface = phys_engine.request_buffer.clone();
                    if phys_engine.stdout_debug() {
                        println!(" [REQUEST DETECTED IN BUFFER] Type: {:?}", req);
                    }
                    emit_ui_event_value(
                        args.ui_events_json,
                        "visible_request",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "request_type": req.as_str(),
                            "surface": request_surface,
                        }),
                    );

                    // Clear buffer after detection
                    phys_engine.request_buffer.clear();
                    phys_engine.hidden_request_candidate = None;
                    phys_engine.hidden_request_streak = 0;
                    phys_engine.last_hidden_request_pressure = 0.0;
                    phys_engine.last_hidden_request = Some(req);

                    let (_applied, focus_gate_msg) = phys_engine.apply_request(req, step);

                    // If Focus Gate triggered, inject the denial message
                    if let Some(msg) = focus_gate_msg {
                        if phys_engine.stdout_debug() {
                            println!(" [FOCUS GATE TRIGGERED] Injecting: {}", msg);
                        }
                        phys_engine.pending_insight = Some(msg);
                    }

                    // === [REQUEST: REMEMBER] vault tether (safer "quiet queue + next-turn side note" design) ===
                    // Model emits REMEMBER (with optional payload) when it needs semantic context from its
                    // own creation / prior chats vault. We use the *current* 64D probe (the one at the step
                    // the tag was decoded) to query Qdrant. We quietly do the search + self-save here (so the
                    // runtime has the live probe and can persist niodoo's own memories keyed by geometry).
                    // We do NOT force-inject mid-generation (no stealing attention / auto-garbage of the current
                    // thought flow). Instead we append the [INTERNAL MEMORY RECALL] block to the *current* turn's
                    // assistant_text (visible in the speech log after the model's response for this turn).
                    // The py driver (run_prior_chats_read_aloud etc.) can then, when building the prompt for the
                    // *next* turn, surface the recall note at the "start" of the new context so the model sees it
                    // before its next response and can decide what to do with it (react with more tags, ignore,
                    // mint new packets from it, etc.). This gives the model agency and matches the safer design.
                    if req == crate::runtime::control_surface::RequestType::Remember {
                        if let Some(probe) = phys_engine.last_probe_bucket_mean_64 {
                            pending_vault_remember = Some((probe, request_surface.clone(), step));
                            eprintln!(
                                " [REMEMBER-VAULT] remember detected; pending save probe_norm={:.3}",
                                probe.iter().map(|x| x * x).sum::<f32>().sqrt()
                            );
                            if let Some(client) = phys_engine.vault_client.clone() {
                                // Quiet retrieval using the exact 64D probe that triggered the tag.
                                let hits = client.search_by_probe_64(&probe, 6);
                                if !hits.is_empty() {
                                    let block = format!(
                                        "\n[INTERNAL MEMORY RECALL]\n{}\n[/INTERNAL MEMORY RECALL]\n",
                                        hits.join("\n---\n")
                                    );
                                    // Append at end of *this* turn (quiet side note after the model's response).
                                    // No mid-gen forced tokens, no KV hijack of the current thought.
                                    // The read-aloud / self-talk driver can re-present the block at the
                                    // "start of the next turn" when it builds the next prompt if desired.
                                    assistant_text.push_str(&block);

                                    // Also surface via pending so telemetry / debug / py side can see it easily.
                                    phys_engine.pending_insight = Some(block.clone());

                                    if phys_engine.stdout_debug() {
                                        println!(" [REMEMBER-VAULT] quiet recall {} hits (probe norm ~{:.3})",
                                            hits.len(),
                                            probe.iter().map(|x| x*x).sum::<f32>().sqrt()
                                        );
                                    }
                                }
                            }
                        } else {
                            eprintln!(
                                " [REMEMBER-VAULT] remember detected but no 64D probe available"
                            );
                        }
                    }
                }

                let display_text = &txt;
                if phys_engine.stdout_debug() {
                    println!(" [DBG: Decoded '{}']", display_text.replace("\n", "\\n"));
                }
                if phys_engine.stdout_profile.chat_enabled() {
                    print!("{}", display_text);
                    std::io::stdout().flush()?;
                }
                emit_ui_event_value(
                    args.ui_events_json,
                    "decoded_token",
                    serde_json::json!({
                        "turn_index": turn_index,
                        "step": step,
                        "token": txt.clone(),
                        "engine_status": phys_engine.last_engine_status.as_str(),
                        "guardrail_active": phys_engine.last_guardrail_active,
                        "math_governor_relief_active": math_governor_relief_active,
                    }),
                );
                if phys_engine.stdout_debug() {
                    std::io::stdout().flush()?;
                }
                assistant_text.push_str(&txt);
                if let Some((probe, mut capture, start_step)) = pending_vault_remember.take() {
                    if step != start_step {
                        capture.push_str(&txt);
                    }
                    let first_line = capture.lines().next().unwrap_or(&capture).trim();
                    let payloads =
                        crate::main_helpers2::extract_agency_tag_payloads(first_line, "REMEMBER");
                    let remember_line_closed =
                        crate::main_helpers2::find_agency_tag(&capture, "REMEMBER")
                            .map(|(_, end)| {
                                capture[end..].contains('\n') || capture[end..].contains('\r')
                            })
                            .unwrap_or(false);
                    let payload_ready = payloads
                        .last()
                        .map(|payload| {
                            !payload.is_empty() && (remember_line_closed || payload.len() >= 96)
                        })
                        .unwrap_or(false);
                    let should_save = payload_ready
                        || capture.len() >= 300
                        || step.saturating_sub(start_step) >= 48;
                    if should_save {
                        let save_text = payloads.last().cloned().unwrap_or_else(|| {
                            capture.trim().chars().take(600).collect::<String>()
                        });
                        if save_text.trim().is_empty() {
                            eprintln!(" [REMEMBER-VAULT] save skipped: empty memory text");
                        } else if let Some(client) = phys_engine.vault_client.clone() {
                            match client.save_self_memory(&probe, &save_text) {
                                Ok(()) => eprintln!(
                                    " [REMEMBER-VAULT] saved self-memory bytes={} probe_norm={:.3}",
                                    save_text.len(),
                                    probe.iter().map(|x| x * x).sum::<f32>().sqrt()
                                ),
                                Err(err) => {
                                    eprintln!(" [REMEMBER-VAULT] save_self_memory error: {:#}", err)
                                }
                            }
                        } else {
                            eprintln!(" [REMEMBER-VAULT] save skipped: no vault client");
                        }
                    } else {
                        pending_vault_remember = Some((probe, capture, start_step));
                    }
                }
                let mut mistake_guard_snapshot = mistake_guard.observe(&assistant_text);
                let mut mistake_reflex_snapshot =
                    mistake_reflex_guard.observe(step, &assistant_text);
                let mistake_reflex_retry_triggered_this_step = args
                    .mistake_reflex_retry_on_old_mistake
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && args.mistake_reflex_action_mode == MistakeReflexActionMode::HiddenControl
                    && mistake_reflex_snapshot.old_mistake_seen
                    && mistake_reflex_snapshot
                        .domains
                        .iter()
                        .any(|domain| domain == "parallel_duration:drying")
                    && mistake_reflex_retry_count < args.mistake_reflex_retry_max;
                if mistake_reflex_retry_triggered_this_step {
                    mistake_reflex_retry_count = mistake_reflex_retry_count.saturating_add(1);
                    mistake_reflex_retry_tokens_remaining = args.mistake_reflex_retry_shield_tokens;
                    mistake_reflex_retry_reason =
                        Some("parallel_duration_serial_path_detected".to_string());
                    let (_applied, _focus_gate_msg) =
                        phys_engine.apply_request(RequestType::Spike, step);
                    phys_engine.focus_lock_remaining_ticks = 0;
                    phys_engine.adrenaline = phys_engine.adrenaline.max(6.0);
                    phys_engine.physics_blend = phys_engine.physics_blend.max(6.5);
                    phys_engine.dynamic_repulsion = phys_engine.dynamic_repulsion.min(-3.0);
                    if args.mistake_reflex_retry_inject_control_surface {
                        let retry_surface = "\n[REQUEST: SPIKE]\n\nVISIBLE REASONING:\n";
                        let retry_ids = model
                            .tokenizer()
                            .encode(retry_surface, false)
                            .map_err(|err| anyhow::anyhow!(err))?
                            .get_ids()
                            .to_vec();
                        mistake_reflex_forced_retry_tokens.extend(retry_ids);
                    }
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_retry_triggered",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "reason": mistake_reflex_retry_reason.clone(),
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "shield_tokens": mistake_reflex_retry_tokens_remaining,
                            "control_surface_injected": args.mistake_reflex_retry_inject_control_surface,
                        }),
                    );
                    if phys_engine.stdout_debug() || phys_engine.stdout_telemetry() {
                        println!(
                            " [MISTAKE_REFLEX_RETRY] step={} reason={} shield_tokens={}",
                            step,
                            mistake_reflex_retry_reason.as_deref().unwrap_or("-"),
                            mistake_reflex_retry_tokens_remaining
                        );
                    }
                }
                let finalization_decision = finalizer.observe_token(step, &assistant_text, &txt);
                let mut finalization_snapshot = finalizer.snapshot();
                if finalization_decision.should_stop && mistake_guard.should_block_finalization() {
                    mistake_guard.record_blocked_lock();
                    finalizer.veto_current_stop();
                    finalization_snapshot = finalizer.snapshot();
                    mistake_guard_snapshot = mistake_guard.snapshot();
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_guard_blocked_lock",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "reason": finalization_decision.reason,
                            "event_ids": mistake_guard_snapshot.event_ids.clone(),
                            "rejected_answer_seen": mistake_guard_snapshot.rejected_answer_seen,
                            "accepted_answer_seen": mistake_guard_snapshot.accepted_answer_seen,
                        }),
                    );
                    if phys_engine.stdout_debug() || phys_engine.stdout_telemetry() {
                        println!(
                            " [MISTAKE_GUARD] blocked_lock step={} matches={} rejected_seen={} accepted_seen={}",
                            step,
                            mistake_guard_snapshot.match_count,
                            mistake_guard_snapshot.rejected_answer_seen,
                            mistake_guard_snapshot.accepted_answer_seen
                        );
                    }
                } else if finalization_decision.should_stop
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && mistake_reflex_guard.should_block_finalization()
                {
                    mistake_reflex_guard.record_blocked_lock();
                    finalizer.veto_current_stop();
                    finalization_snapshot = finalizer.snapshot();
                    mistake_reflex_snapshot = mistake_reflex_guard.snapshot();
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_blocked_lock",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "reason": finalization_decision.reason,
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "evidence_seen": mistake_reflex_snapshot.evidence_seen,
                            "old_mistake_seen": mistake_reflex_snapshot.old_mistake_seen,
                        }),
                    );
                    if phys_engine.stdout_debug() || phys_engine.stdout_telemetry() {
                        println!(
                            " [MISTAKE_REFLEX] blocked_lock step={} matches={} evidence_seen={} old_mistake_seen={}",
                            step,
                            mistake_reflex_snapshot.match_count,
                            mistake_reflex_snapshot.evidence_seen,
                            mistake_reflex_snapshot.old_mistake_seen
                        );
                    }
                } else if finalization_decision.should_stop
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && mistake_reflex_snapshot.matched
                    && mistake_reflex_snapshot.earned_answer_seen
                {
                    finalization_stop_reason = Some("gmms_earned_boundary_lock".to_string());
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_earned_lock",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "earned_answer_text": mistake_reflex_snapshot.earned_answer_text.clone(),
                            "lock_reason": finalization_decision.reason,
                        }),
                    );
                } else if finalization_decision.should_stop {
                    finalization_stop_reason = finalization_decision.reason.clone();
                    emit_ui_event_value(
                        args.ui_events_json,
                        "lock_stop_triggered",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "reason": finalization_decision.reason,
                            "lock_text": finalization_snapshot.lock_text.clone(),
                            "tokens_after_lock": finalization_snapshot.tokens_after_lock,
                        }),
                    );
                }
                let answer_boundary_decision =
                    answer_boundary_finalizer.observe_text(step, &assistant_text);
                if finalization_stop_reason.is_none() && answer_boundary_decision.should_stop {
                    let answer_boundary_snapshot = answer_boundary_finalizer.snapshot();
                    finalization_stop_reason = answer_boundary_decision.reason.clone();
                    emit_ui_event_value(
                        args.ui_events_json,
                        "answer_boundary_finalizer_fired",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "kind": answer_boundary_snapshot.kind.clone(),
                            "source_term": answer_boundary_snapshot.source_term.clone(),
                            "expected_answer": answer_boundary_snapshot.expected_answer.clone(),
                            "operation": answer_boundary_snapshot.operation.clone(),
                            "lhs": answer_boundary_snapshot.lhs,
                            "rhs": answer_boundary_snapshot.rhs,
                            "replacement_answer": answer_boundary_snapshot.replacement_answer.clone(),
                            "stop_reason": answer_boundary_decision.reason.clone(),
                        }),
                    );
                }
                if finalization_stop_reason.is_none()
                    && args.mistake_memory_stop_on_accepted
                    && mistake_guard_snapshot.matched
                    && mistake_guard_snapshot.accepted_boundary_seen
                    && (!mistake_guard_snapshot.rejected_answer_seen
                        || mistake_guard_snapshot.claim_review_evidence_gate_match)
                {
                    mistake_memory_preserve_byte_len =
                        mistake_guard_snapshot.accepted_boundary_byte_len;
                    finalization_stop_reason = Some("mistake_memory_accepted_answer".to_string());
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_memory_stop_on_accepted",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "event_ids": mistake_guard_snapshot.event_ids.clone(),
                        }),
                    );
                }
                if finalization_stop_reason.is_none()
                    && args.mistake_reflex_stop_on_earned_answer
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && mistake_reflex_snapshot.matched
                    && mistake_reflex_snapshot.earned_answer_seen
                    && mistake_reflex_earned_sentence_complete(
                        &assistant_text,
                        mistake_reflex_snapshot.earned_boundary_byte_len,
                    )
                {
                    reflex_preserve_byte_len = Some(assistant_text.len());
                    finalization_stop_reason = Some("gmms_earned_boundary_sentence".to_string());
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_earned_boundary_sentence",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "earned_answer_text": mistake_reflex_snapshot.earned_answer_text.clone(),
                            "earned_boundary_step": mistake_reflex_snapshot.earned_boundary_step,
                            "earned_boundary_byte_len": mistake_reflex_snapshot.earned_boundary_byte_len,
                        }),
                    );
                } else if finalization_stop_reason.is_none()
                    && args.mistake_reflex_stop_on_earned_answer
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && mistake_reflex_snapshot.matched
                    && mistake_reflex_snapshot.earned_answer_seen
                    && mistake_reflex_snapshot.old_path_after_earned
                {
                    reflex_preserve_byte_len = mistake_reflex_snapshot.earned_boundary_byte_len;
                    finalization_stop_reason =
                        Some("gmms_preserve_earned_before_drift".to_string());
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_preserve_earned_before_drift",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "earned_answer_text": mistake_reflex_snapshot.earned_answer_text.clone(),
                            "earned_boundary_step": mistake_reflex_snapshot.earned_boundary_step,
                            "earned_boundary_byte_len": mistake_reflex_snapshot.earned_boundary_byte_len,
                        }),
                    );
                } else if finalization_stop_reason.is_none()
                    && args.mistake_reflex_stop_on_earned_answer
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && mistake_reflex_snapshot.matched
                    && mistake_reflex_snapshot.earned_answer_seen
                    && finalization_snapshot.lock_detected
                {
                    finalization_stop_reason = Some("gmms_earned_boundary_lock".to_string());
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_earned_lock",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "earned_answer_text": mistake_reflex_snapshot.earned_answer_text.clone(),
                        }),
                    );
                } else if finalization_stop_reason.is_none()
                    && args.mistake_reflex_stop_on_earned_answer
                    && args.mistake_reflex_mode == MistakeReflexMode::Influence
                    && mistake_reflex_snapshot.matched
                    && mistake_reflex_snapshot.earned_answer_seen
                    && mistake_reflex_snapshot
                        .earned_boundary_step
                        .map(|boundary| {
                            step.saturating_sub(boundary) >= args.mistake_reflex_earned_taper_tokens
                        })
                        .unwrap_or(false)
                {
                    finalization_stop_reason =
                        Some("gmms_earned_boundary_taper_exhausted".to_string());
                    emit_ui_event_value(
                        args.ui_events_json,
                        "mistake_reflex_earned_taper_exhausted",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "event_ids": mistake_reflex_snapshot.event_ids.clone(),
                            "domains": mistake_reflex_snapshot.domains.clone(),
                            "earned_answer_text": mistake_reflex_snapshot.earned_answer_text.clone(),
                            "earned_boundary_step": mistake_reflex_snapshot.earned_boundary_step,
                        }),
                    );
                }
                let motif_provenance =
                    summarize_runtime_motif_provenance(&phys_engine.runtime_motifs);
                let count_route_memory_effective_finalization_candidate =
                    count_route_memory_enumeration_aggregation_candidate(
                        args.count_route_memory_finalization_enumeration_aggregation_action
                            || args.count_route_memory_finalization_enumeration_preserve_stop,
                        count_route_memory_finalization_candidate.as_ref(),
                        &assistant_text,
                    );
                let count_finalization_candidate_telemetry =
                    count_route_memory_finalization_telemetry(
                        args.count_route_memory_finalization_candidate_telemetry
                            || args.count_route_memory_finalization_replacement_action
                            || args.count_route_memory_finalization_natural_v2_replacement_action
                            || args.count_route_memory_finalization_enumeration_aggregation_action
                            || args.count_route_memory_finalization_enumeration_preserve_stop
                            || args.count_route_memory_finalization_protected_lock_surface,
                        count_route_memory_effective_finalization_candidate.as_ref(),
                        &assistant_text,
                    );
                let (count_finalization_action, replacement_text) =
                    count_route_memory_finalization_action(
                        args.count_route_memory_finalization_replacement_action,
                        args.count_route_memory_finalization_natural_v2_replacement_action,
                        args.count_route_memory_finalization_enumeration_aggregation_action,
                        args.count_route_memory_finalization_enumeration_preserve_stop,
                        args.count_route_memory_finalization_protected_lock_surface,
                        count_route_memory_effective_finalization_candidate.as_ref(),
                        &count_finalization_candidate_telemetry,
                        &assistant_text,
                    );
                if let Some(replacement_text) = replacement_text {
                    assistant_text = replacement_text;
                    finalization_stop_reason = count_finalization_action.stop_reason.clone();
                    emit_ui_event_value(
                        args.ui_events_json,
                        "count_route_memory_finalization_replacement",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "replacement_answer": count_finalization_action.replacement_answer.clone(),
                            "original_answer_window": count_finalization_action.original_answer_window.clone(),
                            "reason": count_finalization_action.action_reason.clone(),
                        }),
                    );
                    if phys_engine.stdout_debug() || phys_engine.stdout_telemetry() {
                        println!(
                            " [COUNT_FINALIZATION] replacement_applied answer={} reason={}",
                            count_finalization_action
                                .replacement_answer
                                .as_deref()
                                .unwrap_or("-"),
                            count_finalization_action
                                .action_reason
                                .as_deref()
                                .unwrap_or("-")
                        );
                        std::io::stdout().flush()?;
                    }
                }

                let mut token_trace = TokenPhysics {
                    token: txt.clone(),
                    step,
                    engine_status: phys_engine.last_engine_status,
                    forces_applied: phys_engine.last_forces_applied,
                    gravity_force: phys_engine.last_gravity_mag,
                    ghost_pre_norm: phys_engine.last_ghost_pre_norm,
                    ghost_gain: phys_engine.last_ghost_gain,
                    applied_ghost_force: if phys_engine.last_forces_applied {
                        phys_engine.last_applied_ghost_mag
                    } else {
                        0.0
                    },
                    applied_ghost_vector: if phys_engine.last_forces_applied {
                        phys_engine.last_applied_ghost_vector.clone()
                    } else {
                        None
                    },
                    goal_force: phys_engine.last_goal_mag,
                    repulsion_force: phys_engine.last_repulsion_mag,
                    motif_force: phys_engine.last_motif_mag,
                    recovery_force: phys_engine.last_recovery_mag,
                    total_force: if phys_engine.last_forces_applied {
                        phys_engine.last_gravity_mag
                            + phys_engine.last_applied_ghost_mag
                            + phys_engine.last_goal_mag
                            + phys_engine.last_repulsion_mag
                            + phys_engine.last_motif_mag
                            + phys_engine.last_recovery_mag
                    } else {
                        0.0
                    },
                    activation_gate: phys_engine.last_activation_gate,
                    empathy_spike: phys_engine.empathy_spike,
                    live_motif_count: phys_engine.last_live_motif_count,
                    bridge_motif_count: motif_provenance.bridge_count,
                    organic_promoted_count: motif_provenance.organic_promoted_count,
                    recovered_promoted_count: motif_provenance.recovered_promoted_count,
                    restored_compact_count: motif_provenance.restored_compact_count,
                    nearest_live_motif_distance: phys_engine.last_live_motif_distance,
                    nearest_live_motif_radius: phys_engine.last_live_motif_radius,
                    bridge_force_selection: phys_engine.last_bridge_force_selection.clone(),
                    bridge_force_selected_count: phys_engine.last_bridge_force_selected_count,
                    bridge_force_selected_ids: phys_engine.last_bridge_force_selected_ids.clone(),
                    bridge_force_selection_source: phys_engine
                        .last_bridge_force_selection_source
                        .clone(),
                    bridge_force_selected_score_max: phys_engine
                        .last_bridge_force_selected_score_max,
                    bridge_force_selected_role: phys_engine.last_bridge_force_selected_role.clone(),
                    bridge_force_second_score: phys_engine.last_bridge_force_second_score,
                    bridge_force_selected_margin: phys_engine.last_bridge_force_selected_margin,
                    bridge_force_role_filter: phys_engine.last_bridge_force_role_filter.clone(),
                    bridge_force_min_margin: phys_engine.last_bridge_force_min_margin,
                    routed_motif_id: phys_engine.last_routed_motif_id.clone(),
                    routed_motif_role: phys_engine.last_routed_motif_role.clone(),
                    routed_motif_score: if phys_engine.last_routed_motif_score.is_finite() {
                        Some(phys_engine.last_routed_motif_score)
                    } else {
                        None
                    },
                    route_surface_id: phys_engine
                        .last_routed_motif_id
                        .clone()
                        .or_else(|| phys_engine.last_bridge_force_selected_ids.first().cloned())
                        .or_else(|| phys_engine.last_nearest_ghost_id.clone()),
                    route_surface_source: if phys_engine.last_routed_motif_id.is_some() {
                        Some("controller_route".to_string())
                    } else if !phys_engine.last_bridge_force_selected_ids.is_empty() {
                        Some("bridge_force_selection".to_string())
                    } else if phys_engine.last_nearest_ghost_id.is_some() {
                        Some("nearest_ghost".to_string())
                    } else {
                        None
                    },
                    route_surface_role: phys_engine
                        .last_routed_motif_role
                        .clone()
                        .or_else(|| phys_engine.last_bridge_force_selected_role.clone()),
                    controller_candidate_count: Some(
                        phys_engine.last_controller_candidates.len() as u32
                    ),
                    live_basin_pressure: phys_engine.last_live_basin_pressure,
                    surface_heuristic_flag: txt.contains("#")
                        || txt.contains("<|")
                        || txt.len() > 15,
                    lock_detected: finalization_snapshot.lock_detected,
                    lock_detected_step: finalization_snapshot.lock_detected_step,
                    lock_text: finalization_snapshot.lock_text.clone(),
                    lock_stop_policy: finalization_snapshot.lock_stop_policy.clone(),
                    lock_taper_remaining: finalization_snapshot.lock_taper_remaining,
                    lock_stop_triggered: finalization_snapshot.lock_stop_triggered,
                    lock_stop_reason: finalization_snapshot.lock_stop_reason.clone(),
                    tokens_after_lock: finalization_snapshot.tokens_after_lock,
                    mistake_memory_matched: mistake_guard_snapshot.matched,
                    mistake_memory_match_count: mistake_guard_snapshot.match_count,
                    mistake_memory_event_ids: mistake_guard_snapshot.event_ids.clone(),
                    mistake_rejected_answer_seen: mistake_guard_snapshot.rejected_answer_seen,
                    mistake_accepted_answer_seen: mistake_guard_snapshot.accepted_answer_seen,
                    mistake_accepted_boundary_seen: mistake_guard_snapshot.accepted_boundary_seen,
                    mistake_guard_blocked_lock: mistake_guard_snapshot.blocked_lock,
                    mistake_guard_blocked_count: mistake_guard_snapshot.blocked_count,
                    mistake_reflex_matched: mistake_reflex_snapshot.matched,
                    mistake_reflex_match_count: mistake_reflex_snapshot.match_count,
                    mistake_reflex_event_ids: mistake_reflex_snapshot.event_ids.clone(),
                    mistake_reflex_domains: mistake_reflex_snapshot.domains.clone(),
                    mistake_reflex_action_level: mistake_reflex_snapshot.action_level,
                    mistake_reflex_resolution_level: mistake_reflex_snapshot.resolution_level,
                    mistake_reflex_vector_slice_available: mistake_reflex_snapshot
                        .vector_slice_available,
                    mistake_reflex_unicode_packet_ids: mistake_reflex_snapshot
                        .unicode_packet_ids
                        .clone(),
                    mistake_reflex_route_preserved: mistake_reflex_snapshot.route_preserved,
                    mistake_reflex_unfold_reason: mistake_reflex_snapshot.unfold_reason.clone(),
                    mistake_reflex_decay_reason: mistake_reflex_snapshot.decay_reason.clone(),
                    mistake_reflex_evidence_seen: mistake_reflex_snapshot.evidence_seen,
                    mistake_reflex_accepted_answer_candidate_seen: mistake_reflex_snapshot
                        .accepted_answer_candidate_seen,
                    mistake_reflex_old_mistake_seen: mistake_reflex_snapshot.old_mistake_seen,
                    mistake_reflex_old_path_after_earned: mistake_reflex_snapshot
                        .old_path_after_earned,
                    mistake_reflex_earned_answer_seen: mistake_reflex_snapshot.earned_answer_seen,
                    mistake_reflex_earned_answer_text: mistake_reflex_snapshot
                        .earned_answer_text
                        .clone(),
                    mistake_reflex_earned_boundary_step: mistake_reflex_snapshot
                        .earned_boundary_step,
                    mistake_reflex_earned_boundary_byte_len: mistake_reflex_snapshot
                        .earned_boundary_byte_len,
                    mistake_reflex_lock_blocked: mistake_reflex_snapshot.blocked_lock,
                    mistake_reflex_blocked_count: mistake_reflex_snapshot.blocked_count,
                    mistake_reflex_retry_triggered: mistake_reflex_retry_triggered_this_step,
                    mistake_reflex_retry_count,
                    mistake_reflex_retry_reason: mistake_reflex_retry_reason.clone(),
                    mistake_reflex_retry_tokens_remaining,
                    mistake_reflex_prompt_applied,
                    mistake_reflex_prompt_injection_timing: mistake_reflex_prompt_injection_timing
                        .clone(),
                    mistake_reflex_prompt_injection_repeated,
                    mistake_reflex_prompt_hint_text: mistake_reflex_prompt_hint_text.clone(),
                    // Bridge Telemetry (niodv4_bridge)
                    bridge_enabled: phys_engine.bridge_enabled,
                    req_id: phys_engine.current_run_id.clone(),
                    prompt_hash: phys_engine.last_prompt_hash.clone(),
                    ghost_basins_loaded: phys_engine.ghost_basins_loaded,
                    nearest_ghost_id: phys_engine.last_nearest_ghost_id.clone(),
                    nearest_ghost_distance: phys_engine.last_nearest_ghost_distance,
                    second_nearest_ghost_distance: phys_engine.last_second_nearest_ghost_distance,
                    route_margin: phys_engine.last_route_margin,
                    projection_strategy: phys_engine.last_projection_strategy.clone(),
                    ghost_pull_delta_norm: phys_engine.last_ghost_pull_delta_norm,
                    intervention_applied: phys_engine.last_intervention_applied,
                    gate34_target_source: if phys_engine.bridge_gate34_latch_active() {
                        Some(phys_engine.gate34_target_source.clone())
                    } else {
                        None
                    },
                    gate34_target_kind: if phys_engine.bridge_gate34_latch_active() {
                        Some(phys_engine.gate34_target_kind().to_string())
                    } else {
                        None
                    },
                    gate34_phase: Some(phys_engine.gate34_phase.as_str().to_string()),
                    gate34_target_ghost_id: phys_engine.gate34_target_ghost_id.clone(),
                    gate34_target_specialist_id: phys_engine.gate34_target_specialist_id.clone(),
                    gate34_target_motif_id: phys_engine.gate34_target_motif_id.clone(),
                    gate34_target_acquired_step: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_target_acquired_step)
                    } else {
                        None
                    },
                    gate34_target_margin_at_acquire: if phys_engine.gate34_target_acquired_step >= 0
                    {
                        Some(phys_engine.gate34_target_margin_at_acquire)
                    } else {
                        None
                    },
                    gate34_target_distance_at_acquire: if phys_engine.gate34_target_acquired_step
                        >= 0
                    {
                        Some(phys_engine.gate34_target_distance_at_acquire)
                    } else {
                        None
                    },
                    gate34_current_target_distance: if phys_engine.gate34_phase
                        == Gate34Phase::Latched
                        || phys_engine.gate34_phase == Gate34Phase::Released
                    {
                        Some(phys_engine.gate34_current_target_distance)
                    } else {
                        None
                    },
                    gate34_warmup_distance_min: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_target_warmup_distance_min)
                    } else {
                        None
                    },
                    gate34_warmup_distance_mean: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_target_warmup_distance_mean)
                    } else {
                        None
                    },
                    gate34_warmup_distance_max: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_target_warmup_distance_max)
                    } else {
                        None
                    },
                    gate34_warmup_distance_std: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_target_warmup_distance_std)
                    } else {
                        None
                    },
                    gate34_distance_drift_score: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_last_distance_drift_score)
                    } else {
                        None
                    },
                    gate34_distance_limit_ratio: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_last_distance_limit_ratio)
                    } else {
                        None
                    },
                    gate34_distance_limit_warmup: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_last_distance_limit_warmup)
                    } else {
                        None
                    },
                    gate34_distance_gate_mode: if phys_engine.gate34_target_acquired_step >= 0 {
                        Some(phys_engine.gate34_last_distance_gate_mode.clone())
                    } else {
                        None
                    },
                    gate34_target_hold_remaining: if phys_engine.gate34_phase
                        == Gate34Phase::Latched
                    {
                        Some(
                            phys_engine
                                .gate34_hold_steps
                                .saturating_sub(phys_engine.gate34_held_step_count),
                        )
                    } else {
                        None
                    },
                    gate34_release_reason: phys_engine.gate34_release_reason.clone(),
                    gate34_latched_steps: Some(phys_engine.gate34_held_step_count),
                    gate34_intervention_count: Some(phys_engine.gate34_intervention_count),
                    active_recovery_specialist_id: phys_engine
                        .active_recovery_specialist_id
                        .clone(),
                    active_recovery_weight: Some(phys_engine.active_recovery_weight),
                    specialist_run_length: Some(phys_engine.specialist_run_length),
                    specialist_worker_enabled: phys_engine.last_specialist_worker_enabled,
                    specialist_worker_mode: phys_engine
                        .specialist_memory_workers_mode
                        .as_str()
                        .to_string(),
                    specialist_worker_selected_id: phys_engine
                        .last_specialist_worker_selected_id
                        .clone(),
                    specialist_worker_packet_id: phys_engine
                        .last_specialist_worker_packet_id
                        .clone(),
                    specialist_worker_unicode_escape: phys_engine
                        .last_specialist_worker_unicode_escape
                        .clone(),
                    specialist_worker_original_route_id: phys_engine
                        .last_specialist_worker_original_route_id
                        .clone(),
                    specialist_worker_decoded_route_id: phys_engine
                        .last_specialist_worker_decoded_route_id
                        .clone(),
                    specialist_worker_route_preserved: phys_engine
                        .last_specialist_worker_route_preserved,
                    specialist_worker_topk_hit: phys_engine.last_specialist_worker_topk_hit,
                    specialist_worker_score: phys_engine.last_specialist_worker_score,
                    specialist_worker_source_prompt_id: phys_engine
                        .last_specialist_worker_source_prompt_id
                        .clone(),
                    specialist_worker_direction_source: phys_engine
                        .last_specialist_worker_direction_source
                        .clone(),
                    specialist_worker_delta_norm_64d: phys_engine
                        .last_specialist_worker_delta_norm_64d,
                    specialist_worker_hidden_delta_norm: phys_engine
                        .last_specialist_worker_hidden_delta_norm,
                    specialist_worker_influence_clamp: phys_engine
                        .last_specialist_worker_influence_clamp,
                    specialist_worker_influence_scale: phys_engine
                        .last_specialist_worker_influence_scale,
                    specialist_worker_probe_signature_64d: phys_engine
                        .last_specialist_worker_probe_signature_64d
                        .clone(),
                    specialist_worker_target_signature_64d: phys_engine
                        .last_specialist_worker_target_signature_64d
                        .clone(),
                    count_route_memory_finalization_candidate_enabled:
                        count_finalization_candidate_telemetry.candidate_enabled,
                    count_route_memory_finalization_candidate_answer:
                        count_finalization_candidate_telemetry
                            .candidate_answer
                            .clone(),
                    count_route_memory_finalization_candidate_word:
                        count_finalization_candidate_telemetry
                            .candidate_word
                            .clone(),
                    count_route_memory_finalization_candidate_target_letter:
                        count_finalization_candidate_telemetry
                            .candidate_target_letter
                            .clone(),
                    count_route_memory_finalization_candidate_parser_confidence:
                        count_finalization_candidate_telemetry.parser_confidence,
                    count_route_memory_finalization_candidate_parser_version:
                        count_finalization_candidate_telemetry
                            .parser_version
                            .clone(),
                    count_route_memory_finalization_candidate_state:
                        count_finalization_candidate_telemetry.state.clone(),
                    count_route_memory_finalization_answer_signature_seen:
                        count_finalization_candidate_telemetry
                            .answer_signature_seen
                            .clone(),
                    count_route_memory_finalization_do_no_harm_protected:
                        count_finalization_candidate_telemetry.do_no_harm_protected,
                    count_route_memory_finalization_would_apply:
                        count_finalization_candidate_telemetry.would_apply,
                    count_route_memory_finalization_action_enabled: count_finalization_action
                        .action_enabled,
                    count_route_memory_finalization_action_applied: count_finalization_action
                        .action_applied,
                    count_route_memory_finalization_action_reason: count_finalization_action
                        .action_reason
                        .clone(),
                    count_route_memory_finalization_replacement_answer: count_finalization_action
                        .replacement_answer
                        .clone(),
                    count_route_memory_finalization_original_answer_window:
                        count_finalization_action.original_answer_window.clone(),
                    count_route_memory_finalization_stop_reason: count_finalization_action
                        .stop_reason
                        .clone(),
                    prompt_embedding_source: if phys_engine.bridge_gate34_latch_active() {
                        Some(phys_engine.prompt_embedding_source.clone())
                    } else {
                        None
                    },
                    prompt_vec_norm: if phys_engine.bridge_gate34_latch_active() {
                        Some(phys_engine.prompt_vec_norm)
                    } else {
                        None
                    },
                    prompt_similarity_unavailable: if phys_engine.bridge_gate34_latch_active() {
                        Some(phys_engine.prompt_similarity_unavailable)
                    } else {
                        None
                    },
                    vq_code_assigned: phys_engine.last_vq_code_assigned,
                    vq_encode_error: phys_engine.last_vq_encode_error,
                    correction_delta_norm: phys_engine.last_correction_delta_norm,
                    specialist_activated: phys_engine.last_specialist_activated,
                    specialist_force_applied: phys_engine.last_specialist_force_applied,
                    specialist_force_norm: phys_engine.last_specialist_force_norm,
                    correction_packet_vq_code: phys_engine.last_correction_packet_vq_code,
                    correction_packet_fire_count: phys_engine.last_correction_packet_fire_count,
                    correction_packet_force_norm: phys_engine.last_correction_packet_force_norm,
                    correction_packet_ids: phys_engine.last_correction_packet_ids.clone(),
                    packet_authority_score: phys_engine.last_packet_authority_score,
                    packet_authority_allowed: phys_engine.last_packet_authority_allowed,
                    packet_authority_reason: phys_engine.last_packet_authority_reason.clone(),
                    packet_authority_blocked_reason: phys_engine
                        .last_packet_authority_blocked_reason
                        .clone(),
                    correction_packet_arbitration_mode: phys_engine
                        .last_correction_packet_arbitration_mode
                        .clone(),
                    correction_packet_arbitration_reason: phys_engine
                        .last_correction_packet_arbitration_reason
                        .clone(),
                    correction_packet_arbitration_candidate_count: phys_engine
                        .last_correction_packet_arbitration_candidate_count,
                    correction_packet_arbitration_min_target_distance: phys_engine
                        .last_correction_packet_arbitration_min_target_distance,
                    correction_packet_arbitration_force_norm_estimate: phys_engine
                        .last_correction_packet_arbitration_force_norm_estimate,
                    correction_packet_prompt_top_k_override: phys_engine
                        .current_prompt_top_k_override,
                    correction_packet_prompt_top_k_match_substring: phys_engine
                        .current_prompt_top_k_match_substring
                        .clone(),
                    correction_packet_suppress_for_current_prompt: phys_engine
                        .correction_packet_suppress_for_current_prompt,
                    correction_packet_prompt_source_target_override: phys_engine
                        .current_prompt_source_target_override
                        .clone(),
                    correction_packet_effective_fire_top_k: phys_engine
                        .last_correction_packet_effective_fire_top_k,
                    correction_packet_effective_pull_avg: phys_engine
                        .last_correction_packet_effective_pull_avg,
                    correction_packet_unfold_active: phys_engine
                        .last_correction_packet_unfold_active,
                    correction_packet_vq_encode_error: phys_engine
                        .last_correction_packet_vq_encode_error,
                    correction_packet_unfold_factor_applied: phys_engine
                        .last_correction_packet_unfold_factor_applied,
                    correction_packet_competence_factor: phys_engine
                        .last_correction_packet_competence_factor,
                    correction_packet_competence_entropy: phys_engine.last_sampling_entropy_norm,
                    correction_packet_min_target_distance: phys_engine
                        .last_correction_packet_min_target_distance,
                    trajectory_classified: phys_engine.trajectory_classified.clone(),
                    codec_active_for_current_prompt: if phys_engine
                        .codec_active_prompt_substrings
                        .is_empty()
                    {
                        None
                    } else {
                        Some(phys_engine.codec_active_for_current_prompt)
                    },
                    trajectory_fire_count_running_mean: if phys_engine.trajectory_fire_count_samples
                        > 0
                    {
                        phys_engine.trajectory_fire_count_sum
                            / phys_engine.trajectory_fire_count_samples as f32
                    } else {
                        0.0
                    },
                    trajectory_fire_count_samples: phys_engine.trajectory_fire_count_samples as u32,
                    ..TokenPhysics::default()
                };
                if phys_engine.tda_shadow_monitor_enabled {
                    phys_engine.update_tda_shadow_monitor(&mut token_trace);
                }
                // Per-token telemetry: gate the serde_json work behind sink presence so
                // an eval with telemetry disabled doesn't pay the serialization cost on
                // every decode token. Two big serde_json calls saved per token when both
                // sinks are off (typical eval mode). Was running 1k×2 = 2000 serializations
                // per 1k-token eval just for nothing.
                let want_stdout = phys_engine.stdout_telemetry();
                let want_ui = args.ui_events_json;
                if want_stdout {
                    let telemetry_json =
                        serde_json::to_string(&token_trace).unwrap_or_else(|_| "{}".to_string());
                    println!("[TELEMETRY] {}", telemetry_json);
                }
                if want_ui {
                    emit_ui_event_value(
                        true,
                        "token_telemetry",
                        serde_json::to_value(&token_trace)
                            .unwrap_or_else(|_| serde_json::json!({})),
                    );
                }
                if phys_engine.stdout_debug() {
                    println!(
                        "[Steering] Status: {} | Gate: {:.2} | Grav: {:.2} | GhostPre: {:.2} | GhostGain: {:.2} | GhostApplied: {:.2} | LiveBasins: {} | Bridge: {} | OrgProm: {} | RecProm: {} | RestProm: {} | BasinDist: {:.3} | BasinRadius: {:.3} | BasinPressure: {:.2} | Goal: {:.2} | Repel: {:.2} | MOTIF_PULL: {:.2} | RECOVERY_PULL: {:.2} | Absence: {:.2} | Trap: {:.2} | Empathy: {:.2} | HiddenReq: {:.3}@{} | Fired: {} | Guardrail: {}",
                        phys_engine.last_engine_status.as_str(),
                        phys_engine.last_activation_gate,
                        phys_engine.last_gravity_mag,
                        phys_engine.last_ghost_pre_norm,
                        phys_engine.last_ghost_gain,
                        phys_engine.last_applied_ghost_mag,
                        phys_engine.last_live_motif_count,
                        motif_provenance.bridge_count,
                        motif_provenance.organic_promoted_count,
                        motif_provenance.recovered_promoted_count,
                        motif_provenance.restored_compact_count,
                        phys_engine.last_live_motif_distance,
                        phys_engine.last_live_motif_radius,
                        phys_engine.last_live_basin_pressure,
                        phys_engine.last_goal_mag,
                        phys_engine.last_repulsion_mag,
                        phys_engine.last_motif_mag,
                        phys_engine.last_recovery_mag,
                        phys_engine.last_absence_signal,
                        phys_engine.last_trap_score,
                        phys_engine.empathy_spike,
                        phys_engine.last_hidden_request_pressure,
                        phys_engine
                            .hidden_request_candidate
                            .map(|req| req.as_str())
                            .unwrap_or("-"),
                        phys_engine
                            .last_hidden_request
                            .map(|req| req.as_str())
                            .unwrap_or("-"),
                        if phys_engine.last_guardrail_active { "on" } else { "off" }
                    );
                }
                phys_engine.last_force_trace = Some(token_trace.clone());
                if phys_engine.stdout_debug() || phys_engine.stdout_telemetry() {
                    std::io::stdout().flush()?;
                }

                if math_governor_relief_active {
                    phys_engine.braking = true;
                    phys_engine.physics_blend = 0.0;
                    phys_engine.dynamic_repulsion = 0.0;
                    phys_engine.adrenaline = 0.0;
                    phys_engine.defibrillator_active = false;
                } else {
                    phys_engine.heartbeat_tick(&token_trace);
                }
                metric_audit.record(&phys_engine);
                mistake_reflex_retry_tokens_remaining =
                    mistake_reflex_retry_tokens_remaining.saturating_sub(1);
                if step % 10 == 0 {
                    let motif_provenance =
                        summarize_runtime_motif_provenance(&phys_engine.runtime_motifs);
                    if phys_engine.stdout_debug() {
                        println!(
                            "[HEARTBEAT] stress={:.2} boredom={:.2} blend={:.2} rep={:.2} defib={} hidden_req_count={}",
                            phys_engine.stress_level,
                            phys_engine.boredom_level,
                            phys_engine.physics_blend,
                            phys_engine.dynamic_repulsion,
                            phys_engine.defibrillator_active,
                            phys_engine.hidden_request_activations
                        );
                    }
                    emit_ui_event_value(
                        args.ui_events_json,
                        "heartbeat",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "stress": phys_engine.stress_level,
                            "boredom": phys_engine.boredom_level,
                            "physics_blend": phys_engine.physics_blend,
                            "dynamic_repulsion": phys_engine.dynamic_repulsion,
                            "defibrillator_active": phys_engine.defibrillator_active,
                            "hidden_request_activations": phys_engine.hidden_request_activations,
                        }),
                    );
                    emit_ui_event_value(
                        args.ui_events_json,
                        "motif_snapshot",
                        serde_json::json!({
                            "turn_index": turn_index,
                            "step": step,
                            "motif_provenance": motif_provenance,
                            "top_motifs": runtime_motif_briefs(&phys_engine.runtime_motifs, 3),
                        }),
                    );
                }
                if phys_engine.stdout_debug() && step % 16 == 0 {
                    print_metric_summary_line(step, &phys_engine);
                }

                cognitive_log.push(token_trace);
            }

            model.append_token(next_token_id);
            phys_engine.current_sentence_tokens.push(next_token_id);
            input = Tensor::new(&[next_token_id], &device)?.unsqueeze(0)?;

            let txt = model
                .tokenizer()
                .decode(&[next_token_id], true)
                .unwrap_or_default();
            let is_bound = phys_engine.sentence_boundary_reached(&txt, next_token_id);

            let h_last = live_hidden_1d.clone();

            phys_engine
                .current_sentence_embeddings
                .push(h_last.detach());
            if phys_engine.secret_sauce_capture_enabled(
                step,
                effective_max_steps,
                finalization_stop_reason.is_some(),
            ) {
                let vector_64d = compress_hidden_state_to_64d(&h_last)?;
                let sentence_anchor_64 =
                    if let Some(sentence_mean) = phys_engine.current_sentence_mean()? {
                        compress_tensor_to_dim(&sentence_mean, 64)?
                    } else {
                        vector_64d.clone()
                    };
                let momentum_16 = if let Some(momentum) = &phys_engine.momentum_buffer {
                    compress_tensor_to_dim(momentum, 16)?
                } else {
                    vec![0.0; 16]
                };
                let promoted_export = phys_engine.compact_runtime_motif_anchor()?;
                let (promoted_sentence_32, promoted_signal) =
                    promoted_export.unwrap_or_else(|| (vec![0.0; 32], 0.0));
                let segments = SecretSauceSegments {
                    hidden_64: sentence_anchor_64.clone(),
                    sentence_32: promoted_sentence_32,
                    momentum_16,
                    scalar_8: vec![
                        phys_engine.last_motif_mag,
                        phys_engine.last_recovery_mag,
                        phys_engine.last_absence_signal,
                        phys_engine.last_trap_score,
                        phys_engine.stress_level,
                        phys_engine.boredom_level,
                        phys_engine.dynamic_gravity,
                        phys_engine.dynamic_repulsion,
                    ],
                    control_8: vec![
                        phys_engine.physics_blend,
                        if phys_engine.last_guardrail_active {
                            1.0
                        } else {
                            -1.0
                        },
                        if phys_engine.orbital_active {
                            1.0
                        } else {
                            -1.0
                        },
                        phys_engine.request_count as f32,
                        match phys_engine.runtime_mode {
                            RuntimeMode::Research => -1.0,
                            RuntimeMode::Agency => 0.0,
                            RuntimeMode::Clean => 1.0,
                        },
                        phys_engine.insight_persistence as f32,
                        phys_engine.empathy_spike,
                        promoted_signal,
                    ],
                };
                if promoted_signal > 0.0 {
                    final_secret_sauce = Some(encode_secret_sauce_v2(&segments)?);
                    final_secret_sauce_version = Some(SecretSauceVersion::V2);
                } else {
                    final_secret_sauce = Some(encode_secret_sauce_v3(&sentence_anchor_64)?);
                    final_secret_sauce_version = Some(SecretSauceVersion::V3);
                }
                final_secret_sauce_segments = Some(segments);
                final_hidden_capture = Some(sentence_anchor_64);

                if let Some(task_anchor) = phys_engine.current_task_anchor_signature.as_ref() {
                    if phys_engine.task_anchor_window_tokens_seen < TASK_ANCHOR_BIND_TOKENS {
                        let similarity =
                            cosine_similarity_slices(task_anchor, &vector_64d).max(0.0);
                        phys_engine.task_anchor_similarity_24tok = similarity;
                        phys_engine.task_anchor_window_tokens_seen += 1;
                        phys_engine.task_anchor_drift =
                            (phys_engine.task_anchor_similarity_start - similarity).abs();
                        if phys_engine.task_anchor_window_tokens_seen == TASK_ANCHOR_BIND_TOKENS {
                            phys_engine.update_task_anchor_similarity_snapshot("24tok");
                        }
                    }
                }
            }

            if phys_engine.secret_sauce_steps_remaining > 0 {
                phys_engine.secret_sauce_steps_remaining -= 1;
            }
            if phys_engine.motif_restore_bias_steps_remaining > 0 {
                phys_engine.motif_restore_bias_steps_remaining -= 1;
                phys_engine.motif_restore_bias_strength =
                    (phys_engine.motif_restore_bias_strength * 0.992).clamp(0.0, 1.0);
            }
            if phys_engine.reentry_clamp_steps_remaining > 0 {
                phys_engine.reentry_clamp_steps_remaining -= 1;
                phys_engine.reentry_clamp_strength =
                    (phys_engine.reentry_clamp_strength * 0.994).clamp(0.0, 1.0);
                phys_engine.reentry_temp_scale =
                    (phys_engine.reentry_temp_scale + 0.008).clamp(0.38, 1.0);
            } else {
                phys_engine.reentry_clamp_strength = 0.0;
                phys_engine.reentry_temp_scale = 1.0;
            }
            if phys_engine.structured_resume_window_remaining > 0 {
                phys_engine.structured_resume_window_remaining -= 1;
            }
            if phys_engine.motif_regression_assist_steps_remaining > 0 {
                phys_engine.motif_regression_assist_steps_remaining -= 1;
                phys_engine.motif_regression_assist_strength =
                    (phys_engine.motif_regression_assist_strength * 0.993).clamp(0.0, 1.0);
            }
            if phys_engine.secret_sauce_steps_remaining == 0 {
                phys_engine.clear_secret_sauce_priors();
            } else if is_bound && phys_engine.secret_sauce_steps_remaining > 0 {
                phys_engine.clear_secret_sauce_priors();
            }

            if (is_bound) && !phys_engine.current_sentence_embeddings.is_empty() {
                // FIX: "The Radioactive Period"
                // Exclude the last token embedding (punctuation) from the Physics Vector
                // This prevents the "." from kicking the model into garbage space.
                let count = phys_engine.current_sentence_embeddings.len();
                let mut embeddings_slice = &phys_engine.current_sentence_embeddings[..];

                if count > 1 {
                    if txt.trim().ends_with('.')
                        || txt.trim().ends_with('!')
                        || txt.trim().ends_with('?')
                    {
                        embeddings_slice = &phys_engine.current_sentence_embeddings[0..count - 1];
                    }
                }

                let stack = Tensor::cat(embeddings_slice, 0)?;
                let dim = h_last.dim(0)?;
                let effective_count = embeddings_slice.len();
                // reshape to [effective_count, dim]
                // If effective_count is 0 (e.g. only period), use whole stack?
                let stack_reshaped = if effective_count > 0 {
                    stack.reshape((effective_count, dim))?
                } else {
                    // Fallback for single period case
                    Tensor::cat(&phys_engine.current_sentence_embeddings, 0)?
                        .reshape((count, dim))?
                };

                let mean = stack_reshaped.mean(0)?;
                let mean_norm = mean.broadcast_div(&mean.sqr()?.sum_all()?.sqrt()?)?;

                let m_coh = phys_engine.compute_m_coh(&mean_norm).unwrap_or(0.5);
                let unique_tokens = phys_engine
                    .current_sentence_tokens
                    .iter()
                    .collect::<HashSet<_>>()
                    .len();
                let m_struct = unique_tokens as f32 / count as f32;
                let m_quantum = phys_engine.compute_quantum_coherence(&mean_norm)?;
                let m_geometric = phys_engine.compute_geometric_score(&mean_norm)?;
                let sub_particles = phys_engine.generate_sub_particles(count)?;

                let m_emo = if let Some(solver) = &phys_engine.symbolic_solver {
                    solver.solve_emo_equation(&mean_norm)?
                } else {
                    0.0
                };

                let full_text = model
                    .tokenizer()
                    .decode(&phys_engine.current_sentence_tokens, true)
                    .unwrap_or_default();

                let total_mass = phys_engine.compute_total_mass(
                    1.0,
                    1.0,
                    m_coh,
                    m_struct,
                    m_quantum,
                    m_geometric,
                    m_emo,
                    0.0,
                );

                let p = SentenceParticle {
                    position: mean_norm.detach(),
                    velocity: Tensor::zeros(dim, DType::F32, &device)?,
                    mass: total_mass,
                    radius: 0.1,
                    birth_step: step,
                    token_count: count,
                    vad: [0.5, 0.5, 0.5],
                    surprisal: 1.0,
                    delta: 0.0,
                    m_info: 1.0,
                    m_sem: 1.0,
                    m_coh,
                    m_struct,
                    m_quantum,
                    m_geometric,
                    m_emo,
                    kl_delta: 0.0,
                    text: full_text,
                    entangled_with: BTreeMap::new(),
                    quantum_state: mean_norm.detach(),
                    latent_thought: Some(latent.detach()),
                    fitness: (m_coh + m_quantum + m_geometric) / 3.0,
                    is_attractor: false,
                    is_repulsor: false,
                    sub_particles,
                    is_lpm_active: true,
                };

                phys_engine.sentence_history.push_back(p);

                let last_idx = phys_engine.sentence_history.len() - 1;
                for i in 0..last_idx {
                    let sim = {
                        let p_last = &phys_engine.sentence_history[last_idx];
                        let p_other = &phys_engine.sentence_history[i];
                        phys_engine.compute_similarity(&p_last.position, &p_other.position)?
                    };

                    if sim > 0.8 {
                        let _ = phys_engine.entangle_particles(last_idx, i);
                    }
                }

                if phys_engine.stdout_debug() {
                    println!(
                        " PARTICLE SPAWNED: {} tokens, mass={:.2}, Q={:.2}, G={:.2}",
                        count, total_mass, m_quantum, m_geometric
                    );
                }
                if !phys_engine.ablate_live_motifs {
                    let _ = phys_engine.mint_or_update_live_motif_from_last_sentence();
                }
                phys_engine.current_sentence_embeddings.clear();
                phys_engine.current_sentence_tokens.clear();
            }

            if let Some(delta) = phys_engine.last_deltas.get(&24) {
                let scale = Tensor::new(0.1f32, delta.device()).unwrap();
                phys_engine.momentum_buffer = Some(delta.broadcast_mul(&scale)?.detach());
                if let Some(lpm) = &mut phys_engine.lpm_collaborator {
                    lpm.inject_priors(delta)?;
                }
            }

            if let Some(reason) = finalization_stop_reason.take() {
                if phys_engine.stdout_debug() || phys_engine.stdout_telemetry() {
                    println!(" [FINALIZATION] stop_triggered reason={reason}");
                    std::io::stdout().flush()?;
                }
                turn_stop_reason = reason;
                break;
            }
        }
        run_timing.add_decode_total_ms(elapsed_ms(decode_started));
        run_timing.add_decode_tokens(cognitive_log.len());
        run_timing.set_stop_reason(turn_stop_reason.clone());
        if turn_stop_reason == "gmms_preserve_earned_before_drift"
            || turn_stop_reason == "gmms_earned_boundary_sentence"
            || turn_stop_reason == "reflex_preserve_earned_before_drift"
        {
            if let Some(byte_len) = reflex_preserve_byte_len {
                if byte_len <= assistant_text.len() {
                    assistant_text.truncate(byte_len);
                }
            }
        } else if turn_stop_reason == "mistake_memory_accepted_answer" {
            if let Some(byte_len) = mistake_memory_preserve_byte_len {
                if byte_len <= assistant_text.len() {
                    assistant_text.truncate(byte_len);
                }
            }
        }

        if !phys_engine.stdout_profile.chat_enabled() {
            print_metric_postmortem(&metric_audit, phys_engine.goal_embedding.is_some());
        }
        let exact_form_repair = apply_exact_form_completion_repair(
            resolved_output_contract_mode,
            user_prompt,
            &compact_resume_state,
            assistant_text.trim(),
        );
        if exact_form_repair.applied {
            eprintln!(
                " [OUTPUT_CONTRACT] repair_applied={} mode={} turn={}",
                exact_form_repair.source,
                resolved_output_contract_mode.as_str(),
                turn_index
            );
            assistant_text = exact_form_repair.text.clone();
        }
        let collaborative_hygiene = apply_collaborative_transparency_hygiene(
            resolved_output_contract_mode,
            user_prompt,
            assistant_text.trim(),
        );
        if collaborative_hygiene.applied {
            eprintln!(
                " [COLLAB_HYGIENE] assistant_removed={} repeated_requests_removed={} correction_tail_truncated={} partial_control_fragment_removed={} turn={}",
                collaborative_hygiene.assistant_surfaces_removed,
                collaborative_hygiene.repeated_request_surfaces_removed,
                collaborative_hygiene.correction_tail_truncated,
                collaborative_hygiene.partial_control_fragment_removed,
                turn_index
            );
            assistant_text = collaborative_hygiene.text.clone();
        }
        let agency_hands = apply_agency_hands(
            resolved_output_contract_mode,
            user_prompt,
            assistant_text.trim(),
            &mut agency_hands_state,
        );
        if agency_hands.applied {
            eprintln!(
                " [AGENCY_HANDS] lock={} remembers_added={} evicted={} tail_truncated={} learning_event={} turn={}",
                agency_hands.lock_payload.as_deref().unwrap_or("none"),
                agency_hands.remembers_added,
                agency_hands.evicted_remembers,
                agency_hands.tail_truncated,
                agency_hands.learning_event_recorded,
                turn_index
            );
            assistant_text = agency_hands.text.clone();
        }
        let packet_agency_transition = detect_packet_agency_transition(
            assistant_text.trim(),
            agency_hands.lock_payload.as_deref(),
            !agency_hands.accepted_remember_payloads.is_empty(),
        );
        // Mint REMEMBER-derived correction packets when --correction-packets-out is set.
        // The user (or model under user direction) emits `[REQUEST: REMEMBER] payload`;
        // the agency-hands store accepts payloads (deduped); we mint one packet per
        // accepted payload tying the REMEMBER text to the codec-bucket of the probe at
        // end-of-turn. Closes the "user as the living correction signal" North Star
        // primitive.
        if !agency_hands.accepted_remember_payloads.is_empty()
            && phys_engine.correction_packets_out.is_some()
        {
            match phys_engine.mint_remember_correction_packets(
                &agency_hands.accepted_remember_payloads,
                user_prompt,
                args.req_id.as_str(),
                packet_agency_transition.as_deref(),
            ) {
                Ok(n) if n > 0 => {
                    eprintln!(
                        " [CORRECTION_PACKET] Minted {} REMEMBER-derived packet(s) at turn {}",
                        n, turn_index
                    );
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!(
                        " [CORRECTION_PACKET] Failed to mint REMEMBER packet(s): {}",
                        e
                    );
                }
            }
        }

        // Mint LOCK-derived "earned answer" correction packet when --correction-packets-out
        // is set and a LOCK was emitted this turn. Higher pull_strength than REMEMBER so
        // future drift back into the earned probe's bucket gets pulled harder toward the
        // locked answer state. Closes the "preserve earned answers before drift" North
        // Star primitive.
        if let Some(lock_payload) = agency_hands.lock_payload.as_deref() {
            // Revalidate any previously-invalidated packets matching THIS LOCK's
            // exact-payload hash. If the user is re-affirming a payload they had
            // earlier contradicted ("never mind, the original was right"), the
            // previously-invalidated earned packet fires again. Runs BEFORE the
            // invalidation step so a normal LOCK that's not a contradiction
            // always gets a chance to revalidate.
            #[cfg(feature = "niodv4_bridge")]
            if phys_engine.correction_packet_revalidate_on_affirmation {
                if let Some(store) = phys_engine.correction_packets.as_ref() {
                    let lock_hash = hash_str(lock_payload.trim());
                    let revalidated = store.revalidate_by_lh_hash(&lock_hash);
                    if revalidated > 0 {
                        eprintln!(
                            " [CORRECTION_PACKET] Revalidated {} prior packet(s) matching reaffirmed lh_{} at turn {}",
                            revalidated, lock_hash, turn_index
                        );
                    }
                }
            }

            // Invalidate any already-loaded correction packets minted from the
            // contradicted prior LOCK before minting the corrected packet. The
            // user's "I changed my mind" signal switches the old basin OFF while
            // §10y's multiplier boosts the new basin's pull. Two-sided dominance.
            #[cfg(feature = "niodv4_bridge")]
            if agency_hands.learning_event_recorded
                && phys_engine.correction_packet_invalidate_on_contradiction
            {
                if let (Some(prior_payload), Some(store)) = (
                    agency_hands.contradicted_lock_payload.as_deref(),
                    phys_engine.correction_packets.as_ref(),
                ) {
                    let prior_trimmed = prior_payload.trim();
                    let prior_hash = hash_str(prior_trimmed);
                    let exact_invalidated = store.invalidate_by_lh_hash(&prior_hash);
                    // Semantic invalidation: also switch off any packet sharing the
                    // contradicted payload's KEY (e.g. `final` from `final=ship_x`),
                    // even when the value differs. Catches semantic contradictions
                    // beyond exact-string match.
                    let prior_key = agency_payload_key(prior_trimmed);
                    let semantic_invalidated = if !prior_key.is_empty() {
                        store.invalidate_by_payload_key(&prior_key)
                    } else {
                        0
                    };
                    if exact_invalidated > 0 || semantic_invalidated > 0 {
                        eprintln!(
                            " [CORRECTION_PACKET] Invalidated {} exact-hash + {} semantic-key packet(s) for contradicted '{}' at turn {}",
                            exact_invalidated, semantic_invalidated, prior_trimmed, turn_index
                        );
                    }
                }
            }
            if phys_engine.correction_packets_out.is_some() {
                let pull_multiplier = if agency_hands.learning_event_recorded {
                    // Compute key from the contradicted prior payload (same source as
                    // §10aa's semantic-key invalidation). Increment the per-key count
                    // and pick the scaled multiplier — escalating user frustration
                    // surfaces as a progressively stronger steering reflex, capped by
                    // --correction-packet-adaptive-contradiction-cap.
                    let key = agency_hands
                        .contradicted_lock_payload
                        .as_deref()
                        .map(|p| agency_payload_key(p.trim()))
                        .unwrap_or_default();
                    if !key.is_empty() {
                        phys_engine.record_contradiction_for_key(&key)
                    } else {
                        phys_engine.correction_packet_lock_contradiction_multiplier
                    }
                } else {
                    1.0
                };
                let is_contradiction = pull_multiplier > 1.0 + 1e-6;
                match phys_engine.mint_lock_correction_packet(
                    lock_payload,
                    user_prompt,
                    args.req_id.as_str(),
                    pull_multiplier,
                    packet_agency_transition.as_deref(),
                ) {
                    Ok(true) => {
                        if is_contradiction {
                            eprintln!(
                                " [CORRECTION_PACKET] Minted CONTRADICTION-LOCK packet at turn {} (pull = {} × {} = {})",
                                turn_index,
                                phys_engine.correction_packet_lock_pull_strength,
                                pull_multiplier,
                                phys_engine.correction_packet_lock_pull_strength * pull_multiplier
                            );
                        } else {
                            eprintln!(
                                " [CORRECTION_PACKET] Minted LOCK-derived earned-answer packet at turn {}",
                                turn_index
                            );
                        }
                    }
                    Ok(false) => {}
                    Err(e) => {
                        eprintln!(" [CORRECTION_PACKET] Failed to mint LOCK packet: {}", e);
                    }
                }
            }
        }
        // §10bg per-turn capture: bypass the agency-hands gate and mint a
        // live_capture packet from the end-of-turn probe regardless of LOCK
        // echoing. Iter-66 found the agency-hands gate prevents bucket
        // diversification when prompts don't reliably elicit LOCK echoes.
        // This flag fires for every turn so non-counting prompts also produce
        // packets, paired with --correction-packet-mint-bucket-cap to keep
        // diversity from devolving to a single bucket.
        #[cfg(feature = "niodv4_bridge")]
        if phys_engine.correction_packet_capture_every_turn
            && phys_engine.correction_packets_out.is_some()
        {
            match phys_engine.flush_correction_packet_capture(
                user_prompt,
                args.req_id.as_str(),
                packet_agency_transition.as_deref(),
            ) {
                Ok(true) => {
                    eprintln!(
                        " [CORRECTION_PACKET] Per-turn live_capture packet minted (turn {})",
                        turn_index
                    );
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!(" [CORRECTION_PACKET] Per-turn live_capture failed: {}", e);
                }
            }
        }
        let mistake_memory_outcome_changed =
            if args.mistake_memory_path.is_some() && !mistake_memory_matches.is_empty() {
                mistake_memory.record_outcome(&mistake_memory_matches, assistant_text.trim())
            } else {
                false
            };
        if mistake_memory_outcome_changed {
            if let Some(path) = &args.mistake_memory_path {
                mistake_memory.save(path)?;
            }
        }
        let mistake_guard_final = mistake_guard.snapshot();
        let mistake_reflex_guard_final = mistake_reflex_guard.snapshot();
        let mistake_reflex_outcome_changed =
            if args.mistake_reflex_path.is_some() && !mistake_reflex_matches.is_empty() {
                mistake_reflex_memory
                    .record_outcome(&mistake_reflex_matches, &mistake_reflex_guard_final)
            } else {
                false
            };
        if mistake_reflex_outcome_changed {
            if let Some(path) = &args.mistake_reflex_path {
                mistake_reflex_memory.save(path)?;
            }
        }
        if mistake_memory_prompt_applied || mistake_guard_final.blocked_lock {
            emit_ui_event_value(
                args.ui_events_json,
                "mistake_memory",
                serde_json::json!({
                    "turn_index": turn_index,
                    "phase": "outcome",
                    "prompt_applied": mistake_memory_prompt_applied,
                    "match_count": mistake_guard_final.match_count,
                    "event_ids": mistake_guard_final.event_ids,
                    "rejected_answer_seen": mistake_guard_final.rejected_answer_seen,
                    "accepted_answer_seen": mistake_guard_final.accepted_answer_seen,
                    "guard_blocked_lock": mistake_guard_final.blocked_lock,
                    "guard_blocked_count": mistake_guard_final.blocked_count,
                    "ledger_updated": mistake_memory_outcome_changed,
                }),
            );
        }
        if mistake_reflex_prompt_applied
            || !mistake_reflex_matches.is_empty()
            || mistake_reflex_guard_final.blocked_lock
        {
            emit_ui_event_value(
                args.ui_events_json,
                "mistake_reflex",
                serde_json::json!({
                    "turn_index": turn_index,
                    "phase": "outcome",
                    "mode": format!("{:?}", args.mistake_reflex_mode).to_ascii_lowercase(),
                    "prompt_applied": mistake_reflex_prompt_applied,
                    "match_count": mistake_reflex_guard_final.match_count,
                    "event_ids": mistake_reflex_guard_final.event_ids,
                    "domains": mistake_reflex_guard_final.domains,
                    "action_level": mistake_reflex_guard_final.action_level,
                    "resolution_level": mistake_reflex_guard_final.resolution_level,
                    "vector_slice_available": mistake_reflex_guard_final.vector_slice_available,
                    "unicode_packet_ids": mistake_reflex_guard_final.unicode_packet_ids,
                    "route_preserved": mistake_reflex_guard_final.route_preserved,
                    "unfold_reason": mistake_reflex_guard_final.unfold_reason,
                    "decay_reason": mistake_reflex_guard_final.decay_reason,
                    "evidence_seen": mistake_reflex_guard_final.evidence_seen,
                    "old_mistake_seen": mistake_reflex_guard_final.old_mistake_seen,
                    "earned_answer_seen": mistake_reflex_guard_final.earned_answer_seen,
                    "earned_answer_text": mistake_reflex_guard_final.earned_answer_text,
                    "guard_blocked_lock": mistake_reflex_guard_final.blocked_lock,
                    "guard_blocked_count": mistake_reflex_guard_final.blocked_count,
                    "ledger_updated": mistake_reflex_outcome_changed,
                }),
            );
        }
        let output_contract_violation_reason =
            output_contract_violation(resolved_output_contract_mode, assistant_text.trim());
        if let Some(reason) = output_contract_violation_reason {
            eprintln!(
                " [OUTPUT_CONTRACT] violation={} mode={} turn={}",
                reason,
                resolved_output_contract_mode.as_str(),
                turn_index
            );
        }
        emit_ui_event_value(
            args.ui_events_json,
            "output_contract",
            serde_json::json!({
                "turn_index": turn_index,
                "configured_mode": args.output_contract_mode.as_str(),
                "resolved_mode": resolved_output_contract_mode.as_str(),
                "prompt_applied": output_contract_prompt_applied,
                "effective_max_steps": effective_max_steps,
                "repair_applied": exact_form_repair.applied,
                "repair_source": exact_form_repair.source,
                "exact_output_marker_count": exact_output_marker_count(assistant_text.trim()),
                "violation": output_contract_violation_reason,
                "exact_block_clean": output_contract_violation_reason.is_none(),
            }),
        );
        emit_ui_event_value(
            args.ui_events_json,
            "collaborative_hygiene",
            serde_json::json!({
                "turn_index": turn_index,
                "resolved_mode": resolved_output_contract_mode.as_str(),
                "applied": collaborative_hygiene.applied,
                "assistant_surfaces_removed": collaborative_hygiene.assistant_surfaces_removed,
                "repeated_request_surfaces_removed": collaborative_hygiene.repeated_request_surfaces_removed,
                "correction_tail_truncated": collaborative_hygiene.correction_tail_truncated,
                "partial_control_fragment_removed": collaborative_hygiene.partial_control_fragment_removed,
            }),
        );
        emit_ui_event_value(
            args.ui_events_json,
            "agency_hands",
            serde_json::json!({
                "turn_index": turn_index,
                "resolved_mode": resolved_output_contract_mode.as_str(),
                "reinjection_applied": agency_state_prompt.is_some(),
                "applied": agency_hands.applied,
                "lock_payload": agency_hands.lock_payload,
                "active_lock": agency_hands_state.active_lock.as_deref(),
                "remember_count": agency_hands_state.remembers.len(),
                "remembers_added": agency_hands.remembers_added,
                "evicted_remembers": agency_hands.evicted_remembers,
                "tail_truncated": agency_hands.tail_truncated,
                "learning_event_recorded": agency_hands.learning_event_recorded,
                "learning_event_count": agency_hands_state.learning_events.len(),
            }),
        );
        update_compact_resume_state_from_turn(
            &mut compact_resume_state,
            user_prompt,
            assistant_text.trim(),
            resolved_output_contract_mode,
        );
        emit_ui_event_value(
            args.ui_events_json,
            "compact_resume_state",
            serde_json::json!({
                "turn_index": turn_index,
                "turn_count": compact_resume_state.turn_count,
                "anchor_count": compact_resume_state.anchor_count(),
                "names": compact_resume_state.names.len(),
                "constraints": compact_resume_state.constraints.len(),
                "deadlines": compact_resume_state.deadlines.len(),
                "preference_flags": compact_resume_state.preference_flags.len(),
                "unresolved_questions": compact_resume_state.unresolved_questions.len(),
                "requested_output_shape": compact_resume_state.requested_output_shape.len(),
                "prior_results": compact_resume_state.prior_results.len(),
                "corrections": compact_resume_state.corrections.len(),
            }),
        );
        emit_ui_event_value(
            args.ui_events_json,
            "turn_end",
            serde_json::json!({
                "turn_index": turn_index,
                "assistant_text": assistant_text.trim(),
                "token_count": cognitive_log.len(),
                "secret_sauce_version": final_secret_sauce_version.map(|version| version.as_str()),
                "resolved_output_contract_mode": resolved_output_contract_mode.as_str(),
                "output_contract_violation": output_contract_violation_reason,
            }),
        );

        if let Some(output_dir) = &args.turn_capture_dir {
            let write_kv_snapshot = args.turn_kv_every > 0
                && (((turn_index + 1) % args.turn_kv_every) == 0
                    || (turn_index + 1) == session_prompt_count);
            let motif_artifact = write_turn_capture_artifacts(
                &args,
                scaling_profile.as_ref(),
                &universe.source_description,
                restored_kv_active,
                turn_index,
                turn_index + 1,
                user_prompt,
                assistant_text.trim(),
                final_hidden_capture.as_ref(),
                final_secret_sauce_segments.as_ref(),
                final_secret_sauce_version.clone(),
                final_secret_sauce.as_deref(),
                hidden_dim,
                // The turn runner closure owns mutable access during generation.
                // By this point generation is complete, and artifact emission only
                // needs a read-only snapshot of the final state for this turn.
                unsafe { *index_pos_ptr },
                unsafe { &*model_ptr },
                unsafe { &*phys_engine_ptr },
                turn_previous_motif_continuity_artifact.as_ref(),
                restored_reference_motifs.as_ref(),
                initial_restored_motif_provenance.as_ref(),
                output_dir,
                write_kv_snapshot,
            )?;
            turn_previous_motif_continuity_artifact = Some(motif_artifact);
        }

        Ok((
            assistant_text.trim().to_string(),
            cognitive_log,
            final_hidden_capture,
            final_secret_sauce_segments,
            final_secret_sauce_version,
            final_secret_sauce,
        ))
    };

    if args.chat_repl {
        eprintln!(" [CHAT] ready. Type /quit or /exit to stop.");
        let stdin = std::io::stdin();
        let mut turn_index = 0usize;
        loop {
            eprint!("you> ");
            std::io::stderr().flush()?;
            let mut user_prompt = String::new();
            if stdin.read_line(&mut user_prompt)? == 0 {
                break;
            }
            let user_prompt = user_prompt.trim().to_string();
            if user_prompt.is_empty() {
                continue;
            }
            if matches!(user_prompt.as_str(), "/quit" | "/exit") {
                break;
            }
            eprint!("niodoo> ");
            std::io::stderr().flush()?;
            let (
                assistant_text,
                cognitive_log,
                final_hidden_capture,
                final_secret_sauce_segments,
                final_secret_sauce_version,
                final_secret_sauce,
            ) = run_assistant_turn(
                turn_index,
                &user_prompt,
                turn_index == 0,
                last_assistant_output.as_deref(),
            )?;
            if chat_output {
                println!();
                std::io::stdout().flush()?;
            }
            if let Some(secret_sauce) = &final_secret_sauce {
                if !chat_output {
                    println!(
                        " [SECRET_SAUCE] turn={} version={} len={} state_mapped=\"{}\"",
                        turn_index,
                        final_secret_sauce_version
                            .map(|version| version.as_str())
                            .unwrap_or("unknown"),
                        secret_sauce.chars().count(),
                        secret_sauce
                    );
                }
            }
            last_assistant_output = Some(assistant_text.clone());
            session_records.push((
                user_prompt,
                assistant_text,
                cognitive_log,
                final_hidden_capture,
                final_secret_sauce_segments,
                final_secret_sauce_version,
                final_secret_sauce,
            ));
            turn_index += 1;
        }
    } else {
        for (turn_index, user_prompt) in session_prompts.iter().enumerate() {
            let (
                assistant_text,
                cognitive_log,
                final_hidden_capture,
                final_secret_sauce_segments,
                final_secret_sauce_version,
                final_secret_sauce,
            ) = run_assistant_turn(
                turn_index,
                user_prompt,
                turn_index == 0,
                last_assistant_output.as_deref(),
            )?;
            if let Some(secret_sauce) = &final_secret_sauce {
                if !chat_output {
                    println!(
                        " [SECRET_SAUCE] turn={} version={} len={} state_mapped=\"{}\"",
                        turn_index,
                        final_secret_sauce_version
                            .map(|version| version.as_str())
                            .unwrap_or("unknown"),
                        secret_sauce.chars().count(),
                        secret_sauce
                    );
                }
            }
            last_assistant_output = Some(assistant_text.clone());
            session_records.push((
                user_prompt.clone(),
                assistant_text,
                cognitive_log,
                final_hidden_capture,
                final_secret_sauce_segments,
                final_secret_sauce_version,
                final_secret_sauce,
            ));
        }
    }
    if let Some(path) = &args.compact_resume_state_save_file {
        // DEEP_DIVE_ROADMAP P1-C: copy the engine's current task anchor
        // signature into the compact resume state before saving. On load,
        // the next turn will use this vector instead of re-hashing the new
        // prompt text — preventing the cross-turn task-payload shear noted
        // in `2026-05-02_138-inertial-task-anchoring`. Empty vec means
        // "no anchor was set this turn"; backward-compatible with old
        // saved files (the field's `skip_serializing_if = "Vec::is_empty"`).
        if let Some(sig) = phys_engine.current_task_anchor_signature.as_ref() {
            compact_resume_state.task_anchor_vector = sig.clone();
        }
        save_compact_resume_state(path, &compact_resume_state)?;
        if chat_output {
            eprintln!(
                " [COMPACT_RESUME] saved={} turns={} anchors={}",
                path.display(),
                compact_resume_state.turn_count,
                compact_resume_state.anchor_count()
            );
        } else {
            println!(
                " [COMPACT_RESUME] saved={} turns={} anchors={}",
                path.display(),
                compact_resume_state.turn_count,
                compact_resume_state.anchor_count()
            );
        }
    }

    if chat_output {
        eprintln!(
            "\n=== CHAT SESSION COMPLETE ===\nTurns: {}",
            session_records.len()
        );
    } else if session_records.len() == 1 {
        println!("\n=== SIMULATION COMPLETE ===");
    } else {
        println!("\n=== SESSION COMPLETE ===");
        println!("Turns: {}", session_records.len());
    }
    if !chat_output {
        println!(
            "Steps/turn: {}, Particles: {}, Total Q: {:.4}, Total G: {:.4}",
            args.max_steps,
            phys_engine.sentence_history.len(),
            phys_engine.compute_total_quantum().unwrap_or(0.0),
            phys_engine.compute_total_geometric().unwrap_or(0.0)
        );
    }
    let motif_provenance = summarize_runtime_motif_provenance(&phys_engine.runtime_motifs);
    if !chat_output {
        println!(
            " [MOTIF_PROVENANCE] bridge={} live={} organic_promoted={} recovered_promoted={} restored_compact={}",
            motif_provenance.bridge_count,
            motif_provenance.live_count,
            motif_provenance.organic_promoted_count,
            motif_provenance.recovered_promoted_count,
            motif_provenance.restored_compact_count
        );
    }
    let motif_carry_forward = restored_reference_motifs
        .as_ref()
        .and_then(|restored| summarize_motif_carry_forward(restored, &phys_engine.runtime_motifs));
    let hinge_summary = build_motif_hinge_summary(
        restored_kv_active,
        initial_restored_motif_provenance.as_ref(),
        &motif_provenance,
        &phys_engine,
    );
    let mut motif_continuity_artifact = MotifContinuityArtifact {
        version: "motif_continuity_v2".to_string(),
        runtime_mode: format!("{:?}", args.runtime_mode),
        restored_run: restored_kv_active,
        max_steps: args.max_steps,
        motif_provenance: motif_provenance.clone(),
        hinge: hinge_summary.clone(),
        routing: MotifRoutingSummary::default(),
        task_anchor: TaskAnchorSummary::default(),
        motif_carry_forward: motif_carry_forward.clone(),
        comparison_to_previous: None,
    };
    if let Some(previous) = &previous_motif_continuity_artifact {
        let comparison = compare_motif_continuity(previous, &motif_continuity_artifact);
        if !chat_output {
            println!(
                " [MOTIF_REGRESSION] verdict={} carry_delta={:.3} sim_delta={:.3} organic_delta={} recovered_delta={} restored_delta={}",
                comparison.verdict,
                comparison.carry_forward_delta,
                comparison.mean_similarity_delta,
                comparison.organic_promoted_delta,
                comparison.recovered_promoted_delta,
                comparison.restored_compact_delta
            );
        }
        motif_continuity_artifact.comparison_to_previous = Some(comparison);
    }
    if let Some(summary) = &motif_carry_forward {
        if !chat_output {
            println!(
                " [MOTIF_CONTINUITY] restored_promoted={} final_promoted={} exact_id={} semantic={} mean_sim={:.3} carry={:.3}",
                summary.restored_promoted_count,
                summary.final_promoted_count,
                summary.exact_id_matches,
                summary.semantic_matches,
                summary.mean_best_similarity,
                summary.carry_forward_ratio
            );
        }
    }
    if !chat_output {
        println!(
            " [MOTIF_HINGE] flipped={} organic={} recovered={} attempts={} failures={} organic_timing={} recovered_timing={}",
            hinge_summary.hinge_flipped,
            hinge_summary.organic_promoted_observed,
            hinge_summary.recovered_promoted_observed,
            hinge_summary.promotion_attempt_count,
            hinge_summary.promotion_failure_count,
            hinge_summary
                .organic_promoted_timing
                .as_deref()
                .unwrap_or("none"),
            hinge_summary
                .recovered_promoted_timing
                .as_deref()
                .unwrap_or("none")
        );
    }
    let human_eval_assistant_preview = session_records
        .last()
        .map(|(_, assistant_text, _, _, _, _, _)| assistant_text.chars().take(200).collect())
        .unwrap_or_default();
    let human_eval_last_prompt = session_records
        .last()
        .map(|(prompt, _, _, _, _, _, _)| prompt.as_str())
        .unwrap_or("");
    let human_eval_last_assistant = session_records
        .last()
        .map(|(_, assistant_text, _, _, _, _, _)| assistant_text.as_str())
        .unwrap_or("");
    let hinge_correlation = detect_pattern_task_correlation(
        human_eval_last_prompt,
        human_eval_last_assistant,
        &hinge_summary,
    );
    let routing_summary = build_motif_routing_summary(
        &phys_engine,
        &hinge_summary,
        &hinge_correlation,
        previous_motif_continuity_artifact
            .as_ref()
            .map(|artifact| &artifact.routing),
    );
    let task_anchor_summary = build_task_anchor_summary(&phys_engine);
    if !chat_output {
        println!(
            " [ROUTING] ticks={} structured={} structured_candidate={} conversational={} tie_breaks={} structured_locks={} escalations={}/{} wrong_basin={} improved={}",
            routing_summary.controller_tick_count,
            routing_summary.controller_selected_structured_count,
            routing_summary.controller_selected_structured_candidate_count,
            routing_summary.controller_selected_conversational_count,
            routing_summary.conflict_tie_break_count,
            routing_summary.structured_basin_lock_count,
            routing_summary.structured_candidate_escalation_wins,
            routing_summary.structured_candidate_escalation_attempts,
            routing_summary.wrong_basin_lock_suspected,
            routing_summary.routing_improved_vs_previous
        );
    }
    motif_continuity_artifact.routing = routing_summary.clone();
    motif_continuity_artifact.task_anchor = task_anchor_summary.clone();
    let human_eval_artifact = build_human_test_eval_artifact(
        &args,
        scaling_profile.as_ref(),
        &universe.source_description,
        restored_kv_active,
        session_records.len(),
        human_eval_assistant_preview,
        &motif_provenance,
        &hinge_summary,
        &routing_summary,
        &hinge_correlation,
        &task_anchor_summary,
        motif_carry_forward.as_ref(),
        motif_continuity_artifact
            .comparison_to_previous
            .as_ref()
            .map(|comparison| comparison.verdict.as_str()),
    );
    let hinge_window_artifact = build_hinge_window_artifact(
        &args,
        restored_kv_active,
        &hinge_summary,
        &task_anchor_summary,
        &phys_engine,
    );
    if !chat_output {
        println!(
            " [HUMAN_EVAL] model_size={} mode={} source={} turns={} flags={}",
            human_eval_artifact.model_size,
            human_eval_artifact.runtime_mode,
            human_eval_artifact.embedding_source,
            human_eval_artifact.turn_count,
            if human_eval_artifact.review_flags.is_empty() {
                "none".to_string()
            } else {
                human_eval_artifact.review_flags.join(",")
            }
        );
    }
    emit_ui_event_value(
        args.ui_events_json,
        "session_summary",
        serde_json::json!({
            "turn_count": session_records.len(),
            "embedding_source": human_eval_artifact.embedding_source.clone(),
            "assistant_preview": human_eval_artifact.assistant_preview.clone(),
            "runtime_mode": human_eval_artifact.runtime_mode.clone(),
            "review_flags": human_eval_artifact.review_flags.clone(),
            "continuity_verdict": human_eval_artifact.continuity_verdict.clone(),
            "hinge": human_eval_artifact.hinge.clone(),
            "routing": human_eval_artifact.routing.clone(),
            "hinge_correlation": human_eval_artifact.hinge_correlation.clone(),
            "task_anchor": human_eval_artifact.task_anchor.clone(),
            "hinge_window": {
                "first_promotion_attempt_step": hinge_window_artifact.first_promotion_attempt_step,
                "first_hinge_step": hinge_window_artifact.first_hinge_step,
                "neutral_basin_occupancy": hinge_window_artifact.neutral_basin_occupancy,
                "structured_candidate_separation": hinge_window_artifact.structured_candidate_separation,
                "record_count": hinge_window_artifact.records.len(),
            },
            "motif_provenance": motif_provenance.clone(),
            "motif_carry_forward": motif_carry_forward.clone(),
            "top_motifs": runtime_motif_briefs(&phys_engine.runtime_motifs, 3),
        }),
    );

    if let Some(path) = &args.state_capture_file {
        if let Some((
            turn_index,
            (
                _prompt,
                assistant_text,
                _cognitive_log,
                Some(vector_64d),
                Some(segments),
                Some(secret_sauce_version),
                secret_sauce,
            ),
        )) = session_records.iter().enumerate().rev().find(
            |(_, (_, _, _, capture, segments, version, _))| {
                capture.is_some() && segments.is_some() && version.is_some()
            },
        ) {
            let packet_secret_sauce = Some(build_state_packet_secret_sauce(
                *secret_sauce_version,
                secret_sauce.clone().unwrap_or_default(),
                segments.clone(),
                vector_64d.clone(),
            ));
            let captured_state_packet =
                StatePacket::capture(&phys_engine, hidden_dim, packet_secret_sauce.clone())?;
            let capture = StateCaptureRecord {
                turn_index,
                token_count: assistant_text.split_whitespace().count(),
                hidden_dim,
                compressed_dim: vector_64d.len(),
                compression: "block_mean_4096_to_64_dry_run".to_string(),
                secret_sauce_version: secret_sauce_version.as_str().to_string(),
                unicode_string: secret_sauce.clone().unwrap_or_default(),
                segments: segments.clone(),
                vector_64d: vector_64d.clone(),
                assistant_preview: assistant_text.chars().take(160).collect(),
                state_packet: Some(captured_state_packet.clone()),
                motif_provenance: Some(motif_provenance.clone()),
                motif_carry_forward: motif_carry_forward.clone(),
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create state capture parent {}", parent.display())
                })?;
            }
            std::fs::write(path, serde_json::to_string_pretty(&capture)?)
                .with_context(|| format!("Failed to write state capture {}", path.display()))?;
            if chat_output {
                eprintln!(
                    " [STATE_CAPTURE] wrote={} turn={} dim={} compression={}",
                    path.display(),
                    turn_index,
                    capture.compressed_dim,
                    capture.compression
                );
            } else {
                println!(
                    " [STATE_CAPTURE] wrote={} turn={} dim={} compression={}",
                    path.display(),
                    turn_index,
                    capture.compressed_dim,
                    capture.compression
                );
            }
            if let Some(codec_trace_artifact) = build_codec_trace_artifact(
                path,
                args.runtime_mode.as_str(),
                turn_index,
                capture.token_count,
                hidden_dim,
                &captured_state_packet,
                vector_64d,
                &capture.unicode_string,
                &hinge_summary,
                &routing_summary,
                &task_anchor_summary,
                &hinge_window_artifact,
            ) {
                write_codec_trace_artifact(path, &codec_trace_artifact)?;
            }
            write_motif_continuity_artifact(path, &motif_continuity_artifact)?;
            write_human_test_eval_artifact(path, &human_eval_artifact)?;
            write_hinge_window_artifact(path, &hinge_window_artifact)?;
        }
    }

    if let Some(path) = &args.kv_state_save_file {
        let assistant_preview = session_records
            .last()
            .map(|(_, assistant_text, _, _, _, _, _)| {
                assistant_text.chars().take(160).collect::<String>()
            })
            .unwrap_or_default();
        let packet_secret_sauce = session_records.last().and_then(
            |(_, _, _, capture, segments, version, secret_sauce)| match (
                capture.as_ref(),
                segments.as_ref(),
                *version,
                secret_sauce.as_ref(),
            ) {
                (Some(vector_64d), Some(segments), Some(version), Some(secret_sauce)) => {
                    Some(build_state_packet_secret_sauce(
                        version,
                        secret_sauce.clone(),
                        segments.clone(),
                        vector_64d.clone(),
                    ))
                }
                _ => None,
            },
        );
        let snapshot = KvStateRecord {
            version: "kv_state_v1".to_string(),
            runtime_mode: format!("{:?}", args.runtime_mode),
            index_pos,
            hidden_dim,
            assistant_preview,
            kv_cache: model.export_kv_cache_snapshot()?,
            state_packet: Some(StatePacket::capture(
                &phys_engine,
                hidden_dim,
                packet_secret_sauce,
            )?),
            engine_state: Some(EngineStateSnapshot::capture(&phys_engine, hidden_dim)?),
            motif_provenance: Some(motif_provenance.clone()),
            motif_carry_forward: motif_carry_forward.clone(),
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create kv state parent {}", parent.display())
            })?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&snapshot)?)
            .with_context(|| format!("Failed to write kv state {}", path.display()))?;
        if chat_output {
            eprintln!(
                " [KV_STATE] wrote={} index_pos={} layers={} engine_state=yes motif_provenance=yes motif_carry={}",
                path.display(),
                snapshot.index_pos,
                snapshot.kv_cache.layers.len(),
                if snapshot.motif_carry_forward.is_some() {
                    "yes"
                } else {
                    "no"
                }
            );
        } else {
            println!(
                " [KV_STATE] wrote={} index_pos={} layers={} engine_state=yes motif_provenance=yes motif_carry={}",
                path.display(),
                snapshot.index_pos,
                snapshot.kv_cache.layers.len(),
                if snapshot.motif_carry_forward.is_some() {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        if let (
            Some(state_packet),
            Some((turn_index, (_, assistant_text, _, Some(vector_64d), _, _, secret_sauce))),
        ) = (
            snapshot.state_packet.as_ref(),
            session_records
                .iter()
                .enumerate()
                .rev()
                .find(|(_, (_, _, _, capture, _, _, _))| capture.is_some()),
        ) {
            if let Some(codec_trace_artifact) = build_codec_trace_artifact(
                path,
                args.runtime_mode.as_str(),
                turn_index,
                assistant_text.split_whitespace().count(),
                hidden_dim,
                state_packet,
                vector_64d,
                secret_sauce.as_deref().unwrap_or_default(),
                &hinge_summary,
                &routing_summary,
                &task_anchor_summary,
                &hinge_window_artifact,
            ) {
                write_codec_trace_artifact(path, &codec_trace_artifact)?;
            }
        }
        write_motif_continuity_artifact(path, &motif_continuity_artifact)?;
        write_human_test_eval_artifact(path, &human_eval_artifact)?;
        write_hinge_window_artifact(path, &hinge_window_artifact)?;
    }

    if let Some(path) = &args.human_eval_summary_file {
        write_human_test_eval_artifact(path, &human_eval_artifact)?;
    }

    for (
        turn_index,
        (prompt, assistant_text, cognitive_log, _capture, _segments, _version, _secret_sauce),
    ) in session_records.iter().enumerate()
    {
        if phys_engine.stdout_debug() {
            let trace = CognitiveTrace {
                prompt: prompt.clone(),
                tokens: cognitive_log.clone(),
                config: format!(
                    "Mode: {:?} | Sigma: {:.3} | Blend: {:.3} | Repulsion: {:.3} | Layers: {}-{} | PressureGate: {:.1}-{:.1} | VisibleRequestGate: {} | WobbleThreshold: {:.1} | Temp: {:.3}",
                    args.runtime_mode,
                    args.sigma,
                    args.physics_blend,
                    args.repulsion_strength,
                    args.physics_start_layer,
                    args.physics_end_layer,
                    NIODOO_PRESSURE_GATE_START,
                    NIODOO_PRESSURE_GATE_FULL,
                    args.visible_request_gate,
                    NIODOO_WOBBLE_PRESSURE_THRESHOLD,
                    args.temperature
                ),
            };
            eprintln!("\n===COGNITIVE_TRACE TURN {}===", turn_index);
            eprintln!(
                "{}",
                serde_json::to_string(&trace).unwrap_or_else(|_| "{}".to_string())
            );
        }
        if !matches!(
            phys_engine.stdout_profile,
            StdoutProfile::Quiet | StdoutProfile::Chat
        ) {
            println!("\n=== TURN {} OUTPUT ===\n{}", turn_index, assistant_text);
        }
    }

    // Save route telemetry to JSONL
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    args.prompt.as_bytes().hash(&mut hasher);
    let prompt_hash = format!("{:x}", hasher.finish());

    let req_id = args.req_id.as_str();

    let output_dir = Path::new("artifacts");
    let output_path = args.telemetry_out.clone().unwrap_or_else(|| {
        output_dir.join(format!(
            "route_telemetry_{}.jsonl",
            if phys_engine.bridge_enabled {
                "on"
            } else {
                "off"
            }
        ))
    });
    eprintln!(" [TELEMETRY] Saving to {:?}", output_path);

    // Create directory if it doesn't exist
    let _ = std::fs::create_dir_all(output_dir);

    // Collect all token records from all turns
    let all_token_records: Vec<TokenPhysics> = session_records
        .iter()
        .flat_map(|(_, _, tokens, _, _, _, _)| tokens.clone())
        .collect();

    let active_context_startup_telemetry_record_for_run = if args.active_context_startup_telemetry {
        let path = args.active_context_adapter_decisions.as_ref().context(
            "--active-context-startup-telemetry requires --active-context-adapter-decisions",
        )?;
        let decisions = load_runtime_adapter_decisions(path)?;
        let summary = summarize_runtime_adapter_decisions(&decisions);
        let diagnostic = runtime_metadata_diagnostic(&summary);
        Some(runtime_startup_telemetry_record(diagnostic))
    } else {
        None
    };

    let telemetry_write_started = timing_now();
    phys_engine.save_route_telemetry(
        req_id,
        &prompt_hash,
        output_path.to_str().unwrap(),
        &all_token_records,
        args.telemetry_profile,
        active_context_startup_telemetry_record_for_run.as_ref(),
    );
    run_timing.add_telemetry_write_ms(elapsed_ms(telemetry_write_started));

    if let Some(path) = &args.answer_logit_probe_out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let probe_write_started = timing_now();
        let mut file = File::create(path)
            .with_context(|| format!("failed to create answer logit probe file {:?}", path))?;
        for record in &answer_logit_probe_records {
            serde_json::to_writer(&mut file, record)
                .with_context(|| format!("failed to write answer logit probe row to {:?}", path))?;
            file.write_all(b"\n")?;
        }
        run_timing.add_telemetry_write_ms(elapsed_ms(probe_write_started));
        eprintln!(
            " [ANSWER_LOGIT_PROBE] Saved {} rows to {:?}",
            answer_logit_probe_records.len(),
            path
        );
    }

    if phys_engine.bridge_gate34_latch_active() {
        let candidate_dump_started = timing_now();
        let candidates_path = output_dir.join(format!("{}_candidates.json", req_id));
        let _ = std::fs::write(
            &candidates_path,
            serde_json::to_string_pretty(&phys_engine.gate34_acquisition_candidates)
                .unwrap_or_else(|_| "[]".to_string()),
        );
        run_timing.add_candidate_dump_ms(elapsed_ms(candidate_dump_started));
        eprintln!(" [GATE34] Saved candidates to {:?}", candidates_path);
    }

    let timing_path = run_timing_path(args.telemetry_out.as_deref(), output_dir, req_id);
    write_run_timing(&timing_path, &run_timing.snapshot())?;
    eprintln!(" [TIMING] Saving to {:?}", timing_path);

    // Mint a CorrectionPacket from the final probe state (when --correction-packets-out is set).
    // The "preserve correction across fresh process boundaries" North Star primitive: the next
    // run loads this file via --correction-packets-path and the captured probe-bucket fires.
    match phys_engine.flush_correction_packet_capture(&args.prompt, req_id, None) {
        Ok(true) => {
            if let Some(path) = phys_engine.correction_packets_out.as_ref() {
                eprintln!(
                    " [CORRECTION_PACKET] Appended live-capture packet to {:?}",
                    path
                );
            }
        }
        Ok(false) => {}
        Err(e) => {
            eprintln!(" [CORRECTION_PACKET] Failed to write packet: {}", e);
        }
    }

    // Persist contradiction counts so the adaptive multiplier in §10ac
    // accumulates across process boundaries (§10ad). Atomic rewrite.
    if let Some(path) = phys_engine.correction_contradiction_counts_path.as_ref() {
        match write_contradiction_counts(path, &phys_engine.contradiction_counts) {
            Ok(n) => {
                eprintln!(
                    " [CONTRADICTION_COUNTS] Persisted {} key(s) to {:?}",
                    n, path
                );
            }
            Err(e) => {
                eprintln!(" [CONTRADICTION_COUNTS] Failed to persist counts: {}", e);
            }
        }
    }

    // Evict long-decayed packets before the state-out write (§10ai). Earned
    // packets (decay_rate=1.0) are immune; only scaffolding with high fire_count
    // and a configured non-zero floor qualifies for removal.
    #[cfg(feature = "niodv4_bridge")]
    if phys_engine.correction_packet_eviction_floor > 0.0 {
        if let Some(store) = phys_engine.correction_packets.as_mut() {
            let evicted = store.evict_below_floor(
                phys_engine.correction_packet_decay_rate,
                phys_engine.correction_packet_eviction_floor,
            );
            if evicted > 0 {
                eprintln!(
                    " [CORRECTION_PACKET] Evicted {} packet(s) below effective_pull ratio {} (engine_decay={})",
                    evicted,
                    phys_engine.correction_packet_eviction_floor,
                    phys_engine.correction_packet_decay_rate
                );
            }
        }
    }

    // Persist the loaded correction-packet store with current fire counters
    // (when --correction-packets-state-out is set). This makes decay and unfold
    // dynamics survive process boundaries — the "scar tissue accumulating across
    // sessions" North Star semantics. Atomic rewrite via temp-file + rename.
    #[cfg(feature = "niodv4_bridge")]
    if let Some(path) = phys_engine.correction_packets_state_out.as_ref() {
        if let Some(store) = phys_engine.correction_packets.as_ref() {
            match store.write_to_jsonl(path) {
                Ok(n) => {
                    eprintln!(
                        " [CORRECTION_PACKET] Persisted {} packets with current counters to {:?}",
                        n, path
                    );
                }
                Err(e) => {
                    eprintln!(" [CORRECTION_PACKET] Failed to persist state: {}", e);
                }
            }
        } else {
            eprintln!(
                " [CORRECTION_PACKET] --correction-packets-state-out set but no store loaded; skipping"
            );
        }
    }

    // Phase 6 instrumentation: print per-stage forward_physics breakdown if the
    // user set NIODOO_PROFILE_FORWARD=1. No-op (single AtomicBool check) when unset.
    niodoo::physics::forward_profile::dump_forward_profile();

    Ok(())
}
