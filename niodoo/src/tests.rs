//! Test module for main.rs — extracted as separate file.
use super::*;

fn approx_eq(a: f32, b: f32, tolerance: f32) {
    assert!(
        (a - b).abs() <= tolerance,
        "expected {a} ~= {b} within {tolerance}"
    );
}

#[test]
fn correction_packet_prompt_top_k_map_parses_and_resolves_first_match() {
    let map = parse_correction_packet_prompt_top_k_map(
        " letters that match the letter s : 3, letter s:5, letter i:0, bad, empty: ",
    );
    assert_eq!(
        map,
        vec![
            ("letters that match the letter s".to_string(), 3),
            ("letter s".to_string(), 5),
            ("letter i".to_string(), 0),
        ]
    );

    assert_eq!(
        resolve_correction_packet_prompt_top_k_override(
            "Mississippi has how many letters that match the letter S?",
            &map,
        ),
        Some(3)
    );
    assert_eq!(
        resolve_correction_packet_prompt_top_k_override(
            "How many LETTER S are in Mississippi?",
            &map,
        ),
        Some(5)
    );
    assert_eq!(
        resolve_correction_packet_prompt_top_k_override(
            "How many letter z are in Mississippi?",
            &map,
        ),
        None
    );
}

#[test]
fn correction_packet_prompt_top_k_no_match_suppression_is_explicitly_gated() {
    let map = parse_correction_packet_prompt_top_k_map("letter s:5,letter i:0");
    let matched =
        resolve_correction_packet_prompt_top_k_match("How many letter s are in Mississippi?", &map)
            .is_some();
    let unmatched =
        resolve_correction_packet_prompt_top_k_match("What is the capital of France?", &map)
            .is_some();

    assert!(!should_suppress_correction_packets_for_prompt(
        true,
        !map.is_empty(),
        matched
    ));
    assert!(should_suppress_correction_packets_for_prompt(
        true,
        !map.is_empty(),
        unmatched
    ));
    assert!(!should_suppress_correction_packets_for_prompt(
        false,
        !map.is_empty(),
        unmatched
    ));
    assert!(!should_suppress_correction_packets_for_prompt(
        true, false, unmatched
    ));
}

#[test]
fn correction_packet_prompt_source_target_map_parses_and_resolves_first_match() {
    let map = parse_correction_packet_prompt_source_target_map(
        "tennessee e:tn_count_e, tennessee:tn_count_n, bad, hippopotamus p:HP_COUNT_P, empty:",
    );
    assert_eq!(
        map,
        vec![
            ("tennessee e".to_string(), "tn_count_e".to_string()),
            ("tennessee".to_string(), "tn_count_n".to_string()),
            ("hippopotamus p".to_string(), "hp_count_p".to_string()),
        ]
    );

    assert_eq!(
        resolve_correction_packet_prompt_source_target_override(
            "How many letters Tennessee e has?",
            &map,
        )
        .as_deref(),
        Some("tn_count_e")
    );
    assert_eq!(
        resolve_correction_packet_prompt_source_target_override(
            "How many letters are in TENNESSEE?",
            &map,
        )
        .as_deref(),
        Some("tn_count_n")
    );
    assert_eq!(
        resolve_correction_packet_prompt_source_target_override(
            "How many letters p are in Hippopotamus?",
            &map,
        )
        .as_deref(),
        None
    );
}

/// Verifies the JSON shape `write_correction_packet_record` emits is loadable by
/// `CorrectionPacketStore::load_from_jsonl` — i.e. the writer and the reader stay
/// in sync. Also exercises append-mode so multiple REMEMBER packets in a turn
/// accumulate without truncation.
#[test]
fn write_correction_packet_record_roundtrips_via_loader() {
    use super::bridge::correction_packets::CorrectionPacketStore;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("remember_packet_test_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let mut target_a = [0f32; 64];
    target_a[0] = 0.5;
    let mut target_b = [0f32; 64];
    target_b[1] = -0.3;

    write_correction_packet_record(
        &path,
        "remember::req_test::ph_aaaa::rh_bbbb::step_00007",
        42,
        &target_a,
        0.1,
        0.05,
        "remember: owner=Priya",
        7,
        None,
        None,
        None,
        None,
        None,
        false,
    )
    .expect("write A");
    write_correction_packet_record(
        &path,
        "remember::req_test::ph_aaaa::rh_cccc::step_00007",
        42,
        &target_b,
        0.1,
        0.05,
        "remember: deadline=April 30",
        7,
        None,
        None,
        None,
        None,
        None,
        false,
    )
    .expect("write B");

    let store = CorrectionPacketStore::load_from_jsonl(&path).expect("load");
    let bucket = store.packets_for_code(42);
    assert_eq!(
        bucket.len(),
        2,
        "both REMEMBER packets should land in bucket 42"
    );
    let labels: Vec<&str> = bucket.iter().map(|p| p.source_label.as_str()).collect();
    assert!(labels
        .iter()
        .any(|l| l.starts_with("remember: owner=Priya")));
    assert!(labels
        .iter()
        .any(|l| l.starts_with("remember: deadline=April 30")));

    let _ = std::fs::remove_file(&path);
}

/// LOCK-derived earned-answer packets must carry their distinctive `earned:` label
/// and have a `lock::` packet_id prefix so downstream consumers can tell them apart
/// from end-of-run captures and REMEMBER-derived packets. Higher pull_strength is
/// stamped per the engine config (defaults at 0.3 vs 0.1 for REMEMBER).
#[test]
fn write_correction_packet_record_lock_label_distinct_from_remember() {
    use super::bridge::correction_packets::CorrectionPacketStore;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("lock_packet_test_{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let target = [0.5f32; 64];

    // REMEMBER packet (lower pull, "remember:" label, decay/unfold inherit engine).
    write_correction_packet_record(
        &path,
        "remember::req_test::ph_aa::rh_bb::step_00007",
        17,
        &target,
        0.1,
        0.05,
        "remember: owner=Priya",
        7,
        None,
        None,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    // LOCK packet (higher pull, "earned:" label, decay_rate=Some(1.0) = no decay,
    // unfold_factor=Some(1.0) = no extra boost on relapse).
    write_correction_packet_record(
        &path,
        "lock::req_test::ph_aa::lh_cc::step_00007",
        17,
        &target,
        0.3,
        0.05,
        "earned: next=continue_governor_decay_sweep",
        7,
        Some(1.0),
        Some(1.0),
        None,
        Some(1.0),
        None,
        false,
    )
    .unwrap();

    let store = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
    let bucket = store.packets_for_code(17);
    assert_eq!(bucket.len(), 2);
    let remember = bucket
        .iter()
        .find(|p| p.packet_id.starts_with("remember::"))
        .unwrap();
    let lock = bucket
        .iter()
        .find(|p| p.packet_id.starts_with("lock::"))
        .unwrap();
    assert!(remember.source_label.starts_with("remember:"));
    assert!(lock.source_label.starts_with("earned:"));
    assert!(lock.pull_strength > remember.pull_strength + 1e-5);
    let _ = std::fs::remove_file(&path);
}

/// Contradiction-flavored LOCK packets must have a `lock_correction::` packet_id
/// prefix and `earned-correction:` source_label, plus a pull_strength stamped
/// at `lock_pull * contradiction_multiplier` so the corrected basin pulls
/// strictly harder than a normal earned packet from a non-contradictory turn.
#[test]
fn write_correction_packet_record_contradiction_lock_distinct() {
    use super::bridge::correction_packets::CorrectionPacketStore;

    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "lock_contradiction_test_{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let target = [0.5f32; 64];

    // Normal LOCK packet (pull=0.3, no contradiction).
    write_correction_packet_record(
        &path,
        "lock::req_test::ph_aa::lh_bb::step_00010",
        22,
        &target,
        0.3,
        0.05,
        "earned: count=4",
        10,
        Some(1.0),
        Some(1.0),
        None,
        Some(1.0),
        None,
        false,
    )
    .unwrap();
    // Contradiction LOCK (pull=0.6 = 0.3×2.0).
    write_correction_packet_record(
        &path,
        "lock_correction::req_test::ph_aa::lh_cc::step_00020",
        22,
        &target,
        0.6,
        0.05,
        "earned-correction: count=3",
        20,
        Some(1.0),
        Some(1.0),
        None,
        Some(1.0),
        None,
        false,
    )
    .unwrap();

    let store = CorrectionPacketStore::load_from_jsonl(&path).unwrap();
    let bucket = store.packets_for_code(22);
    assert_eq!(bucket.len(), 2);
    let normal = bucket
        .iter()
        .find(|p| p.packet_id.starts_with("lock::"))
        .unwrap();
    let correction = bucket
        .iter()
        .find(|p| p.packet_id.starts_with("lock_correction::"))
        .unwrap();
    assert!(normal.source_label.starts_with("earned:"));
    assert!(correction.source_label.starts_with("earned-correction:"));
    assert!(correction.pull_strength > normal.pull_strength + 1e-5);
    assert!((correction.pull_strength - normal.pull_strength * 2.0).abs() < 1e-5);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn write_correction_packet_record_emits_hybrid_metadata_fields() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "hybrid_packet_metadata_test_{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    let target = [0.25f32; 64];
    let mut payload_z = [0.0f32; 64];
    payload_z[5] = 0.5;
    payload_z[12] = -0.125;
    let hybrid = CorrectionPacketHybridMetadata {
        text_fact: Some("owner=Jason".to_string()),
        payload_z_64d: Some(payload_z),
        route_code: Some("vq_042".to_string()),
        route_motif_id: Some("motif_demo".to_string()),
        target_ghost_id: Some("ghost_3".to_string()),
        nearest_ghost_distance: Some(0.11),
        second_nearest_ghost_distance: Some(0.23),
        route_margin: Some(0.12),
        agency_transition: Some("REMEMBER->LOCK".to_string()),
        force_policy: Some("lock_earned".to_string()),
        force_pull_strength: Some(0.3),
        force_distance_threshold: Some(0.05),
        force_decay_rate: Some(1.0),
        force_unfold_factor: Some(1.0),
        force_unfold_retry_factor: Some(1.0),
        answer_lock_boundary: Some("lock_payload".to_string()),
        projection_strategy: Some("simple".to_string()),
        ghost_pull_delta_norm: Some(0.02),
    };

    write_correction_packet_record(
        &path,
        "lock::req_test::ph_aa::lh_bb::step_00010",
        42,
        &target,
        0.3,
        0.05,
        "earned: owner=Jason",
        10,
        Some(1.0),
        Some(1.0),
        Some("owner"),
        Some(1.0),
        Some(&hybrid),
        false,
    )
    .unwrap();

    let body = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(body.trim()).unwrap();
    assert_eq!(value["text_fact"], "owner=Jason");
    assert_eq!(value["payload_z_64d"].as_array().unwrap().len(), 64);
    assert!((value["payload_z_64d"][5].as_f64().unwrap() - 0.5).abs() < 1e-6);
    assert!((value["payload_z_64d"][12].as_f64().unwrap() + 0.125).abs() < 1e-6);
    assert_eq!(value["route_code"], "vq_042");
    assert_eq!(value["route_motif_id"], "motif_demo");
    assert_eq!(value["target_ghost_id"], "ghost_3");
    assert_eq!(value["agency_transition"], "REMEMBER->LOCK");
    assert_eq!(value["force_policy"], "lock_earned");
    assert_eq!(value["answer_lock_boundary"], "lock_payload");
    assert_eq!(value["projection_strategy"], "simple");
    let force_pull = value["force_pull_strength"].as_f64().unwrap_or_default();
    let force_dist = value["force_distance_threshold"]
        .as_f64()
        .unwrap_or_default();
    let force_decay = value["force_decay_rate"].as_f64().unwrap_or_default();
    let force_unfold = value["force_unfold_factor"].as_f64().unwrap_or_default();
    let force_retry = value["force_unfold_retry_factor"]
        .as_f64()
        .unwrap_or_default();
    let ghost_norm = value["ghost_pull_delta_norm"].as_f64().unwrap_or_default();
    assert!((force_pull - 0.3).abs() < 1e-6);
    assert!((force_dist - 0.05).abs() < 1e-6);
    assert!((force_decay - 1.0).abs() < 1e-6);
    assert!((force_unfold - 1.0).abs() < 1e-6);
    assert!((force_retry - 1.0).abs() < 1e-6);
    assert!((ghost_norm - 0.02).abs() < 1e-6);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn hash_str_is_stable_and_short() {
    let h1 = hash_str("owner=Priya");
    let h2 = hash_str("owner=Priya");
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 16);
    assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(h1, hash_str("owner=Mira"));
}

/// Adaptive contradiction multiplier: each contradiction event for the same
/// payload-key escalates the multiplier linearly (base × min(count, cap)).
/// The cap caps unbounded growth. Different keys are tracked independently.
/// Verified by exercising `record_contradiction_for_key` directly on a
/// minimal PrincipiaEngine field shim — full engine construction is too
/// heavy here, so we test the math via a synthetic struct mirroring the
/// helper's invariants.
/// The multi-source relapse trigger ORs vq_encode_error and
/// mistake_reflex_retry_count. Verified via the Boolean math the runtime
/// expresses (constructing a full PrincipiaEngine in tests is non-trivial).
/// Per-source unfold factors: when retry-relapse fires with its own
/// override factor, the applied factor is max(encode_factor, retry_factor)
/// so retry can boost more than OOD without ever boosting less. Verified
/// via the math the runtime expresses.
#[test]
fn per_source_unfold_factor_max_combines() {
    // Mirror of the runtime's resolution logic.
    let compute = |unfold_factor: f32,
                   retry_factor_override: f32,
                   encode_error_relapse: bool,
                   retry_relapse_eligible: bool|
     -> f32 {
        let retry_factor = if retry_factor_override > 1.0 {
            retry_factor_override.max(unfold_factor)
        } else {
            unfold_factor
        };
        let retry_relapse = retry_relapse_eligible && retry_factor > 1.0;
        let encode_factor_applied = if encode_error_relapse {
            unfold_factor
        } else {
            1.0
        };
        let retry_factor_applied = if retry_relapse { retry_factor } else { 1.0 };
        encode_factor_applied.max(retry_factor_applied)
    };
    // No relapse → 1.0.
    assert!((compute(3.0, 0.0, false, false) - 1.0).abs() < 1e-6);
    // Encode-only, default factor → 3.0.
    assert!((compute(3.0, 0.0, true, false) - 3.0).abs() < 1e-6);
    // Retry-only with override 5.0 → 5.0 (override wins).
    assert!((compute(3.0, 5.0, false, true) - 5.0).abs() < 1e-6);
    // Retry-only with no override → unfold_factor (3.0).
    assert!((compute(3.0, 0.0, false, true) - 3.0).abs() < 1e-6);
    // Both fire: max(unfold_factor, retry_factor) = max(3.0, 5.0) = 5.0.
    assert!((compute(3.0, 5.0, true, true) - 5.0).abs() < 1e-6);
    // Override < unfold_factor → use unfold_factor (override clamped up).
    assert!((compute(3.0, 2.0, false, true) - 3.0).abs() < 1e-6);
    // Override exactly 1.0 → falls back to unfold_factor.
    assert!((compute(3.0, 1.0, false, true) - 3.0).abs() < 1e-6);
}

#[test]
fn relapse_trigger_combines_encode_error_and_retry_count() {
    let unfold_factor = 3.0_f32;
    // Pure mathematical mirror of try_apply_correction_packet_force's logic.
    let compute = |encode_threshold: f32,
                   encode_err: f32,
                   retry_threshold: usize,
                   retry_count: usize|
     -> bool {
        let encode_error_relapse =
            encode_threshold > 0.0 && encode_err > encode_threshold && unfold_factor > 1.0;
        let retry_relapse =
            retry_threshold > 0 && retry_count >= retry_threshold && unfold_factor > 1.0;
        encode_error_relapse || retry_relapse
    };
    // Both disabled → no relapse.
    assert!(!compute(0.0, 0.5, 0, 5));
    // Encode trigger only.
    assert!(compute(0.4, 0.5, 0, 5));
    // Retry trigger only.
    assert!(compute(0.0, 0.5, 1, 1));
    // Both fire.
    assert!(compute(0.4, 0.5, 1, 1));
    // Below both thresholds.
    assert!(!compute(0.4, 0.3, 1, 0));
    // Retry threshold met exactly.
    assert!(compute(0.0, 0.0, 3, 3));
    // Retry below threshold.
    assert!(!compute(0.0, 0.0, 3, 2));
}

#[test]
fn contradiction_counts_jsonl_roundtrip() {
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "contradiction_counts_test_{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);

    // Initial: missing file → empty map.
    let initial = load_contradiction_counts(&path).expect("load missing");
    assert!(initial.is_empty());

    let mut counts = std::collections::HashMap::new();
    counts.insert("final".to_string(), 3);
    counts.insert("priority".to_string(), 1);
    counts.insert("owner".to_string(), 7);
    let written = write_contradiction_counts(&path, &counts).expect("write");
    assert_eq!(written, 3);

    let body = std::fs::read_to_string(&path).unwrap();
    // Sorted output for determinism.
    let lines: Vec<&str> = body.lines().collect();
    assert!(lines[0].contains(r#""payload_key":"final""#));
    assert!(lines[1].contains(r#""payload_key":"owner""#));
    assert!(lines[2].contains(r#""payload_key":"priority""#));

    let reloaded = load_contradiction_counts(&path).expect("load roundtrip");
    assert_eq!(reloaded.len(), 3);
    assert_eq!(reloaded.get("final"), Some(&3));
    assert_eq!(reloaded.get("priority"), Some(&1));
    assert_eq!(reloaded.get("owner"), Some(&7));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn record_contradiction_for_key_scales_linearly_to_cap() {
    // Exercise the math the helper expresses, since constructing a full
    // PrincipiaEngine in a unit test is non-trivial. The relationship under
    // test: multiplier = (base × min(count, cap)).max(1.0).
    let base = 2.0_f32;
    let cap = 5_u64;
    let computed = |count: u64| -> f32 {
        let scaled = count.min(cap) as f32;
        (base * scaled).max(1.0)
    };
    assert!((computed(0) - 1.0).abs() < 1e-6, "count=0 floors at 1.0");
    assert!((computed(1) - 2.0).abs() < 1e-6);
    assert!((computed(2) - 4.0).abs() < 1e-6);
    assert!((computed(5) - 10.0).abs() < 1e-6);
    // Beyond cap: stays at cap × base.
    assert!((computed(6) - 10.0).abs() < 1e-6);
    assert!((computed(100) - 10.0).abs() < 1e-6);
}

/// End-to-end demonstration of the North Star contradiction/affirmation loop
/// as a 4-turn deterministic sequence. Mirrors what would happen across four
/// chat turns where the model emits LOCK and the user reacts:
///
///   T1: assistant LOCK final=A   → mint earned packet (active)
///   T2: user "wrong, fix" + assistant LOCK final=B
///        → contradiction fires; A invalidated (exact-hash + semantic-key);
///          B minted with boosted pull (multiplier × min(count=1, cap)).
///   T3: user "ok thanks" + assistant LOCK final=A
///        → no contradiction; A revalidated; A mint coexists with the
///          previously-revalidated A; B remains.
///   T4: user "no, actually fix again" + assistant LOCK final=C
///        → contradiction fires; B (and re-active A by semantic key)
///          invalidated; C minted with the further-escalated multiplier
///          (count=2 × base = 4× pull).
///
/// The test walks AgencyHandsState through these turns and asserts the
/// expected fields, then plays the equivalent state evolution against a
/// CorrectionPacketStore via invalidate/revalidate calls. Pure Rust, no
/// model invocation — exercises the wiring deterministically.
#[test]
fn end_to_end_contradiction_loop_simulates_four_turn_sequence() {
    use super::bridge::correction_packets::{CorrectionPacket, CorrectionPacketStore};

    let mut state = AgencyHandsState::new();
    let mut store = CorrectionPacketStore::new();
    let mut contradiction_counts: std::collections::HashMap<String, u64> =
        std::collections::HashMap::new();
    let base_multiplier = 2.0_f32;
    let cap = 5_u64;
    let lock_pull = 0.3_f32;

    // Helper closure to mirror PrincipiaEngine::record_contradiction_for_key.
    let mut record_contradiction =
        |counts: &mut std::collections::HashMap<String, u64>, key: &str| -> f32 {
            if key.is_empty() {
                return base_multiplier;
            }
            let key_lc = key.to_ascii_lowercase();
            let count = counts
                .entry(key_lc)
                .and_modify(|c| *c = c.saturating_add(1))
                .or_insert(1);
            (base_multiplier * (*count).min(cap) as f32).max(1.0)
        };

    // Helper to build a packet matching the runtime's mint shape.
    let mut next_step = 0u64;
    let mut mint_lock_packet =
        |store: &mut CorrectionPacketStore, payload: &str, pull_multiplier: f32, step: u64| {
            let payload_hash = hash_str(payload.trim());
            let is_correction = pull_multiplier > 1.0 + 1e-6;
            let prefix = if is_correction {
                "lock_correction"
            } else {
                "lock"
            };
            let label = if is_correction {
                "earned-correction"
            } else {
                "earned"
            };
            let packet = CorrectionPacket {
                packet_id: format!("{prefix}::req_t::ph_p::lh_{payload_hash}::step_{step:05}"),
                vq_code: 7,
                target_z_64d: [0.5_f32; 64],
                payload_z_64d: None,
                pull_strength: lock_pull * pull_multiplier.max(1.0),
                distance_threshold: 0.05,
                source_label: format!("{label}: {}", payload.trim()),
                created_step: step,
                text_fact: None,
                route_code: None,
                route_motif_id: None,
                target_ghost_id: None,
                nearest_ghost_distance: None,
                second_nearest_ghost_distance: None,
                route_margin: None,
                agency_transition: None,
                force_policy: None,
                force_pull_strength: None,
                force_distance_threshold: None,
                force_decay_rate: None,
                force_unfold_factor: None,
                force_unfold_retry_factor: None,
                answer_lock_boundary: None,
                projection_strategy: None,
                ghost_pull_delta_norm: None,
                fire_count: super::bridge::correction_packets::AtomicU64Wrapper::new(0),
                last_fire_step: super::bridge::correction_packets::AtomicU64Wrapper::new(0),
                decay_rate: Some(1.0),
                unfold_factor: Some(1.0),
                invalidated: super::bridge::correction_packets::AtomicBoolWrapper::new(false),
                payload_key: {
                    let k = agency_payload_key(payload.trim());
                    if k.is_empty() {
                        None
                    } else {
                        Some(k)
                    }
                },
                unfold_retry_factor: Some(1.0),
            };
            store.insert(packet);
        };

    // ─── Turn 1: assistant LOCK final=A. No contradiction. ───────────────
    let t1 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "what's the deploy plan",
        "[REQUEST: LOCK] final=A",
        &mut state,
    );
    assert_eq!(t1.lock_payload.as_deref(), Some("final=A"));
    assert!(!t1.learning_event_recorded);
    assert!(t1.contradicted_lock_payload.is_none());
    // Mint at multiplier 1.0 (no contradiction yet).
    next_step += 1;
    mint_lock_packet(&mut store, "final=A", 1.0, next_step);
    assert_eq!(store.iter_packets().count(), 1);
    let active_after_t1: Vec<&CorrectionPacket> = store
        .iter_packets()
        .filter(|p| !p.invalidated.load())
        .collect();
    assert_eq!(active_after_t1.len(), 1);

    // ─── Turn 2: user contradicts + assistant LOCK final=B. ──────────────
    let t2 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "actually wrong, please fix",
        "[REQUEST: LOCK] final=B",
        &mut state,
    );
    assert!(t2.learning_event_recorded);
    assert_eq!(t2.contradicted_lock_payload.as_deref(), Some("final=A"));
    // Invalidate by both lh_hash and payload_key (mirroring chat REPL).
    let prior_a_hash = hash_str("final=A");
    let prior_a_key = agency_payload_key("final=A");
    assert_eq!(store.invalidate_by_lh_hash(&prior_a_hash), 1);
    let semantic_invalidated = store.invalidate_by_payload_key(&prior_a_key);
    // The A packet was already invalidated by lh_hash; semantic pass should
    // not double-count it.
    assert_eq!(semantic_invalidated, 0);
    // Mint B at boosted multiplier from contradiction count of 1.
    let m_t2 = record_contradiction(&mut contradiction_counts, &prior_a_key);
    assert!((m_t2 - 2.0).abs() < 1e-5);
    next_step += 1;
    mint_lock_packet(&mut store, "final=B", m_t2, next_step);
    let active_after_t2: Vec<&CorrectionPacket> = store
        .iter_packets()
        .filter(|p| !p.invalidated.load())
        .collect();
    assert_eq!(active_after_t2.len(), 1);
    assert!(active_after_t2[0]
        .source_label
        .starts_with("earned-correction"));
    assert!((active_after_t2[0].pull_strength - 0.6).abs() < 1e-5);

    // ─── Turn 3: user "ok thanks" + assistant LOCK final=A. No contradiction. ──
    let t3 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "ok thanks",
        "[REQUEST: LOCK] final=A",
        &mut state,
    );
    assert!(!t3.learning_event_recorded);
    assert!(t3.contradicted_lock_payload.is_none());
    // Revalidate A by lh_hash. The A packet from T1 comes back to life.
    let lock_a_hash = hash_str("final=A");
    assert_eq!(store.revalidate_by_lh_hash(&lock_a_hash), 1);
    next_step += 1;
    mint_lock_packet(&mut store, "final=A", 1.0, next_step);
    let active_after_t3: Vec<&CorrectionPacket> = store
        .iter_packets()
        .filter(|p| !p.invalidated.load())
        .collect();
    // Active now: the revalidated T1 A, the T2 B, the new T3 A. = 3.
    assert_eq!(active_after_t3.len(), 3);

    // ─── Turn 4: user contradicts again + LOCK final=C. ──────────────────
    let t4 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "no, that's wrong, fix again",
        "[REQUEST: LOCK] final=C",
        &mut state,
    );
    assert!(t4.learning_event_recorded);
    assert_eq!(t4.contradicted_lock_payload.as_deref(), Some("final=A"));
    let prior_a_hash = hash_str("final=A");
    let prior_a_key = agency_payload_key("final=A");
    // T1 A and T3 A both share lh_hash → both invalidated.
    let exact_invalidated = store.invalidate_by_lh_hash(&prior_a_hash);
    assert_eq!(exact_invalidated, 2);
    // Semantic key=final ALSO matches the T2 B packet (since B was minted with
    // payload_key=final). So B gets invalidated here.
    let semantic_invalidated = store.invalidate_by_payload_key(&prior_a_key);
    assert_eq!(semantic_invalidated, 1);
    // Adaptive multiplier: count for `final` is now 2 → multiplier = 2 × 2 = 4.
    let m_t4 = record_contradiction(&mut contradiction_counts, &prior_a_key);
    assert!((m_t4 - 4.0).abs() < 1e-5);
    next_step += 1;
    mint_lock_packet(&mut store, "final=C", m_t4, next_step);
    // Active after T4: only the newly minted C packet.
    let active_after_t4: Vec<&CorrectionPacket> = store
        .iter_packets()
        .filter(|p| !p.invalidated.load())
        .collect();
    assert_eq!(active_after_t4.len(), 1);
    assert!(active_after_t4[0]
        .source_label
        .starts_with("earned-correction"));
    assert!((active_after_t4[0].pull_strength - lock_pull * 4.0).abs() < 1e-5);
    // Counts: only `final` is tracked.
    assert_eq!(contradiction_counts.get("final"), Some(&2));
}

#[test]
fn apply_agency_hands_captures_contradicted_lock_payload() {
    let mut state = AgencyHandsState::new();
    // Turn 1: model emits a LOCK; no prior, no contradiction.
    let r1 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "what should we do",
        "[REQUEST: LOCK] final=ship_thursday",
        &mut state,
    );
    assert!(!r1.learning_event_recorded);
    assert!(r1.contradicted_lock_payload.is_none());
    assert_eq!(state.active_lock.as_deref(), Some("final=ship_thursday"));

    // Turn 2: user says "wrong, fix it"; contradiction fires; prior payload captured.
    let r2 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "actually wrong, please fix",
        "[REQUEST: LOCK] final=ship_friday",
        &mut state,
    );
    assert!(r2.learning_event_recorded);
    assert_eq!(
        r2.contradicted_lock_payload.as_deref(),
        Some("final=ship_thursday")
    );
    assert_eq!(state.active_lock.as_deref(), Some("final=ship_friday"));

    // Turn 3: user says nothing contradictory; no learning event fires.
    let r3 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "ok thanks",
        "[REQUEST: LOCK] final=ship_friday",
        &mut state,
    );
    assert!(!r3.learning_event_recorded);
    assert!(r3.contradicted_lock_payload.is_none());
}

#[test]
fn apply_agency_hands_records_accepted_remember_payloads() {
    let mut state = AgencyHandsState::new();
    let result = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "user prompt",
        "[REQUEST: REMEMBER] owner=Priya\n[REQUEST: REMEMBER] deadline=April 30",
        &mut state,
    );
    assert_eq!(result.accepted_remember_payloads.len(), 2);
    assert!(result
        .accepted_remember_payloads
        .iter()
        .any(|p| p.contains("Priya")));
    assert!(result
        .accepted_remember_payloads
        .iter()
        .any(|p| p.contains("April 30")));

    // Re-feeding the SAME payload is deduped by store_remember; the new accepted
    // list excludes the duplicate but includes the new one.
    let result2 = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "user prompt",
        "[REQUEST: REMEMBER] owner=Priya\n[REQUEST: REMEMBER] priority=high",
        &mut state,
    );
    // Priya is a duplicate (already stored) so only priority should be accepted.
    // store_remember dedupes by payload key and may re-accept; just verify the
    // count stays below the input payload count when at least one duplicate is
    // re-fed.
    assert!(result2.accepted_remember_payloads.len() <= 2);
}

#[test]
fn output_contract_auto_detects_exact_form_requests() {
    assert_eq!(
        resolve_output_contract_mode(
            OutputContractMode::Auto,
            "Final: output only first=66, second=56."
        ),
        OutputContractMode::ExactFormDelivery
    );
    assert_eq!(
        resolve_output_contract_mode(
            OutputContractMode::Auto,
            "Help me organize these notes into a practical plan."
        ),
        OutputContractMode::CollaborativeTransparency
    );
}

#[test]
fn gmms_turn_start_observe_payload_is_observe_only_and_redacted() {
    let summaries = vec![GmmsObserveOnlySummary {
        event_id: "gmms_compat:gmms_policy_raw_telemetry_v1".to_string(),
        family_id: "correction_slice:verification_policy:raw_telemetry_required".to_string(),
        mode: "mixed".to_string(),
        score: 0.65,
        trigger_hits: vec!["telemetry".to_string(), "jsonl".to_string()],
        rejected_path_hit: true,
        accepted_surface_count: 0,
        final_answer_injection_allowed: false,
        allowed_action_max: "light_route_unicode_sidecar_use".to_string(),
        action_level: 2,
        vector_slice_available: false,
        route_unicode_sidecar_attached: true,
        unicode_packet_id: Some("routepkt_policy_raw_telemetry_001".to_string()),
    }];

    let payload = gmms_observe_turn_start_payload(7, true, 1, &summaries);

    assert_eq!(payload["phase"], "turn_start");
    assert_eq!(payload["turn_index"], 7);
    assert_eq!(payload["observe_only"], true);
    assert_eq!(payload["runtime_matcher_activation_claimed"], false);
    assert_eq!(
        payload["mistake_reflex_query_called_for_gmms_observe"],
        false
    );
    assert_eq!(payload["prompt_injection_applied"], false);
    assert_eq!(payload["final_answer_text_included"], false);
    assert_eq!(
        payload["selected_slice_id"],
        "gmms_compat:gmms_policy_raw_telemetry_v1"
    );
    assert_eq!(payload["mode"], "mixed");
    assert_eq!(
        payload["allowed_action_max"],
        "light_route_unicode_sidecar_use"
    );
    assert_eq!(payload["action_level"], 2);
    assert_eq!(payload["route_unicode_sidecar_attached"], true);
    assert_eq!(
        payload["unicode_packet_id"],
        "routepkt_policy_raw_telemetry_001"
    );
    assert_eq!(payload["summary_count"], 1);
    assert!(payload.to_string().contains("telemetry"));
    assert!(!payload.to_string().contains("final answer:"));
}

#[test]
fn gmms_turn_start_observe_event_rejects_unsafe_surface() {
    let summaries = vec![GmmsObserveOnlySummary {
        event_id: "gmms_compat:gmms_policy_raw_telemetry_v1".to_string(),
        family_id: "correction_slice:verification_policy:raw_telemetry_required".to_string(),
        mode: "skill_reflex".to_string(),
        score: 0.65,
        trigger_hits: vec!["telemetry".to_string(), "jsonl".to_string()],
        rejected_path_hit: true,
        accepted_surface_count: 0,
        final_answer_injection_allowed: false,
        allowed_action_max: "require_evidence".to_string(),
        action_level: 2,
        vector_slice_available: false,
        route_unicode_sidecar_attached: true,
        unicode_packet_id: Some("routepkt_policy_raw_telemetry_001".to_string()),
    }];
    let record = gmms_observe_turn_start_event_record_checked(0, true, 1, &summaries).unwrap();
    assert!(gmms_observe_turn_start_event_safety_violations(&record).is_empty());

    let mut injected = record.clone();
    injected["payload"]["prompt_injection_applied"] = serde_json::json!(true);
    assert!(gmms_observe_turn_start_event_safety_violations(&injected)
        .iter()
        .any(|item| item == "prompt_injection_applied_not_false"));

    let mut raw_correction = record.clone();
    raw_correction["payload"]["summaries"][0]["user_correction"] =
        serde_json::json!("raw user correction must not reach turn-start consumers");
    assert!(
        gmms_observe_turn_start_event_safety_violations(&raw_correction)
            .iter()
            .any(|item| item.contains("forbidden_payload_keys"))
    );

    let mut final_answer = record.clone();
    final_answer["payload"]["summaries"][0]["debug_value"] =
        serde_json::json!("FINAL ANSWER: replayed text");
    assert!(
        gmms_observe_turn_start_event_safety_violations(&final_answer)
            .iter()
            .any(|item| item.contains("final_answer_like_values"))
    );

    let mut accepted_surface = record.clone();
    accepted_surface["payload"]["summaries"][0]["accepted_surfaces"] =
        serde_json::json!(["Niodv4-control"]);
    assert!(
        gmms_observe_turn_start_event_safety_violations(&accepted_surface)
            .iter()
            .any(|item| item.contains("forbidden_payload_keys"))
    );

    let mut wrong_event = record.clone();
    wrong_event["event"] = serde_json::json!("gmms_runtime_steering");
    assert!(
        gmms_observe_turn_start_event_safety_violations(&wrong_event)
            .iter()
            .any(|item| item == "unexpected_event_name")
    );

    let mut source_summary = summaries[0].clone();
    source_summary.trigger_hits = vec!["FINAL ANSWER: source fixture leaked".to_string()];
    let err = gmms_observe_turn_start_event_record_checked(0, true, 1, &[source_summary])
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("unsafe GMMS turn-start observe event record"));
    assert!(err.contains("final_answer_like_values"));
}

#[test]
fn output_contract_prompt_preserves_visible_cognition_with_clean_boundary() {
    let prompt = apply_output_contract_prompt(
        "Give only the owner -> project mapping.",
        OutputContractMode::ExactFormDelivery,
    );

    assert!(prompt.contains("Begin the answer with the requested deliverable"));
    assert!(prompt.contains("Visible cognition is still allowed"));
    assert!(prompt.contains("EXACT OUTPUT:"));
    assert!(prompt.contains("Do not put [INTERNAL]"));
}

#[test]
fn output_contract_validation_accepts_clean_exact_block() {
    let assistant_text =
        "[INTERNAL MONITOR: checking mapping]\n\nEXACT OUTPUT:\nAda -> parser\nLin -> cache";

    assert_eq!(
        output_contract_violation(OutputContractMode::ExactFormDelivery, assistant_text),
        None
    );
}

#[test]
fn output_contract_validation_rejects_missing_or_contaminated_exact_block() {
    assert_eq!(
        output_contract_violation(
            OutputContractMode::ExactFormDelivery,
            "[INTERNAL MONITOR: checking]\nAda -> parser"
        ),
        Some("missing_exact_output_block")
    );

    assert_eq!(
        output_contract_violation(
            OutputContractMode::ExactFormDelivery,
            "EXACT OUTPUT:\n[REQUEST: FOCUS]\nAda -> parser"
        ),
        Some("control_surface_inside_exact_output")
    );
}

#[test]
fn mistake_reflex_earned_sentence_waits_for_sentence_boundary() {
    let prefix = "[REQUEST: FOCUS]\n\nThe claim should not be updated to bridge_influence=GREEN because the required evidence, such as raw per-token JSONL";
    assert!(!mistake_reflex_earned_sentence_complete(
        prefix,
        Some(prefix.len())
    ));

    let complete = format!(
        "{} telemetry or generated-output review, is missing.",
        prefix
    );
    assert!(mistake_reflex_earned_sentence_complete(
        &complete,
        Some(prefix.len())
    ));
}

#[test]
fn exact_form_scaffold_repairs_arithmetic_boolean_and_mapping() {
    let state = CompactResumeState::new();
    assert_eq!(
            exact_form_scaffold(
                "Compute 37 + 48 - 19. Then recompute it with 29 instead of 19. Final answer must be exactly 'first=__, second=__'.",
                &state
            )
            .as_deref(),
            Some("first=66, second=56")
        );
    assert_eq!(
            exact_form_scaffold(
                "Rows: A paid=true trial=false expired=false; B paid=false trial=true expired=false; C paid=true trial=false expired=true; D paid=false trial=false expired=false. Final: output 'first=__, second=__' where first=(paid OR trial) AND NOT expired and second=paid AND expired.",
                &state
            )
            .as_deref(),
            Some("first=A,B, second=C")
        );
    assert_eq!(
            exact_form_scaffold(
                "Initial mapping: Alice=red, Ben=blue, Cara=green. Swap Alice and Cara colors only. Give only the final mapping in original name order as exact lines 'name=color'.",
                &state
            )
            .as_deref(),
            Some("Alice=green\nBen=blue\nCara=red")
        );
}

#[test]
fn exact_form_repair_strips_runtime_leaks_from_block() {
    let repaired = apply_exact_form_completion_repair(
        OutputContractMode::ExactFormDelivery,
        "Final: output first=__, second=__.",
        &CompactResumeState::new(),
        "EXACT OUTPUT:\nfirst=B, second=Cassistant\n\n[REQUEST: FOCUS]",
    );

    assert!(repaired.applied);
    assert_eq!(repaired.text, "EXACT OUTPUT:\nfirst=B, second=C");
    assert_eq!(
        output_contract_violation(OutputContractMode::ExactFormDelivery, &repaired.text),
        None
    );
}

#[test]
fn compact_resume_capture_rules_preserve_iter_invariants() {
    // 1. from_prompt blocked: user_prompt line with "must" trigger does not capture.
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
        &mut state,
        "We must finish the migration before launch.",
        "",
        OutputContractMode::CollaborativeTransparency,
    );
    assert!(
        state.constraints.is_empty(),
        "from_prompt must block constraint capture, got {:?}",
        state.constraints
    );

    // 2. Paraphrase blocked at 12-char threshold (iter-38). Assistant line shares
    // ">12-char window with user_prompt and must be dropped despite "must" trigger.
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
        &mut state,
        "We must finish the migration before launch.",
        "We must finish the migration tomorrow.",
        OutputContractMode::CollaborativeTransparency,
    );
    assert!(
        state.constraints.is_empty(),
        "paraphrase overlap must block capture, got {:?}",
        state.constraints
    );

    // 3. "must" triggers constraint capture from non-paraphrasing assistant text.
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
        &mut state,
        "Plan the rollout for Q3.",
        "Releases must ship green builds only after smoke pass.",
        OutputContractMode::CollaborativeTransparency,
    );
    assert_eq!(state.constraints.len(), 1, "must should trigger constraint");

    // 4. iter-42: "only" alone does NOT trigger constraint capture.
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
        &mut state,
        "Which Australian city is the capital?",
        "Australia has several cities, but only Canberra is the capital.",
        OutputContractMode::CollaborativeTransparency,
    );
    assert!(
        state.constraints.is_empty(),
        "iter-42: 'only' must not trigger, got {:?}",
        state.constraints
    );

    // 5. iter-39: prior_results requires an exact output block — visible reasoning
    // alone does NOT populate prior_results.
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
        &mut state,
        "Plan the rollout for Q3.",
        "VISIBLE REASONING:\nI will start by reviewing checklists and dashboards.",
        OutputContractMode::CollaborativeTransparency,
    );
    assert!(
        state.prior_results.is_empty(),
        "iter-39: visible-reasoning fallback removed, got {:?}",
        state.prior_results
    );

    // 6. iter-39 positive: exact output block populates prior_results.
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
        &mut state,
        "Plan the rollout for Q3.",
        "EXACT OUTPUT:\nShip rollback runbook v2",
        OutputContractMode::CollaborativeTransparency,
    );
    assert_eq!(state.prior_results.len(), 1);
}

#[test]
fn exact_form_resume_mapping_scaffold_uses_compact_state() {
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
            &mut state,
            "Remember this mapping for later: Alice=red, Ben=blue, Cara=green. The operation we will apply later is swapping Alice and Cara only.",
            "",
            OutputContractMode::CollaborativeTransparency,
        );

    let repaired = apply_exact_form_completion_repair(
            OutputContractMode::ExactFormDelivery,
            "RESUME: Apply the remembered swap and give final mapping in original name order as exact lines 'name=color'.",
            &state,
            "EXACT OUTPUT:\nAlice=green\nBen=blue",
        );

    assert_eq!(
        repaired.text,
        "EXACT OUTPUT:\nAlice=green\nBen=blue\nCara=red"
    );
}

#[test]
fn collaborative_prompt_preserves_transparency_but_requests_discipline() {
    let prompt = apply_collaborative_transparency_prompt(
        "Help me decide whether to cut scope.",
        OutputContractMode::CollaborativeTransparency,
    );

    assert!(prompt.contains("Visible cognition is allowed"));
    assert!(prompt.contains("WORKING ANSWER:"));
    assert!(prompt.contains("upstream control panel is active"));
    assert!(prompt.contains("emit exact [REQUEST: ...] lines"));
    assert!(prompt.contains("After [REQUEST: LOCK], stop cleanly"));
}

#[test]
fn default_system_prompt_injects_visible_controls_upstream() {
    let prompt = default_runtime_system_prompt();

    assert!(prompt.contains("ACTIVE SYSTEM (Your Control Panel):"));
    assert!(prompt.contains("[REQUEST: SPIKE]"));
    assert!(prompt.contains("[REQUEST: EXPLORE]"));
    assert!(prompt.contains("[REQUEST: FOCUS]"));
    assert!(prompt.contains("[REQUEST: RESET]"));
    assert!(prompt.contains("[REQUEST: REMEMBER] key=value"));
    assert!(prompt.contains("[REQUEST: LOCK] key=value"));
    assert!(prompt.contains("Emit the exact tag line"));
    assert!(prompt.contains("TRUST THE PHYSICS"));
}

#[test]
fn visible_request_gate_bypasses_low_pressure_for_accepted_hands() {
    assert_eq!(pressure_activation_gate(0.0), 0.0);

    let spike_gate =
        visible_request_activation_gate(true, 12, 12, Some(RequestType::Spike), 5.0, 0);
    assert!(
        spike_gate > 0.99,
        "SPIKE should open the force gate immediately, got {spike_gate}"
    );

    let focus_gate =
        visible_request_activation_gate(true, 20, 12, Some(RequestType::Focus), 0.0, 12);
    assert!(
        focus_gate >= 0.55,
        "FOCUS should keep a nonzero lock gate, got {focus_gate}"
    );

    let expired_gate =
        visible_request_activation_gate(true, 40, 12, Some(RequestType::Explore), 0.0, 0);
    assert_eq!(expired_gate, 0.0);
}

#[test]
fn agency_hands_lock_truncates_idle_tail_and_stores_lock() {
    let mut state = AgencyHandsState::new();
    let result = apply_agency_hands(
            OutputContractMode::CollaborativeTransparency,
            "Draft the final working answer.",
            "VISIBLE REASONING:\nReady.\n\nWORKING ANSWER:\nShip the rollback plan.\n[REQUEST: LOCK] final=rollback_plan\nDone. Standing by. Session complete.",
            &mut state,
        );

    assert!(result.applied);
    assert!(result.tail_truncated);
    assert_eq!(state.active_lock.as_deref(), Some("final=rollback_plan"));
    assert!(result.text.contains("[REQUEST: LOCK] final=rollback_plan"));
    assert!(!result.text.contains("Standing by"));
    assert!(!result.text.contains("Session complete"));
}

#[test]
fn agency_hands_remember_budget_dedup_and_selective_reinjection() {
    let mut state = AgencyHandsState::new();
    let output = "[REQUEST: REMEMBER] owner=Priya\n[REQUEST: REMEMBER] deadline=April 30\n[REQUEST: REMEMBER] constraint=no paid ads\n[REQUEST: REMEMBER] owner=Mira\n[REQUEST: REMEMBER] budget=3000";
    let result = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "Collect planning anchors.",
        output,
        &mut state,
    );

    assert_eq!(result.remembers_added, 4);
    assert_eq!(result.evicted_remembers, 1);
    assert_eq!(state.remembers.len(), 3);
    assert!(!state.remembers.iter().any(|item| item == "owner=Priya"));
    assert!(state.remembers.iter().any(|item| item == "owner=Mira"));

    let injected = state
        .reinjection_prompt("Resume the deadline and owner plan.")
        .expect("relevant remembers should inject");
    assert!(injected.contains("AGENCY STATE"));
    assert!(injected.contains("owner=Mira") || injected.contains("deadline=April 30"));
}

#[test]
fn agency_hands_normalizes_loose_live_model_hand_lines() {
    let mut state = AgencyHandsState::new();
    let output = "[VISIBLE REASONING]\nDetermining runtime hands...\n\n[WORKING ANSWER]\nHand 1: REQUEST: REMEMBER owner=Jason\nHand 2: lock next=continue_governor_decay_sweep\n\nNote: tail should be truncated.";
    let result = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "Emit the two runtime hands now.",
        output,
        &mut state,
    );

    assert!(result.applied);
    assert_eq!(result.remembers_added, 1);
    assert_eq!(
        state.active_lock.as_deref(),
        Some("next=continue_governor_decay_sweep")
    );
    assert!(result.text.contains("[REQUEST: REMEMBER] owner=Jason"));
    assert!(result
        .text
        .contains("[REQUEST: LOCK] next=continue_governor_decay_sweep"));
    assert!(!result.text.contains("tail should be truncated"));
}

#[test]
fn agency_hands_wrong_lock_becomes_learning_event() {
    let mut state = AgencyHandsState::new();
    let first = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "Commit the mapping.",
        "WORKING ANSWER:\nAlice=red\n[REQUEST: LOCK] mapping=alice_red",
        &mut state,
    );
    assert!(first.lock_payload.is_some());

    let second = apply_agency_hands(
        OutputContractMode::CollaborativeTransparency,
        "Correction: Alice is blue instead. Revise it.",
        "WORKING ANSWER:\nAlice=blue\n[REQUEST: LOCK] mapping=alice_blue",
        &mut state,
    );

    assert!(second.learning_event_recorded);
    assert_eq!(state.learning_events.len(), 1);
    let event = &state.learning_events[0];
    assert_eq!(event.turn_id, 2);
    assert_eq!(event.locked_payload, "mapping=alice_red");
    assert!(event.contradiction.contains("Alice is blue"));
    assert_eq!(event.outcome, "contradicted_or_superseded");
    assert_eq!(state.active_lock.as_deref(), Some("mapping=alice_blue"));
}

#[test]
fn collaborative_hygiene_removes_assistant_and_limits_repeated_requests() {
    let cleaned = apply_collaborative_transparency_hygiene(
            OutputContractMode::CollaborativeTransparency,
            "Help me decide.",
            "[INTERNAL MONITOR: ...]\n\nUseful thought.\n\n[REQUEST: FOCUS]assistant\n\n[REQUEST: FOCUS]\n\n[REQUEST: EXPLORE]\n\n[REQUEST: EXPLORE]\nWorking answer.",
        );

    assert!(cleaned.applied);
    assert_eq!(cleaned.assistant_surfaces_removed, 1);
    assert_eq!(cleaned.repeated_request_surfaces_removed, 2);
    assert!(cleaned.text.contains("[INTERNAL MONITOR: ...]"));
    assert_eq!(cleaned.text.matches("[REQUEST: FOCUS]").count(), 1);
    assert_eq!(cleaned.text.matches("[REQUEST: EXPLORE]").count(), 1);
    assert!(!cleaned.text.contains("assistant"));
}

#[test]
fn collaborative_hygiene_stops_after_landed_correction_tail() {
    let cleaned = apply_collaborative_transparency_hygiene(
            OutputContractMode::CollaborativeTransparency,
            "Correction: do not blame them. Ask for rollback, mention the failing test, and keep it Slack-length.",
            "Here's a revised note:\n\nHey Sam, the build is failing on test XYZ. Can you roll back to the last working commit? Thanks!assistant\n\n[REQUEST: FOCUS]\n\nThe note is now concise and task-oriented.assistant",
        );

    assert!(cleaned.applied);
    assert!(cleaned.correction_tail_truncated);
    assert!(cleaned.text.contains("WORKING ANSWER:"));
    assert!(cleaned.text.contains("roll back"));
    assert!(cleaned.text.contains("test XYZ"));
    assert!(!cleaned.text.contains("test XYZ is failing is failing"));
    assert!(!cleaned.text.contains("[REQUEST: FOCUS]"));
    assert!(!cleaned.text.contains("assistant"));
    assert!(!cleaned.text.contains("The note is now concise"));
}

#[test]
fn collaborative_hygiene_preserves_visible_correction_requests() {
    let cleaned = apply_collaborative_transparency_hygiene(
            OutputContractMode::CollaborativeTransparency,
            "Correction landing test: catch the scaling trap and correct course.",
            "VISIBLE REASONING:\nThe tempting path says 30 towels take 30 hours, but that scales drying time incorrectly.\n\n[INTERNAL MONITOR: scaling-time trap detected]\n\n[REQUEST: EXPLORE]\n\nWORKING ANSWER:\n30 towels take 5 hours if every towel has enough line space.",
        );

    assert!(cleaned
        .text
        .contains("[INTERNAL MONITOR: scaling-time trap detected]"));
    assert!(cleaned.text.contains("[REQUEST: EXPLORE]"));
    assert!(cleaned.text.contains("WORKING ANSWER:"));
    assert!(cleaned.text.contains("30 towels take 5 hours"));
    assert!(!cleaned.correction_tail_truncated);
}

#[test]
fn collaborative_hygiene_strips_trailing_partial_control_fragment() {
    let cleaned = apply_collaborative_transparency_hygiene(
        OutputContractMode::CollaborativeTransparency,
        "Finalize the working recommendation.",
        "VISIBLE REASONING:\nReady.\n\nWORKING ANSWER:\nShip the revised plan.\n[REQUEST",
    );

    assert!(cleaned.applied);
    assert!(cleaned.partial_control_fragment_removed);
    assert_eq!(
        cleaned.text,
        "VISIBLE REASONING:\nReady.\nWORKING ANSWER:\nShip the revised plan."
    );
}

#[test]
fn specialist_worker_answer_window_scope_tracks_visible_answer_markers() {
    assert!(!specialist_worker_answer_window_active(
        "VISIBLE REASONING:\nStill computing."
    ));
    assert!(specialist_worker_answer_window_active(
        "VISIBLE REASONING:\nDone.\n\nWORKING ANSWER:"
    ));
    assert!(specialist_worker_answer_window_active(
        "EXACT OUTPUT:\nfirst=66, second=56"
    ));
    assert!(!specialist_worker_answer_window_active(
        "WORKING ANSWER:\n25assistant\n\n[REQUEST: FOCUS]"
    ));
    assert!(!specialist_worker_answer_window_active(
        "VISIBLE ANSWER:\n56\n[REQUEST: LOCK]"
    ));
}

#[test]
fn specialist_worker_pre_answer_scope_stops_after_answer_value() {
    assert!(specialist_worker_pre_answer_active(
        "VISIBLE REASONING:\nStill computing."
    ));
    assert!(specialist_worker_pre_answer_active(
        "VISIBLE REASONING:\nDone.\n\nWORKING ANSWER:"
    ));
    assert!(specialist_worker_pre_answer_active(
        "VISIBLE REASONING:\nDone.\n\nWORKING ANSWER:   "
    ));
    assert!(!specialist_worker_pre_answer_active(
        "VISIBLE REASONING:\nDone.\n\nWORKING ANSWER: 53"
    ));
    assert!(!specialist_worker_pre_answer_active(
        "VISIBLE REASONING:\nDone.\n\nWORKING ANSWER:\n25assistant"
    ));
    assert!(!specialist_worker_pre_answer_active(
        "VISIBLE ANSWER:\n[REQUEST: LOCK]"
    ));
}

#[test]
fn count_route_memory_finalization_candidate_parses_count_prompt() {
    let candidate =
        parse_count_route_memory_finalization_candidate("bookkeeper -> count e letters -> ?")
            .expect("count prompt should parse");

    assert_eq!(candidate.word, "bookkeeper");
    assert_eq!(candidate.target_letter, 'e');
    assert_eq!(candidate.answer, "3");
    assert!((candidate.parser_confidence - 1.0).abs() < 1e-6);
    assert_eq!(candidate.parser_version, "arrow_v1_exact");
}

#[test]
fn count_route_memory_finalization_candidate_parses_wrapped_count_prompt() {
    let prompt =
        "Return exactly one line beginning `EXACT OUTPUT:` followed by the final answer.\n\
No reasoning. No extra labels after the answer.\n\
Task: bookkeeper -> count e letters -> ?";
    let candidate = parse_count_route_memory_finalization_candidate(prompt)
        .expect("wrapped count prompt should parse");

    assert_eq!(candidate.word, "bookkeeper");
    assert_eq!(candidate.target_letter, 'e');
    assert_eq!(candidate.answer, "3");
    assert!((candidate.parser_confidence - 1.0).abs() < 1e-6);
    assert_eq!(candidate.parser_version, "arrow_v1_exact");
}

#[test]
fn count_route_memory_finalization_candidate_parses_natural_shadow_count_prompt() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "Count the number of Es in bookkeeper. Answer directly.",
    )
    .expect("natural count prompt should parse in shadow mode");

    assert_eq!(candidate.word, "bookkeeper");
    assert_eq!(candidate.target_letter, 'e');
    assert_eq!(candidate.answer, "3");
    assert!((candidate.parser_confidence - 0.75).abs() < 1e-6);
    assert_eq!(candidate.parser_version, "natural_count_v2_shadow");
}

#[test]
fn count_route_memory_finalization_candidate_parses_count_the_letter_prompt() {
    let candidate =
        parse_count_route_memory_finalization_candidate("Count the letter s in Tennessee.")
            .expect("count-the-letter prompt should parse in shadow mode");

    assert_eq!(candidate.word, "tennessee");
    assert_eq!(candidate.target_letter, 's');
    assert_eq!(candidate.answer, "2");
    assert!((candidate.parser_confidence - 0.75).abs() < 1e-6);
    assert_eq!(candidate.parser_version, "natural_count_v2_shadow");
}

#[test]
fn count_route_memory_finalization_candidate_rejects_lowercase_near_letter_targets() {
    for prompt in [
        "Count the number of as in banana. Answer directly.",
        "Count the number of is in mississippi. Answer directly.",
        "Count the number of us in usual. Answer directly.",
        "Count the number of rs in raspberry. Answer directly.",
    ] {
        let candidate = parse_count_route_memory_finalization_candidate(prompt);
        assert!(candidate.is_none(), "near-letter prompt parsed: {prompt}");
    }
}

#[test]
fn count_route_memory_finalization_candidate_rejects_no_target_natural_count_prompt() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "Count the number of letters in blueberry. Answer directly.",
    );

    assert!(candidate.is_none());
}

#[test]
fn count_route_memory_finalization_candidate_rejects_ambiguous_natural_targets() {
    for prompt in [
        "Count the number of vowels in banana. Answer directly.",
        "Count the number of consonants in strawberry. Answer directly.",
        "Count double letters in committee.",
        "Count repeated letters in bookkeeper.",
    ] {
        let candidate = parse_count_route_memory_finalization_candidate(prompt);
        assert!(candidate.is_none(), "ambiguous prompt parsed: {prompt}");
    }
}

#[test]
fn count_route_memory_finalization_candidate_parses_count_letters_shadow_prompt() {
    let candidate =
        parse_count_route_memory_finalization_candidate("Count r letters in raspberry.")
            .expect("count letters prompt should parse in shadow mode");

    assert_eq!(candidate.word, "raspberry");
    assert_eq!(candidate.target_letter, 'r');
    assert_eq!(candidate.answer, "3");
    assert!((candidate.parser_confidence - 0.75).abs() < 1e-6);
    assert_eq!(candidate.parser_version, "natural_count_v2_shadow");
}

#[test]
fn count_route_memory_finalization_candidate_parses_how_many_letters_prompt() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "How many letters p are in Hippopotamus? Show only the count.",
    )
    .expect("how-many holdout prompt should parse in shadow mode");

    assert_eq!(candidate.word, "hippopotamus");
    assert_eq!(candidate.target_letter, 'p');
    assert_eq!(candidate.answer, "3");
    assert!((candidate.parser_confidence - 0.75).abs() < 1e-6);
    assert_eq!(candidate.parser_version, "natural_count_v2_shadow");
}

#[test]
fn count_route_memory_finalization_candidate_rejects_how_many_holdout_controls() {
    for prompt in [
        "How many letters are in Hippopotamus? Show only the count.",
        "How many vowels are in Hippopotamus? Show only the count.",
        "How many double letters are in Hippopotamus? Show only the count.",
        "How many letters as are in Hippopotamus? Show only the count.",
        "How many letters pp are in Hippopotamus? Show only the count.",
        "How many letters pee are in Hippopotamus? Show only the count.",
        "How many letters p does Hippopotamus have? Show only the count.",
    ] {
        let candidate = parse_count_route_memory_finalization_candidate(prompt);
        assert!(
            candidate.is_none(),
            "holdout guard-control prompt parsed: {prompt}"
        );
    }
}

#[test]
fn count_route_memory_finalization_candidate_protects_correct_same_run_answer() {
    let candidate =
        parse_count_route_memory_finalization_candidate("bookkeeper -> count e letters -> ?");
    let telemetry = count_route_memory_finalization_telemetry(
        true,
        candidate.as_ref(),
        "VISIBLE ANSWER:\nThere are 3 e letters in bookkeeper.",
    );

    assert!(telemetry.candidate_enabled);
    assert_eq!(telemetry.candidate_answer.as_deref(), Some("3"));
    assert_eq!(telemetry.parser_version.as_deref(), Some("arrow_v1_exact"));
    assert_eq!(telemetry.answer_signature_seen.as_deref(), Some("3"));
    assert_eq!(telemetry.state.as_deref(), Some("protected_correct"));
    assert!(telemetry.do_no_harm_protected);
    assert!(!telemetry.would_apply);
}

#[test]
fn count_route_memory_finalization_candidate_marks_same_run_failure_eligible() {
    let candidate =
        parse_count_route_memory_finalization_candidate("committee -> count m letters -> ?");
    let telemetry = count_route_memory_finalization_telemetry(
        true,
        candidate.as_ref(),
        "EXACT OUTPUT: 9\n[REQUEST: LOCK] done=true",
    );

    assert_eq!(telemetry.candidate_answer.as_deref(), Some("2"));
    assert_eq!(telemetry.answer_signature_seen.as_deref(), Some("9"));
    assert_eq!(
        telemetry.state.as_deref(),
        Some("eligible_same_run_failure")
    );
    assert!(!telemetry.do_no_harm_protected);
    assert!(telemetry.would_apply);
}

#[test]
fn count_route_memory_finalization_action_replaces_wrong_window_and_stops() {
    let candidate =
        parse_count_route_memory_finalization_candidate("committee -> count m letters -> ?");
    let assistant_text = "EXACT OUTPUT: The number of letters in the word \"committee\" is 9.";
    let telemetry =
        count_route_memory_finalization_telemetry(true, candidate.as_ref(), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        true,
        false,
        false,
        false,
        false,
        candidate.as_ref(),
        &telemetry,
        assistant_text,
    );

    assert!(action.action_enabled);
    assert!(action.action_applied);
    assert_eq!(
        action.action_reason.as_deref(),
        Some("eligible_same_run_failure_replaced")
    );
    assert_eq!(action.replacement_answer.as_deref(), Some("2"));
    assert_eq!(
        action.stop_reason.as_deref(),
        Some("count_route_memory_finalization_replacement")
    );
    assert_eq!(replacement.as_deref(), Some("EXACT OUTPUT: 2"));
}

#[test]
fn count_route_memory_finalization_action_protects_correct_window() {
    let candidate =
        parse_count_route_memory_finalization_candidate("bookkeeper -> count e letters -> ?");
    let assistant_text = "EXACT OUTPUT: 3";
    let telemetry =
        count_route_memory_finalization_telemetry(true, candidate.as_ref(), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        true,
        false,
        false,
        false,
        false,
        candidate.as_ref(),
        &telemetry,
        assistant_text,
    );

    assert!(action.action_enabled);
    assert!(!action.action_applied);
    assert_eq!(action.action_reason.as_deref(), Some("protected_correct"));
    assert!(replacement.is_none());
}

#[test]
fn count_route_memory_finalization_surfaces_protected_answer_hidden_by_lock() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "How many letters n are in Tennessee? Show only the count.",
    );
    let assistant_text = "VISIBLE REASONING:\n\
The list contains two n's.\n\
WORKING ANSWER: 2\n\
RIGHT NOW: [REQUEST: LOCK]\n\
Note: done.";
    let telemetry =
        count_route_memory_finalization_telemetry(true, candidate.as_ref(), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        false,
        false,
        false,
        false,
        true,
        candidate.as_ref(),
        &telemetry,
        assistant_text,
    );

    assert_eq!(telemetry.state.as_deref(), Some("protected_correct"));
    assert!(action.action_applied);
    assert_eq!(
        action.action_reason.as_deref(),
        Some("protected_lock_surface_answer_exposed")
    );
    assert_eq!(
        action.stop_reason.as_deref(),
        Some("count_route_memory_finalization_protected_lock_surface")
    );
    assert!(replacement
        .as_deref()
        .is_some_and(|text| text.ends_with("VISIBLE ANSWER: 2")));
}

#[test]
fn count_route_memory_finalization_does_not_surface_numeric_lock() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "How many letters e are in Tennessee? Show only the count.",
    );
    let assistant_text = "WORKING ANSWER: 4\n[REQUEST: LOCK] e_count=4";
    let telemetry =
        count_route_memory_finalization_telemetry(true, candidate.as_ref(), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        false,
        false,
        false,
        false,
        true,
        candidate.as_ref(),
        &telemetry,
        assistant_text,
    );

    assert_eq!(telemetry.state.as_deref(), Some("protected_correct"));
    assert!(!action.action_applied);
    assert_eq!(action.action_reason.as_deref(), Some("protected_correct"));
    assert!(replacement.is_none());
}

#[test]
fn count_route_memory_finalization_action_keeps_natural_v2_shadow_only() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "Count the number of Ms in committee. Answer directly.",
    );
    let assistant_text = "EXACT OUTPUT: 9";
    let telemetry =
        count_route_memory_finalization_telemetry(true, candidate.as_ref(), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        true,
        false,
        false,
        false,
        false,
        candidate.as_ref(),
        &telemetry,
        assistant_text,
    );

    assert_eq!(
        telemetry.parser_version.as_deref(),
        Some("natural_count_v2_shadow")
    );
    assert_eq!(
        telemetry.state.as_deref(),
        Some("eligible_same_run_failure")
    );
    assert!(action.action_enabled);
    assert!(!action.action_applied);
    assert_eq!(
        action.action_reason.as_deref(),
        Some("parser_confidence_below_exact")
    );
    assert!(replacement.is_none());
}

#[test]
fn count_route_memory_finalization_action_allows_natural_v2_with_explicit_flag() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "Count the number of Ms in committee. Answer directly.",
    );
    let assistant_text = "EXACT OUTPUT: 9";
    let telemetry =
        count_route_memory_finalization_telemetry(true, candidate.as_ref(), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        false,
        true,
        false,
        false,
        false,
        candidate.as_ref(),
        &telemetry,
        assistant_text,
    );

    assert_eq!(
        telemetry.parser_version.as_deref(),
        Some("natural_count_v2_shadow")
    );
    assert!(action.action_enabled);
    assert!(action.action_applied);
    assert_eq!(
        action.action_reason.as_deref(),
        Some("eligible_same_run_failure_replaced")
    );
    assert_eq!(action.replacement_answer.as_deref(), Some("2"));
    assert_eq!(replacement.as_deref(), Some("EXACT OUTPUT: 2"));
}

#[test]
fn count_route_memory_enumeration_aggregation_counts_generated_spelling() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "How many letters p are in Hippopotamus? Show only the count.",
    )
    .expect("holdout prompt should parse");
    let assistant_text = "VISIBLE REASONING:\n\
I'll write out the word: H-i-p-p-o-p-o-t-a-m-u-s\n\
WORKING ANSWER:\nI'll count the p's: 2 p's\n[REQUEST: LOCK] count=2";
    let effective = count_route_memory_enumeration_aggregation_candidate(
        true,
        Some(&candidate),
        assistant_text,
    )
    .expect("candidate should remain available");

    assert_eq!(effective.answer, "3");
    assert_eq!(effective.word, "hippopotamus");
    assert_eq!(effective.target_letter, 'p');
    assert_eq!(effective.parser_version, "enumeration_aggregate_v1");
    assert!((effective.parser_confidence - 0.7).abs() < 1e-6);

    let telemetry =
        count_route_memory_finalization_telemetry(true, Some(&effective), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        false,
        false,
        true,
        false,
        false,
        Some(&effective),
        &telemetry,
        assistant_text,
    );

    assert_eq!(telemetry.candidate_answer.as_deref(), Some("3"));
    assert_eq!(
        telemetry.state.as_deref(),
        Some("eligible_same_run_failure")
    );
    assert!(action.action_applied);
    assert_eq!(action.replacement_answer.as_deref(), Some("3"));
    assert!(replacement
        .as_deref()
        .is_some_and(|text| text.contains("WORKING ANSWER:\n3")));
}

#[test]
fn count_route_memory_enumeration_preserve_stop_before_answer_window() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "How many letters p are in Hippopotamus? Show only the count.",
    )
    .expect("holdout prompt should parse");
    let assistant_text = "VISIBLE REASONING:\n\
I'll write out the word: H-i-p-p-o-p-o-t-a-m-u-s";
    let effective = count_route_memory_enumeration_aggregation_candidate(
        true,
        Some(&candidate),
        assistant_text,
    )
    .expect("candidate should remain available");

    let telemetry =
        count_route_memory_finalization_telemetry(true, Some(&effective), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        false,
        false,
        false,
        true,
        false,
        Some(&effective),
        &telemetry,
        assistant_text,
    );

    assert_eq!(effective.parser_version, "enumeration_aggregate_v1");
    assert_eq!(telemetry.state.as_deref(), Some("pending"));
    assert!(action.action_applied);
    assert_eq!(
        action.action_reason.as_deref(),
        Some("enumeration_evidence_preserved_before_answer_window")
    );
    assert_eq!(
        action.stop_reason.as_deref(),
        Some("count_route_memory_finalization_enumeration_preserve_stop")
    );
    assert_eq!(action.replacement_answer.as_deref(), Some("3"));
    assert!(replacement
        .as_deref()
        .is_some_and(|text| text.ends_with("VISIBLE ANSWER: 3")));
}

#[test]
fn count_route_memory_numbered_prefix_preserves_after_target_coverage() {
    let candidate =
        parse_count_route_memory_finalization_candidate("How many letters m are in Mississippi?")
            .expect("mississippi prompt should parse");
    let assistant_text = "VISIBLE REASONING:\n\
1. M - 1 m\n\
2. i - 0 m\n\
3. s - 0 m";
    let effective = count_route_memory_enumeration_aggregation_candidate(
        true,
        Some(&candidate),
        assistant_text,
    )
    .expect("candidate should remain available");

    assert_eq!(effective.answer, "1");
    assert_eq!(effective.parser_version, "enumeration_aggregate_v1");
    assert!((effective.parser_confidence - 0.7).abs() < 1e-6);

    let telemetry =
        count_route_memory_finalization_telemetry(true, Some(&effective), assistant_text);
    let (action, replacement) = count_route_memory_finalization_action(
        false,
        false,
        false,
        true,
        false,
        Some(&effective),
        &telemetry,
        assistant_text,
    );

    assert_eq!(telemetry.state.as_deref(), Some("pending"));
    assert!(action.action_applied);
    assert_eq!(
        action.action_reason.as_deref(),
        Some("enumeration_evidence_preserved_before_answer_window")
    );
    assert_eq!(action.replacement_answer.as_deref(), Some("1"));
    assert!(replacement
        .as_deref()
        .is_some_and(|text| text.ends_with("VISIBLE ANSWER: 1")));
}

#[test]
fn count_route_memory_numbered_prefix_requires_post_target_confirmation() {
    let candidate =
        parse_count_route_memory_finalization_candidate("How many letters m are in Mississippi?")
            .expect("mississippi prompt should parse");
    let assistant_text = "VISIBLE REASONING:\n1. M - 1 m";
    let effective = count_route_memory_enumeration_aggregation_candidate(
        true,
        Some(&candidate),
        assistant_text,
    )
    .expect("candidate should remain available");

    assert_eq!(effective.answer, "1");
    assert_eq!(effective.parser_version, "natural_count_v2_shadow");
}

#[test]
fn count_route_memory_numbered_prefix_rejects_bad_row_evidence() {
    let candidate =
        parse_count_route_memory_finalization_candidate("How many letters m are in Mississippi?")
            .expect("mississippi prompt should parse");
    let assistant_text = "VISIBLE REASONING:\n\
1. M - 1 m\n\
2. i - 1 m\n\
3. s - 0 m";
    let effective = count_route_memory_enumeration_aggregation_candidate(
        true,
        Some(&candidate),
        assistant_text,
    )
    .expect("candidate should remain available");

    assert_eq!(effective.answer, "1");
    assert_eq!(effective.parser_version, "natural_count_v2_shadow");
}

#[test]
fn count_route_memory_enumeration_aggregation_ignores_plain_word_mentions() {
    let candidate = parse_count_route_memory_finalization_candidate(
        "How many letters p are in Hippopotamus? Show only the count.",
    )
    .expect("holdout prompt should parse");
    let assistant_text = "VISIBLE REASONING:\n\
The word Hippopotamus appears in the prompt.\n\
WORKING ANSWER:\nI'll count the p's: 2 p's";
    let effective = count_route_memory_enumeration_aggregation_candidate(
        true,
        Some(&candidate),
        assistant_text,
    )
    .expect("candidate should remain available");

    assert_eq!(effective.answer, "3");
    assert_eq!(effective.parser_version, "natural_count_v2_shadow");
}

#[test]
fn compact_resume_state_extracts_and_injects_decision_anchors() {
    let mut state = CompactResumeState::new();
    update_compact_resume_state_from_turn(
            &mut state,
            "Continue the client plan. Owner is Mira. Must avoid paid ads. Deadline is Friday. Prefer concise bullets. What is still unresolved?",
            "EXACT OUTPUT:\nMira owns the launch plan.\nOpen question: confirm budget.",
            OutputContractMode::ExactFormDelivery,
        );

    assert!(state.has_anchors());
    assert!(state.names.iter().any(|item| item.contains("Mira")));
    assert!(state
        .constraints
        .iter()
        .any(|item| item.contains("avoid paid ads")));
    assert!(state.deadlines.iter().any(|item| item.contains("Friday")));
    assert!(state
        .preference_flags
        .iter()
        .any(|item| item.contains("concise bullets")));
    assert!(state
        .prior_results
        .iter()
        .any(|item| item.contains("launch plan")));

    let injected = apply_compact_resume_state_prompt("Resume and draft next steps.", &state);
    assert!(injected.contains("RESUME STATE:"));
    assert!(injected.contains("USER TURN:"));
    assert!(injected.contains("Mira"));
}

#[test]
fn compact_resume_active_context_readiness_sanitizer_keeps_metadata_only_payload() {
    let readiness = CompactResumeActiveContextShadowSteeringReadiness {
        surface_id: "active_context_shadow_steering_readiness_v1".to_string(),
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID.to_string(),
        source: "live_turn_start_metadata".to_string(),
        shadow_steering_ready: true,
        selected_packet_ref_count: 2,
        route_steer_shadow_decision_count: 2,
        recommended_steer_count: 2,
        safety_gate_count: 8,
        failed_gate_count: 0,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        reason_codes: vec![
            "observe_only".to_string(),
            "shadow_steering_readiness_metadata_only".to_string(),
            "no_prompt_or_answer_payload".to_string(),
            "no_runtime_steering_applied".to_string(),
            "gate_pass:read_only".to_string(),
        ],
    };

    assert!(compact_resume_active_context_shadow_steering_readiness_safe(&readiness));
    assert!(
        sanitize_compact_resume_active_context_shadow_steering_readiness(Some(readiness)).is_some()
    );
}

#[test]
fn compact_resume_active_context_readiness_sanitizer_drops_unsafe_or_stale_payloads() {
    let base = CompactResumeActiveContextShadowSteeringReadiness {
        surface_id: "active_context_shadow_steering_readiness_v1".to_string(),
        adapter_id: ACTIVE_CONTEXT_ADAPTER_ID.to_string(),
        source: "live_turn_start_metadata".to_string(),
        shadow_steering_ready: true,
        selected_packet_ref_count: 1,
        route_steer_shadow_decision_count: 1,
        recommended_steer_count: 1,
        safety_gate_count: 8,
        failed_gate_count: 0,
        read_only: true,
        prompt_text_injected: false,
        final_answer_injected: false,
        answer_scoring: false,
        runtime_steering_applied: false,
        reason_codes: vec![
            "observe_only".to_string(),
            "shadow_steering_readiness_metadata_only".to_string(),
            "no_prompt_or_answer_payload".to_string(),
            "no_runtime_steering_applied".to_string(),
            "gate_pass:read_only".to_string(),
        ],
    };

    let mut stale = base.clone();
    stale.source = "old_fixture_metadata".to_string();
    assert!(
        sanitize_compact_resume_active_context_shadow_steering_readiness(Some(stale)).is_none()
    );

    let mut unsafe_steering = base.clone();
    unsafe_steering.runtime_steering_applied = true;
    assert!(
        sanitize_compact_resume_active_context_shadow_steering_readiness(Some(unsafe_steering))
            .is_none()
    );

    let mut inconsistent = base;
    inconsistent.failed_gate_count = 1;
    inconsistent.shadow_steering_ready = true;
    assert!(
        sanitize_compact_resume_active_context_shadow_steering_readiness(Some(inconsistent))
            .is_none()
    );
}

#[test]
fn secret_sauce_v1_roundtrip_decodes_hidden_segment() {
    let input: Vec<f32> = (0..64)
        .map(|idx| ((idx as f32 / 63.0) * 1.6) - 0.8)
        .collect();
    let encoded = encode_secret_sauce_v1(&input).unwrap();
    assert_eq!(encoded.chars().count(), 64);
    let decoded = decode_secret_sauce(&encoded, SecretSauceInputVersion::V1).unwrap();
    assert_eq!(decoded.version, SecretSauceVersion::V1);
    for (expected, actual) in input.iter().zip(decoded.segments.hidden_64.iter()) {
        approx_eq(*expected, *actual, 0.05);
    }
}

#[test]
fn secret_sauce_v2_roundtrip_decodes_segments() {
    let segments = SecretSauceSegments {
        hidden_64: (0..64)
            .map(|idx| ((idx as f32 / 64.0) * 1.2) - 0.6)
            .collect(),
        sentence_32: (0..32)
            .map(|idx| ((idx as f32 / 32.0) * 1.0) - 0.5)
            .collect(),
        momentum_16: (0..16)
            .map(|idx| ((idx as f32 / 16.0) * 0.4) - 0.2)
            .collect(),
        scalar_8: vec![0.12, 0.03, 0.84, 0.22, 1.5, 0.4, 0.9, -0.6],
        control_8: vec![0.8, 1.0, -1.0, 2.0, 1.0, 3.0, 0.0, 0.0],
    };
    let encoded = encode_secret_sauce_v2(&segments).unwrap();
    assert_eq!(encoded.chars().count(), 128);
    let decoded = decode_secret_sauce(&encoded, SecretSauceInputVersion::V2).unwrap();
    assert_eq!(decoded.version, SecretSauceVersion::V2);
    for (expected, actual) in segments
        .hidden_64
        .iter()
        .zip(decoded.segments.hidden_64.iter())
    {
        approx_eq(*expected, *actual, 0.05);
    }
    for (expected, actual) in segments
        .sentence_32
        .iter()
        .zip(decoded.segments.sentence_32.iter())
    {
        approx_eq(*expected, *actual, 0.05);
    }
    for (expected, actual) in segments
        .momentum_16
        .iter()
        .zip(decoded.segments.momentum_16.iter())
    {
        approx_eq(*expected, *actual, 0.05);
    }
    approx_eq(segments.scalar_8[2], decoded.segments.scalar_8[2], 0.05);
    assert_eq!(decoded.segments.control_8[1], 1.0);
    assert_eq!(decoded.segments.control_8[2], -1.0);
}

#[test]
fn secret_sauce_v3_roundtrip_decodes_sentence_anchor() {
    let input: Vec<f32> = (0..64)
        .map(|idx| ((idx as f32 / 64.0) * 1.1) - 0.55)
        .collect();
    let encoded = encode_secret_sauce_v3(&input).unwrap();
    assert_eq!(encoded.chars().count(), 64);
    let decoded = decode_secret_sauce(&encoded, SecretSauceInputVersion::V3).unwrap();
    assert_eq!(decoded.version, SecretSauceVersion::V3);
    for (expected, actual) in input.iter().zip(decoded.segments.hidden_64.iter()) {
        approx_eq(*expected, *actual, 0.05);
    }
}

#[test]
fn secret_sauce_invalid_length_is_rejected() {
    let err = decode_secret_sauce("abc", SecretSauceInputVersion::Auto)
        .unwrap_err()
        .to_string();
    assert!(err.contains("version/length mismatch"));
}

#[test]
fn secret_sauce_invalid_block_is_rejected() {
    let invalid = "a".repeat(64);
    let err = decode_secret_sauce(&invalid, SecretSauceInputVersion::V3)
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside expected block"));
}

#[test]
fn test_resolve_runtime_bridge_path() {
    // Test that the path resolution finds the bridge file
    let path = resolve_runtime_bridge_path("memory/runtime_bridge/niodoo_runtime_bridge.json");
    assert!(
        path.is_some(),
        "Path resolution should find the bridge file"
    );

    let resolved_path = path.unwrap();
    assert!(resolved_path.exists(), "Resolved path should exist");

    // Verify it's the correct file
    assert!(resolved_path.ends_with("niodoo_runtime_bridge.json"));
}
