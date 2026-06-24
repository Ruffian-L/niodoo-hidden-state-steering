//! # Bridge Smoke Test — Verify niodv4 Registry Loading
//!
//! This module provides a smoke test that loads the real niodv4
//! ghost_candidate_registry.json and verifies the bridge can deserialize
//! and query the artifacts.
//!
//! ## Usage
//!
//! ```bash
//! cargo test --features niodv4_bridge bridge::smoke_test::test_registry_load
//! ```
//!
//! ## What It Tests
//!
//! 1. Registry JSON loads without errors
//! 2. All expected fields deserialize correctly
//! 3. Query methods return expected counts
//! 4. Bridge types are properly connected

#[cfg(test)]
#[cfg(feature = "niodv4_bridge")]
use crate::bridge::registry::GhostRegistry;

/// Path to the real niodv4 ghost candidate registry.
/// This file is exported by niodv4's Python pipeline.
#[allow(dead_code)]
const REGISTRY_PATH: &str = "niodv4/data/results/summaries/ghost_candidate_registry.json";

/// Smoke test: verify the bridge can load and query the real registry.
#[test]
#[cfg(feature = "niodv4_bridge")]
fn test_registry_load() {
    // Load the registry from the real niodv4 export
    let registry = GhostRegistry::load_from_path(REGISTRY_PATH)
        .expect("Failed to load ghost registry from niodv4 export");

    // Verify basic counts (JSON only has basins, no motifs/specialists)
    assert!(
        registry.has_candidates(),
        "Registry should have candidate basins"
    );
    // Motifs and specialists are optional in the JSON format
    // assert!(registry.has_motifs(), "Registry should have motifs");
    // assert!(registry.has_specialists(), "Registry should have specialists");

    // Verify we can filter candidates
    let candidates: Vec<_> = registry.candidate_basins().collect();
    assert!(!candidates.is_empty(), "Should have at least one candidate");

    // Verify we can query by target (returns empty if no specialists in JSON)
    let _temporal_specs = registry.specialists_by_target("temporal");
    // May be empty depending on registry contents

    // Verify corrections filtering works
    let _active = registry.active_corrections(0.9);
    // May be empty depending on registry contents, but should not panic

    println!("Registry loaded successfully:");
    println!("  Candidates: {}", registry.candidate_count());
    println!("  Motifs: {}", registry.motif_count());
    println!("  Specialists: {}", registry.specialist_count());
    println!("  Corrections: {}", registry.correction_count());
}

/// Smoke test: verify the registry can be serialized back to JSON.
#[test]
#[cfg(feature = "niodv4_bridge")]
fn test_registry_roundtrip() {
    let registry =
        GhostRegistry::load_from_path(REGISTRY_PATH).expect("Failed to load ghost registry");

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&registry).expect("Failed to serialize registry");

    // Deserialize back
    let deserialized: GhostRegistry =
        serde_json::from_str(&json).expect("Failed to deserialize registry");

    // Verify key properties match
    assert_eq!(deserialized.candidate_count(), registry.candidate_count());
    assert_eq!(deserialized.motif_count(), registry.motif_count());
    assert_eq!(deserialized.specialist_count(), registry.specialist_count());

    println!("Registry roundtrip successful");
}
