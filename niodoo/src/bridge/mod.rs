//! # Bridge Module — niodv4 Integration Pathway
//!
//! This module provides gated, non-default integration paths for niodv4 research artifacts.
//! All bridges are OFF by default via feature flags and config switches.
//!
//! ## Current Status
//!
//! - **GhostRegistry**: Container for all ghost basins from niodv4 export
//! - **Adapter**: Converts niodv4 JSON formats to Rust GhostRegistry/basins format
//! - **Loader**: Loads converted artifacts with feature-gated activation
//!
//! ## Feature Flags
//!
//! ```toml
//! [features]
//! niodv4_bridge = []
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use niodoo::bridge::registry::GhostRegistry;
//!
//! // Load niodv4 ghost candidate registry
//! let registry = GhostRegistry::load_from_path("niodv4/data/results/summaries/ghost_candidate_registry.json")?;
//!
//! // Access ghost basins (64D vectors)
//! for basin in &registry.basins {
//!     println!("Ghost {}: {}D", basin.id, basin.dimension);
//! }
//! ```

pub mod adapter;
pub mod codebook;
pub mod correction_packets;
pub mod ghost_basin;
pub mod rave_codec;
pub mod registry;
pub mod secret_sauce;
pub mod smoke_test;
pub mod specialist;
pub mod specialist_bank;
pub mod startup_log;
pub mod tede_corrector;

pub use codebook::CodebookVQ;
pub use correction_packets::{
    decide_packet_authority, CorrectionPacket, CorrectionPacketStore, PacketAuthorityContext,
    PacketAuthorityDecision,
};
pub use specialist_bank::RuleBasedSpecialist;
