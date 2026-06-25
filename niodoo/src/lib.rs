// Library surface for the runtime binary. Only modules the eval binary actually
// reaches are exposed here; the historical niodv4 research module tree has been
// retired from the public tree.
#[cfg(feature = "niodv4_bridge")]
pub mod bridge;
pub mod ontological_inversion;
pub mod physics;
pub mod types;
pub mod visualizer;

pub mod config;
pub mod gpu;
pub mod indexing;
pub mod tivm;
pub mod utils;

/// Scalar Braille/Cuneiform transport for secret_sauce V3 (tests + round-trip fidelity).
pub mod secret_sauce_codec;

pub use types::{SplatId, SplatInput, SplatMeta};
