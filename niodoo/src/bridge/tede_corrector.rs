//! TEDE corrector — Rust inference port of niodv4's `TinyDipoleExpertTrainer`
//! (DEEP_DIVE_ROADMAP P3-A, Rust-integration half).
//!
//! The training side already exists in `niodv4/src/tede_supervised_train.py` with trained
//! checkpoints (`tede_specialist_targetonly_baseline_60.pth`, phase1/2/3). This module is
//! the missing piece: load such a checkpoint and run it as a learned corrector inside the
//! `niodoo` runtime, replacing hand-tuned rule-based nudges.
//!
//! Architecture (verified against the checkpoint state_dicts, 450 params):
//! `Linear(8→16) → tanh → Linear(16→16) → tanh → Linear(16→2)`.
//! Output is a 2D nudge on the protected core dimensions, clamped to ±[`DELTA_CLAMP`]
//! exactly as the niodv4 training rollout clamps it.
//!
//! Input is the 8-vector assembled in the training order (see [`TedeCorrector::build_input`]):
//! `[core(2), vel(2), ghost_target(2), energy(1), entropy_proxy(1)]`.
//!
//! Loaded from a safetensors file produced by `scripts/convert_tede_to_safetensors.py`,
//! which renames the PyTorch `net.0/2/4` modules to `fc0/fc1/fc2` for `candle_nn::linear`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::path::Path;

/// Input feature count: core(2)+vel(2)+ghost_target(2)+energy(1)+entropy_proxy(1).
pub const TEDE_IN_DIM: usize = 8;
const HIDDEN_DIM: usize = 16;
/// Output nudge dimensions (the protected core dims).
pub const TEDE_OUT_DIM: usize = 2;
/// Per-step delta clamp, matching the niodv4 training rollout (`torch.clamp(delta, -0.03, 0.03)`).
pub const DELTA_CLAMP: f32 = 0.03;

/// A loaded TEDE corrector MLP ready for single-vector inference.
#[derive(Debug)]
pub struct TedeCorrector {
    fc0: Linear,
    fc1: Linear,
    fc2: Linear,
    device: Device,
}

impl TedeCorrector {
    /// Load from a safetensors file (keys `fc{0,1,2}.{weight,bias}`).
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
            .with_context(|| format!("loading TEDE safetensors from {}", path.display()))?;
        Self::from_var_builder(vb, device)
    }

    /// Build from a `VarBuilder` — shared by [`Self::load`] and the unit tests.
    pub fn from_var_builder(vb: VarBuilder, device: &Device) -> Result<Self> {
        let fc0 = linear(TEDE_IN_DIM, HIDDEN_DIM, vb.pp("fc0")).context("TEDE fc0 (8→16)")?;
        let fc1 = linear(HIDDEN_DIM, HIDDEN_DIM, vb.pp("fc1")).context("TEDE fc1 (16→16)")?;
        let fc2 = linear(HIDDEN_DIM, TEDE_OUT_DIM, vb.pp("fc2")).context("TEDE fc2 (16→2)")?;
        Ok(Self {
            fc0,
            fc1,
            fc2,
            device: device.clone(),
        })
    }

    /// Run the corrector on a single 8D input, returning the 2D delta clamped to
    /// ±[`DELTA_CLAMP`].
    pub fn correct(&self, input: &[f32; TEDE_IN_DIM]) -> Result<[f32; TEDE_OUT_DIM]> {
        let x = Tensor::from_slice(input, (1, TEDE_IN_DIM), &self.device)?;
        let h = self.fc0.forward(&x)?.tanh()?;
        let h = self.fc1.forward(&h)?.tanh()?;
        let out = self.fc2.forward(&h)?;
        let out = out.clamp(-DELTA_CLAMP, DELTA_CLAMP)?;
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        Ok([v[0], v[1]])
    }

    /// Assemble the 8D input in the exact order the niodv4 trainer used
    /// (`torch.cat([core, vel, target_full[:, :2], energy, entropy_proxy])`).
    pub fn build_input(
        core: [f32; 2],
        vel: [f32; 2],
        ghost_target: [f32; 2],
        energy: f32,
        entropy_proxy: f32,
    ) -> [f32; TEDE_IN_DIM] {
        [
            core[0],
            core[1],
            vel[0],
            vel[1],
            ghost_target[0],
            ghost_target[1],
            energy,
            entropy_proxy,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::collections::HashMap;

    fn device() -> Device {
        Device::Cpu
    }

    /// Build a corrector from explicit weights so forward math is deterministic.
    fn corrector_from(
        weights: HashMap<String, Tensor>,
    ) -> TedeCorrector {
        let vb = VarBuilder::from_tensors(weights, DType::F32, &device());
        TedeCorrector::from_var_builder(vb, &device()).expect("build corrector")
    }

    fn zeros(shape: (usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &device()).unwrap()
    }
    fn zeros1(n: usize) -> Tensor {
        Tensor::zeros(n, DType::F32, &device()).unwrap()
    }

    /// All-zero weights → output is exactly the (zero) final bias → [0, 0].
    #[test]
    fn zero_weights_give_zero_delta() {
        let mut w = HashMap::new();
        w.insert("fc0.weight".into(), zeros((16, 8)));
        w.insert("fc0.bias".into(), zeros1(16));
        w.insert("fc1.weight".into(), zeros((16, 16)));
        w.insert("fc1.bias".into(), zeros1(16));
        w.insert("fc2.weight".into(), zeros((2, 16)));
        w.insert("fc2.bias".into(), zeros1(2));
        let c = corrector_from(w);
        let out = c.correct(&[0.1, -0.2, 0.3, 0.0, 0.5, -0.5, 1.0, 0.0]).unwrap();
        assert!(out[0].abs() < 1e-6 && out[1].abs() < 1e-6, "out={:?}", out);
    }

    /// A large final bias with zero weights drives both outputs past the clamp, so the
    /// result must saturate at ±DELTA_CLAMP — proves the clamp is applied.
    #[test]
    fn output_is_clamped() {
        let mut w = HashMap::new();
        w.insert("fc0.weight".into(), zeros((16, 8)));
        w.insert("fc0.bias".into(), zeros1(16));
        w.insert("fc1.weight".into(), zeros((16, 16)));
        w.insert("fc1.bias".into(), zeros1(16));
        w.insert("fc2.weight".into(), zeros((2, 16)));
        w.insert(
            "fc2.bias".into(),
            Tensor::from_slice(&[10.0f32, -10.0], 2, &device()).unwrap(),
        );
        let c = corrector_from(w);
        let out = c.correct(&[0.0; 8]).unwrap();
        assert!((out[0] - DELTA_CLAMP).abs() < 1e-6, "out0={}", out[0]);
        assert!((out[1] + DELTA_CLAMP).abs() < 1e-6, "out1={}", out[1]);
    }

    #[test]
    fn build_input_orders_features() {
        let v = TedeCorrector::build_input([1.0, 2.0], [3.0, 4.0], [5.0, 6.0], 7.0, 8.0);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    /// End-to-end load of a real converted checkpoint, if present. Produce it with
    /// `scripts/convert_tede_to_safetensors.py`. Skips when the asset is absent so the
    /// suite stays green without committing weights.
    #[test]
    fn loads_real_checkpoint_if_present() {
        let path = Path::new("runtime_assets/tede_corrector.safetensors");
        if !path.exists() {
            eprintln!("tede_corrector.safetensors absent, skipping load smoke");
            return;
        }
        let c = TedeCorrector::load(path, &device()).expect("load real checkpoint");
        let input =
            TedeCorrector::build_input([0.05, -0.1], [0.0, 0.01], [0.02, -0.11], 0.4, 0.0);
        let delta = c.correct(&input).expect("forward");
        for d in delta {
            assert!(d.is_finite(), "delta not finite: {:?}", delta);
            assert!(d.abs() <= DELTA_CLAMP + 1e-6, "delta exceeds clamp: {:?}", delta);
        }
    }
}
