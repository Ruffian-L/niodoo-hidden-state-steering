//! Rave Hidden State Codec — Rust port of niodv4 trained PyTorch codec.
//!
//! Source of truth: `niodv4/experiments/encode_decode/rave_hidden_codec.py`
//! Architecture: `conv` variant. input_channels=1, latent_dim=64, hidden_channels=32,
//! channel_multipliers=[1,2,4,8], input_dim=4096.
//!
//! Replaces the cyclic-modulo / bucket-expansion projection in
//! `project_bridge_vector_to_hidden` with the trained decoder.
//!
//! Loaded from a safetensors file produced by `scripts/convert_rave_codec_to_safetensors.py`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    conv1d, conv_transpose1d, linear, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig,
    Linear, Module, VarBuilder,
};
use std::path::Path;

const INPUT_CHANNELS: usize = 1;
const HIDDEN_CHANNELS: usize = 32;
const LATENT_DIM: usize = 64;
const INPUT_DIM: usize = 4096;
const CHANNEL_MULTIPLIERS: [usize; 4] = [1, 2, 4, 8];

const DECODER_INNER_LEN: usize = 16;
const DECODER_OUT_LEN: usize = 256;

fn down_conv_cfg() -> Conv1dConfig {
    Conv1dConfig {
        padding: 1,
        stride: 2,
        ..Default::default()
    }
}

fn initial_conv_cfg() -> Conv1dConfig {
    Conv1dConfig {
        padding: 3,
        stride: 1,
        ..Default::default()
    }
}

fn residual_conv_cfg() -> Conv1dConfig {
    Conv1dConfig {
        padding: 1,
        stride: 1,
        ..Default::default()
    }
}

fn deconv_cfg() -> ConvTranspose1dConfig {
    ConvTranspose1dConfig {
        padding: 1,
        output_padding: 0,
        stride: 2,
        dilation: 1,
        groups: 1,
    }
}

fn final_conv_cfg() -> Conv1dConfig {
    Conv1dConfig {
        padding: 1,
        stride: 1,
        ..Default::default()
    }
}

#[derive(Debug)]
pub struct ResidualBlock1D {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl ResidualBlock1D {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        // PyTorch: nn.Sequential([Conv1d(c,c,k=3,p=1), GELU, Conv1d(c,c,k=3,p=1)])
        // State dict keys: net.0, net.2
        let net = vb.pp("net");
        let conv1 = conv1d(channels, channels, 3, residual_conv_cfg(), net.pp("0"))
            .with_context(|| format!("ResidualBlock1D({channels}).net.0"))?;
        let conv2 = conv1d(channels, channels, 3, residual_conv_cfg(), net.pp("2"))
            .with_context(|| format!("ResidualBlock1D({channels}).net.2"))?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(x)?.gelu()?;
        let h = self.conv2.forward(&h)?;
        Ok((x + h)?)
    }
}

#[derive(Debug)]
pub struct RaveEncoder {
    initial_conv: Conv1d,            // conv.0:  (1 -> 32, k=7, p=3)
    down_convs: Vec<Conv1d>,         // conv.{2,5,8,11}
    residuals: Vec<ResidualBlock1D>, // conv.{4,7,10,13}
    mu_head: Linear,                 // mu: (256 -> 64)
}

impl RaveEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv = vb.pp("conv");
        let initial_conv = conv1d(
            INPUT_CHANNELS,
            HIDDEN_CHANNELS,
            7,
            initial_conv_cfg(),
            conv.pp("0"),
        )
        .context("encoder.conv.0 (initial Conv1d)")?;

        let mut down_convs = Vec::with_capacity(4);
        let mut residuals = Vec::with_capacity(4);

        // Encoder Sequential indices: 0(initial), 1(GELU),
        //                              2(down0), 3(GELU),  4(res0),
        //                              5(down1), 6(GELU),  7(res1),
        //                              8(down2), 9(GELU), 10(res2),
        //                             11(down3),12(GELU), 13(res3)
        let down_indices = [2usize, 5, 8, 11];
        let res_indices = [4usize, 7, 10, 13];

        let mut current_channels = HIDDEN_CHANNELS;
        for (i, mult) in CHANNEL_MULTIPLIERS.iter().enumerate() {
            let next_channels = HIDDEN_CHANNELS * mult;
            let down = conv1d(
                current_channels,
                next_channels,
                4,
                down_conv_cfg(),
                conv.pp(&down_indices[i].to_string()),
            )
            .with_context(|| format!("encoder.conv.{} downsample", down_indices[i]))?;
            let res = ResidualBlock1D::load(next_channels, conv.pp(&res_indices[i].to_string()))?;
            down_convs.push(down);
            residuals.push(res);
            current_channels = next_channels;
        }

        let mu_head = linear(current_channels, LATENT_DIM, vb.pp("mu"))
            .context("encoder.mu Linear(256 -> 64)")?;

        Ok(Self {
            initial_conv,
            down_convs,
            residuals,
            mu_head,
        })
    }

    /// Encode (B, 1, 4096) → (B, 64) deterministic mu.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.initial_conv.forward(x)?.gelu()?;
        for (down, res) in self.down_convs.iter().zip(self.residuals.iter()) {
            h = down.forward(&h)?.gelu()?;
            h = res.forward(&h)?;
        }
        // AdaptiveAvgPool1d(1) over the time dim: mean across last axis.
        // h shape: (B, 256, 1) after pool. Squeeze to (B, 256).
        let pooled = h.mean(2)?;
        let mu = self.mu_head.forward(&pooled)?;
        Ok(mu)
    }
}

#[derive(Debug)]
pub struct RaveDecoder {
    project: Linear,                 // project: 64 -> 4096 (reshape to (B, 256, 16))
    deconvs: Vec<ConvTranspose1d>,   // blocks.{0, 3, 6, 9}
    residuals: Vec<ResidualBlock1D>, // blocks.{2, 5, 8} (only first 3 deconvs have a residual after)
    final_conv: Conv1d,              // blocks.11: 32 -> 1, k=3, p=1
}

impl RaveDecoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        // Decoder Sequential indices:
        //   0  ConvTranspose1d(256 -> 128)
        //   1  GELU
        //   2  ResidualBlock1D(128)
        //   3  ConvTranspose1d(128 -> 64)
        //   4  GELU
        //   5  ResidualBlock1D(64)
        //   6  ConvTranspose1d(64 -> 32)
        //   7  GELU
        //   8  ResidualBlock1D(32)
        //   9  ConvTranspose1d(32 -> 32)
        //  10  GELU
        //  11  Conv1d(32 -> 1, k=3, p=1)
        let project = linear(
            LATENT_DIM,
            HIDDEN_CHANNELS * 8 * DECODER_INNER_LEN,
            vb.pp("project"),
        )
        .context("decoder.project Linear(64 -> 4096)")?;

        let blocks = vb.pp("blocks");

        let deconv_specs: [(usize, usize, usize); 4] = [
            // (in, out, sequential_index)
            (HIDDEN_CHANNELS * 8, HIDDEN_CHANNELS * 4, 0), // 256 -> 128
            (HIDDEN_CHANNELS * 4, HIDDEN_CHANNELS * 2, 3), // 128 -> 64
            (HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS, 6),     // 64 -> 32
            (HIDDEN_CHANNELS, HIDDEN_CHANNELS, 9),         // 32 -> 32
        ];
        let res_specs: [(usize, usize); 3] = [
            (HIDDEN_CHANNELS * 4, 2), // 128
            (HIDDEN_CHANNELS * 2, 5), // 64
            (HIDDEN_CHANNELS, 8),     // 32
        ];

        let mut deconvs = Vec::with_capacity(4);
        for (in_c, out_c, idx) in deconv_specs {
            let dc = conv_transpose1d(in_c, out_c, 4, deconv_cfg(), blocks.pp(&idx.to_string()))
                .with_context(|| {
                    format!("decoder.blocks.{idx} ConvTranspose1d({in_c}->{out_c})")
                })?;
            deconvs.push(dc);
        }

        let mut residuals = Vec::with_capacity(3);
        for (channels, idx) in res_specs {
            residuals.push(ResidualBlock1D::load(
                channels,
                blocks.pp(&idx.to_string()),
            )?);
        }

        let final_conv = conv1d(
            HIDDEN_CHANNELS,
            INPUT_CHANNELS,
            3,
            final_conv_cfg(),
            blocks.pp("11"),
        )
        .context("decoder.blocks.11 final Conv1d(32 -> 1)")?;

        Ok(Self {
            project,
            deconvs,
            residuals,
            final_conv,
        })
    }

    /// Decode (B, 64) → (B, 4096). Drops the unit channel automatically.
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let batch = z.dim(0)?;
        // project + reshape to (B, 256, 16)
        let projected = self.project.forward(z)?;
        let x = projected.reshape((batch, HIDDEN_CHANNELS * 8, DECODER_INNER_LEN))?;

        // 3 stages with residuals, then 1 stage without
        let mut h = x;
        for i in 0..3 {
            h = self.deconvs[i].forward(&h)?.gelu()?;
            h = self.residuals[i].forward(&h)?;
        }
        h = self.deconvs[3].forward(&h)?.gelu()?;
        // Final Conv1d -> (B, 1, 256)
        let out_short = self.final_conv.forward(&h)?;
        // Upsample 256 -> 4096 (16x). Candle's `interpolate1d` (UpsampleNearest1d kernel) is
        // not CUDA-implemented, so do nearest-neighbor manually: unsqueeze -> broadcast -> reshape.
        // (B, 1, 256) -> (B, 1, 256, 1) -> (B, 1, 256, 16) -> (B, 1, 4096)
        let scale = INPUT_DIM / DECODER_OUT_LEN;
        let dims = out_short.dims();
        debug_assert_eq!(dims.len(), 3);
        let (batch, channels, time) = (dims[0], dims[1], dims[2]);
        let unsq = out_short.unsqueeze(3)?; // (B, 1, 256, 1)
        let bcast = unsq.broadcast_as((batch, channels, time, scale))?;
        let upsampled = bcast.reshape((batch, channels, time * scale))?;
        // (B, 1, 4096) -> (B, 4096)
        let flat = upsampled.i((.., 0, ..))?;
        Ok(flat)
    }
}

#[derive(Debug)]
pub struct RaveCodec {
    pub encoder: RaveEncoder,
    pub decoder: RaveDecoder,
    pub device: Device,
}

impl RaveCodec {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)
                .with_context(|| format!("loading rave codec from {}", path.display()))?
        };
        let encoder = RaveEncoder::load(vb.pp("encoder"))?;
        let decoder = RaveDecoder::load(vb.pp("decoder"))?;
        Ok(Self {
            encoder,
            decoder,
            device: device.clone(),
        })
    }

    /// Encode (B, 4096) hidden state to (B, 64) deterministic latent.
    pub fn encode(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.dims();
        let prepared = match dims.len() {
            1 => hidden.unsqueeze(0)?.unsqueeze(1)?, // (4096) -> (1, 1, 4096)
            2 => hidden.unsqueeze(1)?,               // (B, 4096) -> (B, 1, 4096)
            3 => hidden.clone(),                     // already (B, C, 4096)
            other => anyhow::bail!("RaveCodec::encode unsupported rank {other}"),
        };
        Ok(self.encoder.forward(&prepared)?)
    }

    /// Decode (B, 64) latent to (B, 4096) hidden state.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        let dims = latent.dims();
        let prepared = match dims.len() {
            1 => latent.unsqueeze(0)?, // (64) -> (1, 64)
            2 => latent.clone(),
            other => anyhow::bail!("RaveCodec::decode unsupported rank {other}"),
        };
        Ok(self.decoder.forward(&prepared)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_smoke() -> Result<()> {
        // Only meaningful if the codec safetensors exist; otherwise skip.
        let path = Path::new("niodoo/runtime_assets/rave_codec.safetensors");
        if !path.exists() {
            eprintln!("rave_codec.safetensors absent, skipping shape_smoke");
            return Ok(());
        }
        let device = Device::Cpu;
        let codec = RaveCodec::load(path, &device)?;
        let z = Tensor::zeros((2, LATENT_DIM), DType::F32, &device)?;
        let out = codec.decode(&z)?;
        assert_eq!(out.dims(), &[2, INPUT_DIM]);
        let h = Tensor::zeros((2, INPUT_DIM), DType::F32, &device)?;
        let mu = codec.encode(&h)?;
        assert_eq!(mu.dims(), &[2, LATENT_DIM]);
        Ok(())
    }

    /// Math smoke for the codec-mediated specialist correction force used by
    /// `PrincipiaEngine::try_apply_specialist_correction_force`. Verifies:
    /// (1) a non-zero 2D delta added to the encoded latent produces a non-zero 4096D force
    ///     after `decode(z + delta) - decode(z)`,
    /// (2) a zero delta produces a zero force.
    /// Skips when the safetensors file is absent. When it runs, this is the actual
    /// end-to-end correctness check for the hollow-flag fix.
    #[test]
    fn codec_mediated_specialist_force_smoke() -> Result<()> {
        let path = Path::new("niodoo/runtime_assets/rave_codec.safetensors");
        if !path.exists() {
            eprintln!(
                "rave_codec.safetensors absent, skipping codec_mediated_specialist_force_smoke"
            );
            return Ok(());
        }
        let device = Device::Cpu;
        let codec = RaveCodec::load(path, &device)?;

        let probe = Tensor::randn(0f32, 1.0f32, (INPUT_DIM,), &device)?;
        let probe_2d = probe.unsqueeze(0)?;
        let z = codec.encode(&probe_2d)?;

        // Non-zero delta in the first 2 latent dims (matches phase2 specialist semantics).
        let mut delta_64 = vec![0f32; LATENT_DIM];
        delta_64[0] = 0.5;
        delta_64[1] = -0.3;
        let delta_t = Tensor::from_vec(delta_64, (1, LATENT_DIM), &device)?;
        let z_prime = (&z + &delta_t)?;

        let decoded_z = codec.decode(&z)?;
        let decoded_zp = codec.decode(&z_prime)?;
        let force = (decoded_zp.flatten_all()? - decoded_z.flatten_all()?)?;
        let force_norm: f32 = force.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
        assert!(
            force_norm > 1e-4,
            "non-zero specialist delta must produce non-zero codec force; got {force_norm}"
        );

        // Zero delta should produce ~zero force (same input → same decode → zero diff).
        let zero_delta = Tensor::zeros((1, LATENT_DIM), DType::F32, &device)?;
        let z_unchanged = (&z + &zero_delta)?;
        let decoded_unchanged = codec.decode(&z_unchanged)?;
        let zero_force = (decoded_unchanged.flatten_all()? - decoded_z.flatten_all()?)?;
        let zero_force_norm: f32 = zero_force.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
        assert!(
            zero_force_norm < 1e-3,
            "zero delta must produce ~zero force; got {zero_force_norm}"
        );

        Ok(())
    }
}
