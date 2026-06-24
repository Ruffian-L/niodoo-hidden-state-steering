use crate::physics::naked_llama::{
    LayerKvCacheSnapshot, ModelKvCacheSnapshot, PhysicsEngine, RmsNorm,
};
use anyhow::{Context, Result};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Module};
use serde::Serialize;

const MAX_SEQ_LEN: usize = 8192;

#[derive(Debug, Clone, Serialize)]
pub struct Qwen35GgufMetadata {
    pub architecture: String,
    pub hidden_size: usize,
    pub layer_count: usize,
    pub vocab_size: usize,
    pub context_length: Option<usize>,
    pub full_attention_layers: Vec<usize>,
    pub linear_attention_layers: Vec<usize>,
    pub first_tensor_names: Vec<String>,
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> candle_core::Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.forward(&xs.contiguous()?)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(xs)?)?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(gate * up)?)
    }
}

#[derive(Debug, Clone)]
struct FullAttention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    kv_cache: Option<(Tensor, Tensor)>,
}

#[derive(Debug, Clone)]
struct LinearAttention {
    qkv: QMatMul,
    gate: QMatMul,
    beta: QMatMul,
    alpha: QMatMul,
    conv1d: Tensor,
    dt: Tensor,
    a: Tensor,
    norm_weight: Tensor,
    out: QMatMul,
    conv_state: Tensor,
    ssm_state: Tensor,
}

#[derive(Debug, Clone)]
enum Qwen35Attention {
    Full(FullAttention),
    Linear(LinearAttention),
}

#[derive(Debug, Clone)]
struct Qwen35Layer {
    attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    attn: Qwen35Attention,
    mlp: Mlp,
}

#[derive(Debug, Clone)]
pub struct QuantizedQwen35Hybrid {
    tok_embeddings: Embedding,
    layers: Vec<Qwen35Layer>,
    norm: RmsNorm,
    output: QMatMul,
    metadata: Qwen35GgufMetadata,
    hidden_dim: usize,
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    linear_key_heads: usize,
    linear_value_heads: usize,
    linear_head_dim: usize,
}

fn gguf_metadata_string(content: &gguf_file::Content, key: &str) -> Option<String> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::String(value)) => Some(value.clone()),
        _ => None,
    }
}

fn gguf_metadata_usize(content: &gguf_file::Content, key: &str) -> Option<usize> {
    content
        .metadata
        .get(key)
        .and_then(|value| value.to_u32().ok())
        .map(|value| value as usize)
}

fn gguf_metadata_f32(content: &gguf_file::Content, key: &str) -> Option<f32> {
    content
        .metadata
        .get(key)
        .and_then(|value| value.to_f32().ok())
}

pub fn summarize_qwen35_metadata(
    content: &gguf_file::Content,
    tokenizer_vocab_size: usize,
) -> Result<Qwen35GgufMetadata> {
    let architecture = gguf_metadata_string(content, "general.architecture")
        .unwrap_or_else(|| "qwen35".to_string());
    let hidden_size = gguf_metadata_usize(content, "qwen35.embedding_length")
        .context("qwen35 GGUF missing qwen35.embedding_length")?;
    let layer_count = gguf_metadata_usize(content, "qwen35.block_count")
        .context("qwen35 GGUF missing qwen35.block_count")?;
    let context_length = gguf_metadata_usize(content, "qwen35.context_length");
    let vocab_size = gguf_metadata_usize(content, "qwen35.vocab_size")
        .or_else(|| {
            content
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|value| match value {
                    gguf_file::Value::Array(values) => Some(values.len()),
                    _ => None,
                })
        })
        .unwrap_or(tokenizer_vocab_size);

    let mut full_attention_layers = Vec::new();
    let mut linear_attention_layers = Vec::new();
    for layer_idx in 0..layer_count {
        let full_key = format!("blk.{layer_idx}.attn_q.weight");
        let linear_key = format!("blk.{layer_idx}.attn_qkv.weight");
        if content.tensor_infos.contains_key(&full_key) {
            full_attention_layers.push(layer_idx);
        } else if content.tensor_infos.contains_key(&linear_key) {
            linear_attention_layers.push(layer_idx);
        }
    }

    let mut first_tensor_names: Vec<String> = content
        .tensor_infos
        .keys()
        .filter(|name| {
            name.as_str() == "token_embd.weight"
                || name.as_str() == "output.weight"
                || name.as_str() == "output_norm.weight"
                || name.starts_with("blk.")
        })
        .cloned()
        .collect();
    first_tensor_names.sort();
    first_tensor_names.truncate(32);

    Ok(Qwen35GgufMetadata {
        architecture,
        hidden_size,
        layer_count,
        vocab_size,
        context_length,
        full_attention_layers,
        linear_attention_layers,
        first_tensor_names,
    })
}

fn precompute_freqs_cis(
    rope_dim: usize,
    freq_base: f32,
    device: &Device,
) -> candle_core::Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..rope_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / rope_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((idx_theta.cos()?, idx_theta.sin()?))
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}

fn softplus(xs: &Tensor) -> candle_core::Result<Tensor> {
    (xs.exp()? + 1.0)?.log()
}

fn l2_norm_last_dim(xs: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let denom = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    xs.broadcast_div(&denom)
}

fn rms_norm_with_weight(xs: &Tensor, weight: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let denom = (xs.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    xs.broadcast_div(&denom)?.broadcast_mul(weight)
}

impl QuantizedQwen35Hybrid {
    pub fn load_gguf<R: std::io::Seek + std::io::Read>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        tokenizer_vocab_size: usize,
    ) -> Result<Self> {
        let metadata = summarize_qwen35_metadata(&content, tokenizer_vocab_size)?;
        let md_get = |key: &str| match content.metadata.get(key) {
            None => candle_core::bail!("cannot find {key} in metadata"),
            Some(value) => Ok(value),
        };

        let layer_count = metadata.layer_count;
        let hidden_dim = metadata.hidden_size;
        let head_count = md_get("qwen35.attention.head_count")?.to_u32()? as usize;
        let kv_head_count = md_get("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen35.attention.key_length")?.to_u32()? as usize;
        let rope_dim = md_get("qwen35.rope.dimension_count")?.to_u32()? as usize;
        let rms_eps = md_get("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base =
            gguf_metadata_f32(&content, "qwen35.rope.freq_base").unwrap_or(10_000_000.0);
        let linear_inner = md_get("qwen35.ssm.inner_size")?.to_u32()? as usize;
        let linear_key_heads = md_get("qwen35.ssm.group_count")?.to_u32()? as usize;
        let linear_value_heads = md_get("qwen35.ssm.time_step_rank")?.to_u32()? as usize;
        let linear_head_dim = md_get("qwen35.ssm.state_size")?.to_u32()? as usize;
        let conv_kernel = md_get("qwen35.ssm.conv_kernel")?.to_u32()? as usize;
        let conv_channels = linear_inner + 2 * linear_key_heads * linear_head_dim;
        let (cos, sin) = precompute_freqs_cis(rope_dim, rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        let tok_embeddings_q = content.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let output =
            QMatMul::from_qtensor(match content.tensor(reader, "output.weight", device) {
                Ok(tensor) => tensor,
                Err(_) => tok_embeddings_q,
            })?;
        let norm = RmsNorm::from_qtensor(
            content.tensor(reader, "output_norm.weight", device)?,
            rms_eps,
        )?;

        let mut layers = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            let prefix = format!("blk.{layer_idx}");
            let attn_norm = RmsNorm::from_qtensor(
                content.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_eps,
            )?;
            let post_attn_norm = RmsNorm::from_qtensor(
                content.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_eps,
            )?;
            let mlp = Mlp {
                gate: QMatMul::from_qtensor(content.tensor(
                    reader,
                    &format!("{prefix}.ffn_gate.weight"),
                    device,
                )?)?,
                up: QMatMul::from_qtensor(content.tensor(
                    reader,
                    &format!("{prefix}.ffn_up.weight"),
                    device,
                )?)?,
                down: QMatMul::from_qtensor(content.tensor(
                    reader,
                    &format!("{prefix}.ffn_down.weight"),
                    device,
                )?)?,
            };
            let attn = if content
                .tensor_infos
                .contains_key(&format!("{prefix}.attn_q.weight"))
            {
                Qwen35Attention::Full(FullAttention {
                    wq: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.attn_q.weight"),
                        device,
                    )?)?,
                    wk: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.attn_k.weight"),
                        device,
                    )?)?,
                    wv: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.attn_v.weight"),
                        device,
                    )?)?,
                    wo: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.attn_output.weight"),
                        device,
                    )?)?,
                    q_norm: RmsNorm::from_qtensor(
                        content.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                        rms_eps,
                    )?,
                    k_norm: RmsNorm::from_qtensor(
                        content.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                        rms_eps,
                    )?,
                    kv_cache: None,
                })
            } else {
                let conv_state =
                    Tensor::zeros((conv_kernel - 1, conv_channels), DType::F32, device)?;
                let ssm_state = Tensor::zeros(
                    (linear_value_heads, linear_head_dim, linear_head_dim),
                    DType::F32,
                    device,
                )?;
                Qwen35Attention::Linear(LinearAttention {
                    qkv: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.attn_qkv.weight"),
                        device,
                    )?)?,
                    gate: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.attn_gate.weight"),
                        device,
                    )?)?,
                    beta: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.ssm_beta.weight"),
                        device,
                    )?)?,
                    alpha: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.ssm_alpha.weight"),
                        device,
                    )?)?,
                    conv1d: content
                        .tensor(reader, &format!("{prefix}.ssm_conv1d.weight"), device)?
                        .dequantize(device)?
                        .t()?
                        .contiguous()?,
                    dt: content
                        .tensor(reader, &format!("{prefix}.ssm_dt.bias"), device)?
                        .dequantize(device)?,
                    a: content
                        .tensor(reader, &format!("{prefix}.ssm_a"), device)?
                        .dequantize(device)?,
                    norm_weight: content
                        .tensor(reader, &format!("{prefix}.ssm_norm.weight"), device)?
                        .dequantize(device)?,
                    out: QMatMul::from_qtensor(content.tensor(
                        reader,
                        &format!("{prefix}.ssm_out.weight"),
                        device,
                    )?)?,
                    conv_state,
                    ssm_state,
                })
            };
            layers.push(Qwen35Layer {
                attn_norm,
                post_attn_norm,
                attn,
                mlp,
            });
        }

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, hidden_dim),
            layers,
            norm,
            output,
            metadata,
            hidden_dim,
            head_count,
            kv_head_count,
            head_dim,
            rope_dim,
            cos,
            sin,
            neg_inf,
            linear_key_heads,
            linear_value_heads,
            linear_head_dim,
        })
    }

    pub fn metadata(&self) -> &Qwen35GgufMetadata {
        &self.metadata
    }

    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    pub fn embed_tokens_forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.tok_embeddings.forward(input)
    }

    pub fn clear_kv_cache(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            match &mut layer.attn {
                Qwen35Attention::Full(attn) => {
                    attn.kv_cache = None;
                }
                Qwen35Attention::Linear(attn) => {
                    attn.conv_state = attn.conv_state.zeros_like()?.detach();
                    attn.ssm_state = attn.ssm_state.zeros_like()?.detach();
                }
            }
        }
        Ok(())
    }

    pub fn export_kv_cache_snapshot(&self) -> Result<ModelKvCacheSnapshot> {
        let mut layers = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            match &layer.attn {
                Qwen35Attention::Full(attn) => {
                    if let Some((k, v)) = &attn.kv_cache {
                        let k = k.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        let v = v.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                        let k_shape = k.dims4()?;
                        let v_shape = v.dims4()?;
                        layers.push(LayerKvCacheSnapshot {
                            layer_idx,
                            state_kind: Some("attention_kv".to_string()),
                            k_shape: Some([k_shape.0, k_shape.1, k_shape.2, k_shape.3]),
                            v_shape: Some([v_shape.0, v_shape.1, v_shape.2, v_shape.3]),
                            k_data: Some(k.flatten_all()?.to_vec1::<f32>()?),
                            v_data: Some(v.flatten_all()?.to_vec1::<f32>()?),
                            conv_shape: None,
                            conv_data: None,
                            ssm_shape: None,
                            ssm_data: None,
                        });
                    }
                }
                Qwen35Attention::Linear(attn) => {
                    let conv = attn
                        .conv_state
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?;
                    let ssm = attn
                        .ssm_state
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?;
                    let conv_shape = conv.dims2()?;
                    let ssm_shape = ssm.dims3()?;
                    layers.push(LayerKvCacheSnapshot {
                        layer_idx,
                        state_kind: Some("qwen35_linear".to_string()),
                        k_shape: None,
                        v_shape: None,
                        k_data: None,
                        v_data: None,
                        conv_shape: Some([conv_shape.0, conv_shape.1]),
                        conv_data: Some(conv.flatten_all()?.to_vec1::<f32>()?),
                        ssm_shape: Some([ssm_shape.0, ssm_shape.1, ssm_shape.2]),
                        ssm_data: Some(ssm.flatten_all()?.to_vec1::<f32>()?),
                    });
                }
            }
        }
        Ok(ModelKvCacheSnapshot {
            layer_count: self.layers.len(),
            layers,
        })
    }

    pub fn import_kv_cache_snapshot(
        &mut self,
        snapshot: &ModelKvCacheSnapshot,
        device: &Device,
    ) -> Result<()> {
        self.clear_kv_cache()?;
        let layer_count = self.layers.len();
        for layer in &snapshot.layers {
            let target = self.layers.get_mut(layer.layer_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "qwen35 snapshot layer {} out of range {}",
                    layer.layer_idx,
                    layer_count
                )
            })?;
            match &mut target.attn {
                Qwen35Attention::Full(attn) => {
                    let k_shape = layer.k_shape.ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 full-attention layer {} missing k_shape",
                            layer.layer_idx
                        )
                    })?;
                    let v_shape = layer.v_shape.ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 full-attention layer {} missing v_shape",
                            layer.layer_idx
                        )
                    })?;
                    let k_data = layer.k_data.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 full-attention layer {} missing k_data",
                            layer.layer_idx
                        )
                    })?;
                    let v_data = layer.v_data.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 full-attention layer {} missing v_data",
                            layer.layer_idx
                        )
                    })?;
                    let k = Tensor::from_vec(
                        k_data.clone(),
                        (k_shape[0], k_shape[1], k_shape[2], k_shape[3]),
                        &Device::Cpu,
                    )?
                    .to_device(device)?;
                    let v = Tensor::from_vec(
                        v_data.clone(),
                        (v_shape[0], v_shape[1], v_shape[2], v_shape[3]),
                        &Device::Cpu,
                    )?
                    .to_device(device)?;
                    attn.kv_cache = Some((k, v));
                }
                Qwen35Attention::Linear(attn) => {
                    let conv_shape = layer.conv_shape.ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 linear-attention layer {} missing conv_shape",
                            layer.layer_idx
                        )
                    })?;
                    let ssm_shape = layer.ssm_shape.ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 linear-attention layer {} missing ssm_shape",
                            layer.layer_idx
                        )
                    })?;
                    let conv_data = layer.conv_data.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 linear-attention layer {} missing conv_data",
                            layer.layer_idx
                        )
                    })?;
                    let ssm_data = layer.ssm_data.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "qwen35 linear-attention layer {} missing ssm_data",
                            layer.layer_idx
                        )
                    })?;
                    attn.conv_state = Tensor::from_vec(
                        conv_data.clone(),
                        (conv_shape[0], conv_shape[1]),
                        &Device::Cpu,
                    )?
                    .to_device(device)?;
                    attn.ssm_state = Tensor::from_vec(
                        ssm_data.clone(),
                        (ssm_shape[0], ssm_shape[1], ssm_shape[2]),
                        &Device::Cpu,
                    )?
                    .to_device(device)?;
                }
            }
        }
        Ok(())
    }

    fn forward_full_attn_static(
        head_count: usize,
        kv_head_count: usize,
        head_dim: usize,
        rope_dim: usize,
        cos: &Tensor,
        sin: &Tensor,
        neg_inf: &Tensor,
        attn: &mut FullAttention,
        xs: &Tensor,
        index_pos: usize,
    ) -> candle_core::Result<Tensor> {
        use crate::physics::forward_profile::{measure, Stage};
        let dev = xs.device().clone();

        let (b_sz, seq_len, _) = xs.dims3()?;
        let q_full = measure(Stage::Wq, &dev, || attn.wq.forward(xs))?;
        let q_full = q_full.reshape((b_sz, seq_len, head_count, 2, head_dim))?;
        let q = q_full.i((.., .., .., 0, ..))?.transpose(1, 2)?;
        let gate =
            q_full
                .i((.., .., .., 1, ..))?
                .reshape((b_sz, seq_len, head_count * head_dim))?;
        let k = measure(Stage::Wk, &dev, || attn.wk.forward(xs))?
            .reshape((b_sz, seq_len, kv_head_count, head_dim))?
            .transpose(1, 2)?;
        let v = measure(Stage::Wv, &dev, || attn.wv.forward(xs))?
            .reshape((b_sz, seq_len, kv_head_count, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = measure(Stage::QkNorm, &dev, || -> candle_core::Result<_> {
            let q = attn.q_norm.forward(&q.contiguous()?)?;
            let k = attn.k_norm.forward(&k.contiguous()?)?;
            Ok((q, k))
        })?;
        let (q, k) = measure(Stage::Rope, &dev, || -> candle_core::Result<_> {
            let apply_rope = |t: &Tensor| -> candle_core::Result<Tensor> {
                let rot = t.narrow(3, 0, rope_dim)?;
                let pass = t.narrow(3, rope_dim, head_dim - rope_dim)?;
                let cos = cos.narrow(0, index_pos, seq_len)?;
                let sin = sin.narrow(0, index_pos, seq_len)?;
                let rot = candle_nn::rotary_emb::rope_i(&rot.contiguous()?, &cos, &sin)?;
                Tensor::cat(&[&rot, &pass], 3)
            };
            Ok((apply_rope(&q)?, apply_rope(&k)?))
        })?;

        let (k, v) = measure(Stage::KvCat, &dev, || -> candle_core::Result<_> {
            let (k, v) = match &attn.kv_cache {
                None => (k, v),
                Some((k_cache, v_cache)) if index_pos > 0 => (
                    Tensor::cat(&[k_cache, &k], 2)?,
                    Tensor::cat(&[v_cache, &v], 2)?,
                ),
                Some(_) => (k, v),
            };
            attn.kv_cache = Some((k.clone(), v.clone()));
            Ok((k, v))
        })?;
        let k_rep = crate::physics::naked_llama::repeat_kv(k, head_count / kv_head_count)?;
        let v_rep = crate::physics::naked_llama::repeat_kv(v, head_count / kv_head_count)?;
        let att_scores = measure(Stage::AttQkMatmul, &dev, || {
            q.contiguous()?
                .matmul(&k_rep.t()?.contiguous()?)
                .and_then(|s| s / (head_dim as f64).sqrt())
        })?;
        // For seq_len == 1 (decode) the causal mask would be all zeros: position i=0
        // attends to all keys at j ∈ [0, index_pos], no future to hide. Skip the CPU
        // Vec build, the H2D upload, and the where_cond. Only prefill (seq_len > 1)
        // actually needs the mask.
        let att_scores = if seq_len == 1 {
            att_scores
        } else {
            let total_len = index_pos + seq_len;
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..total_len).map(move |j| u8::from(j > index_pos + i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, total_len), xs.device())?;
            let mask = mask.broadcast_as(att_scores.shape())?;
            masked_fill(&att_scores, &mask, neg_inf)?
        };
        let att_probs = measure(Stage::Softmax, &dev, || {
            candle_nn::ops::softmax_last_dim(&att_scores)
        })?;
        let y = measure(Stage::AttVMatmul, &dev, || {
            att_probs.contiguous()?.matmul(&v_rep.contiguous()?)
        })?;
        let y = y
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, head_count * head_dim))?;
        let y = (y * candle_nn::ops::sigmoid(&gate)?)?;
        measure(Stage::OutProj, &dev, || attn.wo.forward(&y))
    }

    fn forward_linear_attn_static(
        linear_key_heads: usize,
        linear_value_heads: usize,
        linear_head_dim: usize,
        attn: &mut LinearAttention,
        xs: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        if b_sz != 1 || seq_len != 1 {
            candle_core::bail!("qwen35 linear attention expects token-by-token forward");
        }
        let key_width = linear_key_heads * linear_head_dim;
        let value_width = linear_value_heads * linear_head_dim;
        let conv_channels = 2 * key_width + value_width;

        let qkv = attn.qkv.forward(xs)?.reshape((conv_channels,))?;
        let z = attn
            .gate
            .forward(xs)?
            .reshape((linear_value_heads, linear_head_dim))?;
        let beta =
            candle_nn::ops::sigmoid(&attn.beta.forward(xs)?.reshape((linear_value_heads,))?)?;
        let alpha = attn.alpha.forward(xs)?.reshape((linear_value_heads,))?;
        let gate = (softplus(&(alpha + &attn.dt)?)? * &attn.a)?;

        let conv_input = Tensor::cat(&[&attn.conv_state, &qkv.reshape((1, conv_channels))?], 0)?;
        let conv_state_rows = attn.conv_state.dim(0)?;
        attn.conv_state = conv_input.narrow(0, 1, conv_state_rows)?.detach();
        let conv = (conv_input * &attn.conv1d)?.sum(0)?;
        let conv = candle_nn::ops::silu(&conv)?;

        let q = conv
            .narrow(0, 0, key_width)?
            .reshape((linear_key_heads, linear_head_dim))?;
        let k = conv
            .narrow(0, key_width, key_width)?
            .reshape((linear_key_heads, linear_head_dim))?;
        let v = conv
            .narrow(0, 2 * key_width, value_width)?
            .reshape((linear_value_heads, linear_head_dim))?;
        let q = l2_norm_last_dim(&q, 1e-6)?;
        let k = l2_norm_last_dim(&k, 1e-6)?;
        let repeat = linear_value_heads / linear_key_heads;
        let q =
            Tensor::cat(&vec![&q; repeat], 0)?.reshape((linear_value_heads, linear_head_dim))?;
        let k =
            Tensor::cat(&vec![&k; repeat], 0)?.reshape((linear_value_heads, linear_head_dim))?;

        let q = (q / (linear_head_dim as f64).sqrt())?;
        let exp_gate = gate.exp()?.reshape((linear_value_heads, 1, 1))?;
        let mut state = attn.ssm_state.broadcast_mul(&exp_gate)?;
        let sk = state
            .broadcast_mul(&k.reshape((linear_value_heads, linear_head_dim, 1))?)?
            .sum(1)?;
        let d = (&v - sk)?.broadcast_mul(&beta.reshape((linear_value_heads, 1))?)?;
        let kd = k
            .reshape((linear_value_heads, linear_head_dim, 1))?
            .broadcast_mul(&d.reshape((linear_value_heads, 1, linear_head_dim))?)?;
        state = (state + kd)?;
        let out = state
            .broadcast_mul(&q.reshape((linear_value_heads, linear_head_dim, 1))?)?
            .sum(1)?;
        attn.ssm_state = state.detach();

        let out = rms_norm_with_weight(&out, &attn.norm_weight, 1e-6)?;
        let out = (out * candle_nn::ops::silu(&z)?)?;
        let out = out.reshape((1, 1, linear_value_heads * linear_head_dim))?;
        attn.out.forward(&out)
    }

    fn forward_one<P: PhysicsEngine>(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        physics: &mut P,
        ghost_vector: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        use crate::physics::forward_profile::{measure, note_forward_call, Stage};
        let dev = input.device().clone();

        // Hoisted out of the per-layer loop: physics range, blend params, and
        // the blend scalar tensor don't change layer-to-layer within a single
        // forward call. Was 32× per token of redundant work for an N-layer
        // model. blend_t in particular allocated a 1-elem GPU tensor per layer.
        let (physics_start_layer, physics_end_layer) = physics.get_physics_layer_range();
        let physics_blend_factor = physics.get_physics_blend();
        let physics_use_mult = physics.use_multiplicative_blend();
        let blend_t = Tensor::new(physics_blend_factor, &dev)?;

        let mut layer_in = measure(Stage::Embed, &dev, || self.tok_embeddings.forward(input))?;
        for layer_idx in 0..self.layers.len() {
            let residual = layer_in.clone();
            let xs = measure(Stage::AttnNorm, &dev, || {
                self.layers[layer_idx]
                    .attn_norm
                    .forward(&layer_in.contiguous()?)
            })?;
            let mut attn_out = match &mut self.layers[layer_idx].attn {
                Qwen35Attention::Full(attn) => Self::forward_full_attn_static(
                    self.head_count,
                    self.kv_head_count,
                    self.head_dim,
                    self.rope_dim,
                    &self.cos,
                    &self.sin,
                    &self.neg_inf,
                    attn,
                    &xs,
                    index_pos,
                )?,
                Qwen35Attention::Linear(attn) => Self::forward_linear_attn_static(
                    self.linear_key_heads,
                    self.linear_value_heads,
                    self.linear_head_dim,
                    attn,
                    &xs,
                )?,
            };
            let in_primary = layer_idx >= physics_start_layer && layer_idx <= physics_end_layer;
            let early_worker = physics.physics_invoke_for_early_worker_influence(layer_idx);
            if in_primary || early_worker {
                let force_delta = measure(Stage::Physics, &dev, || {
                    physics.apply_forces(&attn_out, layer_idx, ghost_vector)
                })?;
                attn_out = measure(Stage::Blend, &dev, || -> candle_core::Result<Tensor> {
                    if physics_use_mult {
                        let scaled_force = force_delta.broadcast_mul(&blend_t)?;
                        let scale = (scaled_force.ones_like()? + scaled_force)?;
                        attn_out.broadcast_mul(&scale)
                    } else {
                        attn_out + force_delta.broadcast_mul(&blend_t)?
                    }
                })?;
            }
            let post_attn = (attn_out + residual)?;
            let residual = post_attn.clone();
            let ffn_in = measure(Stage::PostAttnNorm, &dev, || {
                self.layers[layer_idx]
                    .post_attn_norm
                    .forward(&post_attn.contiguous()?)
            })?;
            layer_in = measure(Stage::Mlp, &dev, || {
                self.layers[layer_idx]
                    .mlp
                    .forward(&ffn_in)
                    .and_then(|x| x + &residual)
            })?;
        }
        let x = measure(Stage::NormFinal, &dev, || {
            self.norm.forward(&layer_in.contiguous()?)
        })?;
        // Take the LAST position so this works for both decode (seq_len=1, last=0) and
        // batched prefill (seq_len=N, last=N-1). Was hardcoded to position 0 before, which
        // forced forward_physics to loop one token at a time during prefill (~N× slower).
        let last_idx = x.dim(1)? - 1;
        let x_last = x.i((.., last_idx, ..))?;
        let logits = measure(Stage::LmHead, &dev, || self.output.forward(&x_last))?;
        note_forward_call();
        Ok((logits, x_last))
    }

    pub fn forward_physics<P: PhysicsEngine>(
        &mut self,
        input: &Tensor,
        index_pos: usize,
        physics: &mut P,
        ghost_vector: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let seq_len = input.dim(1)?;
        if seq_len == 0 {
            return Err(candle_core::Error::Msg("empty qwen35 input".to_string()));
        }
        if seq_len == 1 {
            return self.forward_one(input, index_pos, physics, ghost_vector);
        }

        // Qwen3.5 hybrid layers include recurrent linear/SSM attention. Full-attention
        // layers can prefill as a batch, but linear layers must advance one token at a
        // time so conv_state/ssm_state and full-attention KV stay aligned.
        let mut last: Option<(Tensor, Tensor)> = None;
        for offset in 0..seq_len {
            let token = input.narrow(1, offset, 1)?;
            last = Some(self.forward_one(&token, index_pos + offset, physics, ghost_vector)?);
        }
        last.ok_or_else(|| candle_core::Error::Msg("empty qwen35 input".to_string()))
    }
}
