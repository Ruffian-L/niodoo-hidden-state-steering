//! Residual TCT-splat-lite loader + Gaussian force (hydro → niodoo-live apply path).
//!
//! Wire format matches `hydrodynamic-swarm/src/tct.rs` **tct-splat-lite** (TCT1, v3):
//! residual centers + σ + signed α (gain) + λ + trigger_kind + prompt_fp.
//!
//! This is **not** the 64D correction-packet path. Packets compress through bucket-mean
//! VQ; residual TCT scars act directly in hidden residual space (typically 4096 for
//! Llama-3.1-8B, 2560 for Gemma-class hydro runs). Dim must match the live model
//! (`[INV-5]` model_fp / dim guard).
//!
//! Claims discipline: no feelings. Memory as measurable geometry only.

use std::io::Read;
use std::path::Path;

/// Wire magic `TCT1`.
pub const TCT_MAGIC: [u8; 4] = *b"TCT1";
/// Schema version: 3 = per-record prompt_fp after trigger_kind.
pub const TCT_VERSION: u16 = 3;

pub const FLAG_HAS_LOCALITY: u16 = 1 << 0;
pub const FLAG_RESIDUAL_SPACE: u16 = 1 << 1;

pub const TRIGGER_SURPRISE_DELTA: u32 = 3;
pub const TRIGGER_MANUAL: u32 = 4;
/// Prefill-bridge scar (continuity at next-run start basin).
pub const TRIGGER_PREFILL_BRIDGE: u32 = 5;

/// One localized residual memory (one splat).
#[derive(Debug, Clone)]
pub struct TctLocalityRecord {
    pub center: Vec<f32>,
    pub sigma: f32,
    /// Signed gain: + pleasure, − pain (splat alpha).
    pub gain: f32,
    pub decay_constant: f32,
    pub created_at_ms: u64,
    pub scale: u8,
    pub is_anchor: bool,
    pub trigger_kind: u32,
    /// FNV of prompt text for prefill-bridges (0 = unknown / trail scar).
    pub prompt_fp: u32,
}

impl TctLocalityRecord {
    pub fn is_prefill_bridge(&self) -> bool {
        self.trigger_kind == TRIGGER_PREFILL_BRIDGE
    }
}

/// Portable store: header + locality records.
#[derive(Debug, Clone)]
pub struct TctSplatStore {
    pub version: u16,
    pub flags: u16,
    pub model_dim: u32,
    /// Fingerprint of base model; 0 = unknown / not enforced.
    pub model_fp: u32,
    pub records: Vec<TctLocalityRecord>,
}

/// Per-query force stats for telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct TctForceStats {
    pub n_active: usize,
    pub force_norm: f32,
    pub potential: f32,
    pub nearest_scar_dist: f32,
    pub n_considered: usize,
}

impl TctSplatStore {
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn n_prefill_bridges(&self) -> usize {
        self.records.iter().filter(|r| r.is_prefill_bridge()).count()
    }

    pub fn read_binary(path: &Path) -> anyhow::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if magic != TCT_MAGIC {
            anyhow::bail!(
                "bad TCT magic: {:?} (expected TCT1) — not a tct-splat-lite file",
                magic
            );
        }
        let version = read_u16(&mut f)?;
        let flags = read_u16(&mut f)?;
        let model_dim = read_u32(&mut f)?;
        let model_fp = read_u32(&mut f)?;
        let n = read_u32(&mut f)? as usize;
        let mut skip = [0u8; 16];
        f.read_exact(&mut skip)?;

        let mut records = Vec::with_capacity(n);
        for _ in 0..n {
            let dim = read_u32(&mut f)? as usize;
            let sigma = read_f32(&mut f)?;
            let gain = read_f32(&mut f)?;
            let decay_constant = read_f32(&mut f)?;
            let created_at_ms = read_u64(&mut f)?;
            let mut meta = [0u8; 4];
            f.read_exact(&mut meta)?;
            let scale = meta[0];
            let is_anchor = meta[1] != 0;
            let trigger_kind = read_u32(&mut f)?;
            let prompt_fp = if version >= 3 {
                read_u32(&mut f)?
            } else {
                0
            };
            let mut center = vec![0f32; dim];
            for c in &mut center {
                *c = read_f32(&mut f)?;
            }
            records.push(TctLocalityRecord {
                center,
                sigma,
                gain,
                decay_constant,
                created_at_ms,
                scale,
                is_anchor,
                trigger_kind,
                prompt_fp,
            });
        }

        Ok(Self {
            version,
            flags,
            model_dim,
            model_fp,
            records,
        })
    }

    /// Hard dim guard: every record center length and header model_dim must match
    /// the live hidden size. Mismatched hydro (e.g. Gemma 2560) vs Llama 4096 is
    /// rejected rather than silently truncated.
    pub fn enforce_dim(&self, hidden_dim: usize) -> anyhow::Result<()> {
        if self.model_dim != 0 && self.model_dim as usize != hidden_dim {
            anyhow::bail!(
                "TCT model_dim={} != live hidden_dim={} — residual scars are model-native; re-export from a matching-dim run [INV-5]",
                self.model_dim,
                hidden_dim
            );
        }
        for (i, r) in self.records.iter().enumerate() {
            if r.center.len() != hidden_dim {
                anyhow::bail!(
                    "TCT record[{}] center dim={} != live hidden_dim={} [INV-5]",
                    i,
                    r.center.len(),
                    hidden_dim
                );
            }
        }
        Ok(())
    }

    /// Gaussian residual force matching hydro `SplatMemory::query_force`:
    /// F = Σ_i α_i · exp(−‖μ_i − p‖² / σ_i²) · (μ_i − p), then 1/√n_active.
    ///
    /// When `bridge_only`, only `trigger_kind == PREFILL_BRIDGE` scars contribute.
    pub fn query_force(&self, pos: &[f32], bridge_only: bool) -> (Vec<f32>, TctForceStats) {
        let d = pos.len();
        let mut force = vec![0f32; d];
        let mut n_active = 0usize;
        let mut potential = 0f32;
        let mut nearest = f32::INFINITY;
        let mut n_considered = 0usize;

        for r in &self.records {
            if bridge_only && !r.is_prefill_bridge() {
                continue;
            }
            if r.center.len() != d {
                continue;
            }
            n_considered += 1;
            let mut dist_sq = 0f32;
            for i in 0..d {
                let diff = r.center[i] - pos[i];
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();
            if dist < nearest {
                nearest = dist;
            }
            let sigma = r.sigma.max(1e-6);
            let sigma_sq = sigma * sigma;
            let kernel = (-dist_sq / sigma_sq).exp();
            let scale = r.gain * kernel;
            potential += scale;
            if scale.abs() < 1e-7 {
                continue;
            }
            for i in 0..d {
                force[i] += scale * (r.center[i] - pos[i]);
            }
            n_active += 1;
        }

        if n_active > 1 {
            let norm = 1.0 / (n_active as f32).sqrt();
            for v in &mut force {
                *v *= norm;
            }
        }

        let mut force_norm_sq = 0f32;
        for &v in &force {
            force_norm_sq += v * v;
        }
        let force_norm = force_norm_sq.sqrt();
        if !nearest.is_finite() {
            nearest = f32::INFINITY;
        }

        (
            force,
            TctForceStats {
                n_active,
                force_norm,
                potential,
                nearest_scar_dist: nearest,
                n_considered,
            },
        )
    }

    /// L2-clamp a force vector in place. `clamp <= 0` leaves force unchanged.
    pub fn clamp_force(force: &mut [f32], clamp: f32) -> f32 {
        let mut nsq = 0f32;
        for &v in force.iter() {
            nsq += v * v;
        }
        let n = nsq.sqrt();
        if clamp > 0.0 && n > clamp && n.is_finite() && n > 1e-12 {
            let s = clamp / n;
            for v in force.iter_mut() {
                *v *= s;
            }
            clamp
        } else {
            n
        }
    }

    /// Write a minimal TCT1 binary (tests + synthetic fixtures).
    pub fn write_binary(&self, path: &Path) -> anyhow::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(&TCT_MAGIC)?;
        f.write_all(&self.version.to_le_bytes())?;
        f.write_all(&self.flags.to_le_bytes())?;
        f.write_all(&self.model_dim.to_le_bytes())?;
        f.write_all(&self.model_fp.to_le_bytes())?;
        f.write_all(&(self.records.len() as u32).to_le_bytes())?;
        f.write_all(&[0u8; 16])?;
        for r in &self.records {
            f.write_all(&(r.center.len() as u32).to_le_bytes())?;
            f.write_all(&r.sigma.to_le_bytes())?;
            f.write_all(&r.gain.to_le_bytes())?;
            f.write_all(&r.decay_constant.to_le_bytes())?;
            f.write_all(&r.created_at_ms.to_le_bytes())?;
            f.write_all(&[r.scale, r.is_anchor as u8, 0, 0])?;
            f.write_all(&r.trigger_kind.to_le_bytes())?;
            f.write_all(&r.prompt_fp.to_le_bytes())?;
            for &x in &r.center {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

fn read_u16(r: &mut impl Read) -> anyhow::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> anyhow::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> anyhow::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("niodoo_tct_{}_{}", std::process::id(), name));
        p
    }

    fn bridge_store(dim: usize) -> TctSplatStore {
        let mut center = vec![0f32; dim];
        center[0] = 1.0;
        center[1] = 2.0;
        TctSplatStore {
            version: TCT_VERSION,
            flags: FLAG_HAS_LOCALITY | FLAG_RESIDUAL_SPACE,
            model_dim: dim as u32,
            model_fp: 0,
            records: vec![TctLocalityRecord {
                center,
                sigma: 10.0,
                gain: 0.75,
                decay_constant: 0.005,
                created_at_ms: 0,
                scale: 2,
                is_anchor: false,
                trigger_kind: TRIGGER_PREFILL_BRIDGE,
                prompt_fp: 0xabcdu32,
            }],
        }
    }

    #[test]
    fn roundtrip_binary_v3() {
        let path = tmp_path("roundtrip.tct");
        let store = bridge_store(8);
        store.write_binary(&path).unwrap();
        let loaded = TctSplatStore::read_binary(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded.version, TCT_VERSION);
        assert_eq!(loaded.model_dim, 8);
        assert_eq!(loaded.records.len(), 1);
        assert_eq!(loaded.records[0].prompt_fp, 0xabcd);
        assert!(loaded.records[0].is_prefill_bridge());
        assert!((loaded.records[0].center[0] - 1.0).abs() < 1e-6);
        assert!((loaded.records[0].gain - 0.75).abs() < 1e-6);
    }

    #[test]
    fn force_pulls_toward_center() {
        let store = bridge_store(4);
        // Sit away from center on dim 0 → force should push toward +1
        let pos = vec![0.0, 2.0, 0.0, 0.0];
        let (force, stats) = store.query_force(&pos, false);
        assert!(stats.n_active >= 1);
        assert!(force[0] > 0.0, "should pull toward center[0]=1");
        assert!(stats.potential > 0.0);
        assert!(stats.nearest_scar_dist.is_finite());
    }

    #[test]
    fn bridge_only_skips_trail() {
        let mut store = bridge_store(4);
        store.records.push(TctLocalityRecord {
            center: vec![10.0, 0.0, 0.0, 0.0],
            sigma: 5.0,
            gain: 2.0,
            decay_constant: 0.01,
            created_at_ms: 0,
            scale: 0,
            is_anchor: false,
            trigger_kind: TRIGGER_SURPRISE_DELTA,
            prompt_fp: 0,
        });
        let pos = vec![10.0, 0.0, 0.0, 0.0];
        let (_f, stats_all) = store.query_force(&pos, false);
        let (_f, stats_bridge) = store.query_force(&pos, true);
        assert!(stats_all.n_considered >= 2);
        assert_eq!(stats_bridge.n_considered, 1);
    }

    #[test]
    fn enforce_dim_rejects_mismatch() {
        let store = bridge_store(8);
        assert!(store.enforce_dim(8).is_ok());
        assert!(store.enforce_dim(4096).is_err());
    }

    #[test]
    fn clamp_force_caps_norm() {
        let mut f = vec![3.0, 4.0]; // norm 5
        let n = TctSplatStore::clamp_force(&mut f, 1.0);
        assert!((n - 1.0).abs() < 1e-5);
        let got = (f[0] * f[0] + f[1] * f[1]).sqrt();
        assert!((got - 1.0).abs() < 1e-5);
    }

    #[test]
    fn load_hydro_tct_if_present() {
        // Optional: read real hydro export when sibling checkout exists.
        let hydro = Path::new("/home/ruffianl/projects/hydrodynamic-swarm/data/splat_memory.tct");
        if !hydro.exists() {
            return;
        }
        let store = TctSplatStore::read_binary(hydro).expect("hydro TCT");
        assert_eq!(&TCT_MAGIC, b"TCT1");
        assert!(store.version >= 1);
        assert!(store.len() > 0);
        // Hydro Gemma path is 2560 — document via assert when present.
        if store.model_dim == 2560 {
            assert!(store.enforce_dim(4096).is_err());
            assert!(store.enforce_dim(2560).is_ok());
        }
    }
}
