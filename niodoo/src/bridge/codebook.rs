use serde::Deserialize;
use std::path::Path;

/// 256-entry × 64D vector quantization codebook loaded from codebook_256.json.
#[derive(Debug, Clone)]
pub struct CodebookVQ {
    /// 256 centroids, each 64 floats.
    pub entries: Vec<[f32; 64]>,
}

#[derive(Deserialize)]
struct CodebookJson {
    #[serde(rename = "K")]
    k: usize,
    #[serde(rename = "D")]
    d: usize,
    centroids: Vec<Vec<f64>>,
}

impl CodebookVQ {
    /// Load a codebook from a JSON file with `centroids: [[64 floats] × 256]`.
    pub fn load_from_json(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let raw: CodebookJson = serde_json::from_str(&data)?;
        if raw.k != 256 {
            return Err(format!("expected K=256, got K={}", raw.k).into());
        }
        if raw.d != 64 {
            return Err(format!("expected D=64, got D={}", raw.d).into());
        }
        if raw.centroids.len() != 256 {
            return Err(format!("expected 256 centroids, got {}", raw.centroids.len()).into());
        }
        let mut entries: Vec<[f32; 64]> = Vec::with_capacity(256);
        for (i, row) in raw.centroids.iter().enumerate() {
            if row.len() != 64 {
                return Err(format!("centroid {} has {} dims, expected 64", i, row.len()).into());
            }
            let mut arr = [0f32; 64];
            for (j, &v) in row.iter().enumerate() {
                arr[j] = v as f32;
            }
            entries.push(arr);
        }
        Ok(Self { entries })
    }

    /// Encode a 64D probe vector to the nearest centroid index (argmin L2).
    pub fn encode(&self, probe: &[f32]) -> u8 {
        let len = probe.len().min(64);
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let dist: f32 = (0..len)
                .map(|j| {
                    let d = probe[j] - entry[j];
                    d * d
                })
                .sum();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx as u8
    }

    /// Return the L2 distance from probe to its nearest centroid (quantization error).
    pub fn encode_error(&self, probe: &[f32], code: u8) -> f32 {
        let entry = &self.entries[code as usize];
        let len = probe.len().min(64);
        (0..len)
            .map(|j| {
                let d = probe[j] - entry[j];
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Decode a centroid index back to its 64D vector.
    pub fn decode(&self, code: u8) -> &[f32; 64] {
        &self.entries[code as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_codebook() -> CodebookVQ {
        let mut entries = Vec::with_capacity(256);
        for i in 0u8..=255 {
            let mut arr = [0f32; 64];
            arr[0] = i as f32;
            entries.push(arr);
        }
        CodebookVQ { entries }
    }

    #[test]
    fn test_encode_nearest() {
        let cb = dummy_codebook();
        let mut probe = [0f32; 64];
        probe[0] = 5.1;
        let code = cb.encode(&probe);
        assert_eq!(code, 5);
    }

    #[test]
    fn test_encode_error_zero_on_exact() {
        let cb = dummy_codebook();
        let probe: Vec<f32> = cb.decode(42).to_vec();
        let code = cb.encode(&probe);
        assert_eq!(code, 42);
        assert!(cb.encode_error(&probe, code) < 1e-5);
    }

    #[test]
    fn test_decode_roundtrip() {
        let cb = dummy_codebook();
        let decoded = cb.decode(100);
        assert!((decoded[0] - 100.0).abs() < 1e-5);
    }
}
