use anyhow::Context;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct RunTimingSnapshot {
    pub model_load_ms: u64,
    pub prompt_build_ms: u64,
    pub prefill_ms: u64,
    pub decode_total_ms: u64,
    pub decode_tokens: usize,
    pub ms_per_token: f64,
    pub telemetry_write_ms: u64,
    pub candidate_dump_ms: u64,
    pub total_ms: u64,
    pub gpu_name: String,
    pub max_steps: usize,
    pub actual_steps: usize,
    pub stop_reason: String,
}

#[derive(Debug, Clone)]
pub struct RunTiming {
    started_at: Instant,
    model_load_ms: u64,
    prompt_build_ms: u64,
    prefill_ms: u64,
    decode_total_ms: u64,
    decode_tokens: usize,
    telemetry_write_ms: u64,
    candidate_dump_ms: u64,
    gpu_name: String,
    max_steps: usize,
    stop_reason: String,
}

impl RunTiming {
    pub fn start(gpu_name: impl Into<String>, max_steps: usize) -> Self {
        Self {
            started_at: Instant::now(),
            model_load_ms: 0,
            prompt_build_ms: 0,
            prefill_ms: 0,
            decode_total_ms: 0,
            decode_tokens: 0,
            telemetry_write_ms: 0,
            candidate_dump_ms: 0,
            gpu_name: gpu_name.into(),
            max_steps,
            stop_reason: "not_recorded".to_string(),
        }
    }

    pub fn add_model_load_ms(&mut self, ms: u64) {
        self.model_load_ms = self.model_load_ms.saturating_add(ms);
    }

    pub fn add_prompt_build_ms(&mut self, ms: u64) {
        self.prompt_build_ms = self.prompt_build_ms.saturating_add(ms);
    }

    pub fn add_prefill_ms(&mut self, ms: u64) {
        self.prefill_ms = self.prefill_ms.saturating_add(ms);
    }

    pub fn add_decode_total_ms(&mut self, ms: u64) {
        self.decode_total_ms = self.decode_total_ms.saturating_add(ms);
    }

    pub fn add_decode_tokens(&mut self, tokens: usize) {
        self.decode_tokens = self.decode_tokens.saturating_add(tokens);
    }

    pub fn add_telemetry_write_ms(&mut self, ms: u64) {
        self.telemetry_write_ms = self.telemetry_write_ms.saturating_add(ms);
    }

    pub fn add_candidate_dump_ms(&mut self, ms: u64) {
        self.candidate_dump_ms = self.candidate_dump_ms.saturating_add(ms);
    }

    pub fn set_gpu_name(&mut self, gpu_name: impl Into<String>) {
        self.gpu_name = gpu_name.into();
    }

    pub fn set_stop_reason(&mut self, reason: impl Into<String>) {
        self.stop_reason = reason.into();
    }

    pub fn snapshot(&self) -> RunTimingSnapshot {
        let ms_per_token = if self.decode_tokens == 0 {
            0.0
        } else {
            self.decode_total_ms as f64 / self.decode_tokens as f64
        };
        RunTimingSnapshot {
            model_load_ms: self.model_load_ms,
            prompt_build_ms: self.prompt_build_ms,
            prefill_ms: self.prefill_ms,
            decode_total_ms: self.decode_total_ms,
            decode_tokens: self.decode_tokens,
            ms_per_token,
            telemetry_write_ms: self.telemetry_write_ms,
            candidate_dump_ms: self.candidate_dump_ms,
            total_ms: elapsed_ms(self.started_at),
            gpu_name: self.gpu_name.clone(),
            max_steps: self.max_steps,
            actual_steps: self.decode_tokens,
            stop_reason: self.stop_reason.clone(),
        }
    }
}

pub fn now() -> Instant {
    Instant::now()
}

pub fn elapsed_ms(started_at: Instant) -> u64 {
    let millis = started_at.elapsed().as_millis();
    millis.min(u64::MAX as u128) as u64
}

pub fn run_timing_path(telemetry_out: Option<&Path>, output_dir: &Path, req_id: &str) -> PathBuf {
    if let Some(path) = telemetry_out {
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .unwrap_or("run");
        let file_name = format!("{stem}.run_timing.json");
        return path.with_file_name(file_name);
    }

    output_dir.join(format!("{}_run_timing.json", safe_artifact_name(req_id)))
}

pub fn write_run_timing(path: &Path, snapshot: &RunTimingSnapshot) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create timing artifact parent {}",
                parent.display()
            )
        })?;
    }
    std::fs::write(path, serde_json::to_string_pretty(snapshot)?)
        .with_context(|| format!("Failed to write timing artifact {}", path.display()))
}

fn safe_artifact_name(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_timing_path_from_telemetry_path() {
        let path = run_timing_path(
            Some(Path::new("artifacts/run/telemetry.jsonl")),
            Path::new("artifacts"),
            "req/1",
        );
        assert_eq!(
            path,
            PathBuf::from("artifacts/run/telemetry.run_timing.json")
        );
    }

    #[test]
    fn sanitizes_default_timing_path() {
        let path = run_timing_path(None, Path::new("artifacts"), "req/1:two");
        assert_eq!(path, PathBuf::from("artifacts/req_1_two_run_timing.json"));
    }

    #[test]
    fn computes_ms_per_token() {
        let mut timing = RunTiming::start("Cuda(0)", 128);
        timing.add_decode_total_ms(250);
        timing.add_decode_tokens(10);
        let snapshot = timing.snapshot();
        assert_eq!(snapshot.ms_per_token, 25.0);
        assert_eq!(snapshot.actual_steps, 10);
    }
}
