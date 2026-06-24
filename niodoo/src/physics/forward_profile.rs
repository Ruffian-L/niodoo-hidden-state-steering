//! Per-stage timer for `forward_physics` / `forward_one` in the LLM forward
//! path. Goal: find which stage dominates the 1423 ms/token outlier from the
//! fixed-001 timing so we can target a real fix.
//!
//! Usage: set `NIODOO_PROFILE_FORWARD=1` in the env, run any code path that
//! invokes `qwen35_hybrid::forward_physics`, then call
//! [`dump_forward_profile`] before exit (or just observe the per-call dumps if
//! `NIODOO_PROFILE_FORWARD_VERBOSE=1` is also set).
//!
//! The profiler synchronizes the CUDA device between every stage so that
//! per-stage wall time reflects actual GPU completion, not just kernel-launch
//! overhead. This serializes work that would normally pipeline — absolute
//! numbers will be higher than production, but RELATIVE breakdown is valid.
//!
//! Tracks per-bucket: total / count / max / min. The **max** is critical —
//! a 1423 ms outlier against a 10 ms typical hides easily in totals. Verbose
//! mode (`NIODOO_PROFILE_FORWARD_VERBOSE=1`) dumps a per-call snapshot so you
//! can localize position-sensitive outliers (e.g. always-on-token-0 patterns
//! that point at shape-specific QMatMul fallback).
//!
//! Zero overhead when `NIODOO_PROFILE_FORWARD` is unset (single AtomicBool
//! check — branch predictor friendly).

use candle_core::Device;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

#[derive(Copy, Clone, Debug)]
#[repr(usize)]
pub enum Stage {
    Embed = 0,
    AttnNorm,
    Wq,
    Wk,
    Wv,
    QkNorm,
    Rope,
    KvCat,
    AttQkMatmul,
    Softmax,
    AttVMatmul,
    OutProj,
    Physics,
    Blend,
    PostAttnNorm,
    Mlp,
    NormFinal,
    LmHead,
}

const N_STAGES: usize = Stage::LmHead as usize + 1;

const STAGE_NAMES: &[&str] = &[
    "embed",
    "attn_norm",
    "wq",
    "wk",
    "wv",
    "qk_norm",
    "rope",
    "kv_cat",
    "attn_qk_matmul",
    "softmax",
    "attn_v_matmul",
    "out_proj",
    "physics",
    "blend",
    "post_attn_norm",
    "mlp",
    "norm_final",
    "lm_head",
];

// Cumulative across the whole run.
static STAGE_NS_TOTAL: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];
static STAGE_HITS: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];
static STAGE_NS_MAX: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];
// Sentinel: u64::MAX means "no observation yet". fetch_min handles it correctly.
static STAGE_NS_MIN: [AtomicU64; N_STAGES] = [const { AtomicU64::new(u64::MAX) }; N_STAGES];
// Per-call snapshot, reset on note_forward_call(). Lets verbose mode dump THIS
// call's breakdown (vs cumulative).
static STAGE_NS_CURCALL: [AtomicU64; N_STAGES] = [const { AtomicU64::new(0) }; N_STAGES];
static FORWARD_CALLS: AtomicU64 = AtomicU64::new(0);

fn enabled() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("NIODOO_PROFILE_FORWARD").is_ok())
}

fn verbose() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| std::env::var("NIODOO_PROFILE_FORWARD_VERBOSE").is_ok())
}

/// Run `f` and, if profiling is enabled, sync the device and accumulate
/// elapsed wall time into the bucket for `stage`. Updates total / count /
/// max / min and the per-call snapshot.
#[inline(always)]
pub fn measure<R>(stage: Stage, dev: &Device, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let t0 = std::time::Instant::now();
    let result = f();
    if let Err(e) = dev.synchronize() {
        eprintln!("[forward_profile] sync error after {:?}: {e}", stage);
    }
    let ns = t0.elapsed().as_nanos() as u64;
    let i = stage as usize;
    STAGE_NS_TOTAL[i].fetch_add(ns, Ordering::Relaxed);
    STAGE_NS_CURCALL[i].fetch_add(ns, Ordering::Relaxed);
    STAGE_HITS[i].fetch_add(1, Ordering::Relaxed);
    STAGE_NS_MAX[i].fetch_max(ns, Ordering::Relaxed);
    STAGE_NS_MIN[i].fetch_min(ns, Ordering::Relaxed);
    result
}

/// Mark one full forward call complete. In verbose mode, dumps the per-call
/// snapshot before resetting it. Otherwise just bumps the call counter and
/// resets the snapshot.
pub fn note_forward_call() {
    if !enabled() {
        return;
    }
    let call_idx = FORWARD_CALLS.fetch_add(1, Ordering::Relaxed);
    if verbose() {
        dump_per_call_snapshot(call_idx);
    }
    for a in STAGE_NS_CURCALL.iter() {
        a.store(0, Ordering::Relaxed);
    }
}

fn dump_per_call_snapshot(call_idx: u64) {
    let mut rows: Vec<(usize, u64)> = (0..N_STAGES)
        .map(|i| (i, STAGE_NS_CURCALL[i].load(Ordering::Relaxed)))
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));
    let total_ns: u64 = rows.iter().map(|(_, ns)| ns).sum();
    if total_ns == 0 {
        return;
    }
    let top: Vec<String> = rows
        .iter()
        .filter(|(_, ns)| *ns > 0)
        .take(5)
        .map(|(i, ns)| format!("{}={:.2}ms", STAGE_NAMES[*i], *ns as f64 / 1e6))
        .collect();
    eprintln!(
        "[forward_profile call#{:>4}] total={:.2}ms  top: {}",
        call_idx,
        total_ns as f64 / 1e6,
        top.join("  ")
    );
}

/// Print the cumulative per-stage breakdown to stderr. Sorted by total_ms
/// descending. Shows total / max / min / mean / % per bucket. Use this at end
/// of run for the canonical "where did time go" view.
pub fn dump_forward_profile() {
    if !enabled() {
        eprintln!("[forward_profile] disabled (set NIODOO_PROFILE_FORWARD=1 to enable)");
        return;
    }
    let calls = FORWARD_CALLS.load(Ordering::Relaxed);
    let total_ns: u64 = STAGE_NS_TOTAL
        .iter()
        .map(|a| a.load(Ordering::Relaxed))
        .sum();
    if total_ns == 0 {
        eprintln!("[forward_profile] no stages have run yet");
        return;
    }
    eprintln!("─── forward_profile ───────────────────────────────────────────");
    eprintln!(
        "  forward calls : {}   total stage time : {:.3} ms",
        calls,
        total_ns as f64 / 1_000_000.0
    );
    eprintln!(
        "  {:<16} {:>10} {:>9} {:>9} {:>9} {:>7} {:>8}",
        "stage", "total_ms", "max_us", "min_us", "mean_us", "%total", "hits"
    );

    let mut rows: Vec<(usize, u64, u64, u64, u64)> = (0..N_STAGES)
        .map(|i| {
            (
                i,
                STAGE_NS_TOTAL[i].load(Ordering::Relaxed),
                STAGE_HITS[i].load(Ordering::Relaxed),
                STAGE_NS_MAX[i].load(Ordering::Relaxed),
                STAGE_NS_MIN[i].load(Ordering::Relaxed),
            )
        })
        .collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1));

    for (i, ns, hits, max_ns, min_ns) in rows {
        if hits == 0 {
            continue;
        }
        let total_ms = ns as f64 / 1_000_000.0;
        let mean_us = (ns as f64 / hits as f64) / 1_000.0;
        let max_us = max_ns as f64 / 1_000.0;
        let min_us = if min_ns == u64::MAX {
            0.0
        } else {
            min_ns as f64 / 1_000.0
        };
        let pct = 100.0 * ns as f64 / total_ns as f64;
        // Flag rows where max > 10× mean — that's the outlier-hidden-in-mean signature.
        let outlier_flag = if mean_us > 0.0 && max_us > 10.0 * mean_us {
            "  ← MAX>>MEAN"
        } else {
            ""
        };
        eprintln!(
            "  {:<16} {:>10.3} {:>9.1} {:>9.1} {:>9.1} {:>6.1}% {:>8}{}",
            STAGE_NAMES[i], total_ms, max_us, min_us, mean_us, pct, hits, outlier_flag
        );
    }
    eprintln!("───────────────────────────────────────────────────────────────");
    eprintln!("  Tip: rows flagged ← MAX>>MEAN have outliers hiding in averages. Set");
    eprintln!("       NIODOO_PROFILE_FORWARD_VERBOSE=1 to print a per-call snapshot");
    eprintln!("       so you can spot position-sensitive patterns (e.g. always-on-token-0).");
}

/// Reset all counters. Useful for warm-up vs measured runs.
pub fn reset_forward_profile() {
    for a in STAGE_NS_TOTAL.iter() {
        a.store(0, Ordering::Relaxed);
    }
    for a in STAGE_NS_CURCALL.iter() {
        a.store(0, Ordering::Relaxed);
    }
    for a in STAGE_HITS.iter() {
        a.store(0, Ordering::Relaxed);
    }
    for a in STAGE_NS_MAX.iter() {
        a.store(0, Ordering::Relaxed);
    }
    for a in STAGE_NS_MIN.iter() {
        a.store(u64::MAX, Ordering::Relaxed);
    }
    FORWARD_CALLS.store(0, Ordering::Relaxed);
}
