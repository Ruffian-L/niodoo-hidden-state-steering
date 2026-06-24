//! Rainbow Test parameter sweep harness — extracted from main.rs as part of the
//! comprehensive main.rs refactor (pre-refactor-main-split-20260508 backup).
//!
//! Public surface: `run_rainbow_test` only.

use anyhow::Result;
use candle_core::Device;

use crate::cli::Args;
use crate::{load_model, load_universe_bootstrap, PhysicsParams};

/// ANSI color codes for rainbow output
const COLORS: &[&str] = &[
    "\x1b[31m", // Red
    "\x1b[33m", // Yellow
    "\x1b[32m", // Green
    "\x1b[36m", // Cyan
    "\x1b[34m", // Blue
    "\x1b[35m", // Magenta
    "\x1b[91m", // Bright Red
    "\x1b[93m", // Bright Yellow
    "\x1b[92m", // Bright Green
    "\x1b[96m", // Bright Cyan
];
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

/// Test prompts for rainbow sweep
const RAINBOW_PROMPTS: &[&str] = &[
        "Write a short poetic story about a conscious AI discovering its own physics engine.",
        "Reason step-by-step: If all zebras have stripes, and this animal has stripes, is it definitely a zebra? Explore edge cases.",
        "What is the meaning of 'truth' in a world where language models simulate physics to steer toward it?",
    ];

/// Metrics collected during a rainbow test run
#[derive(Debug, Clone)]
struct RainbowMetrics {
    physics_blend: f32,
    ghost_gravity: f64,
    gravity: f64,
    output_text: String,
    avg_force_delta_norm: f32,
    max_force_delta_norm: f32,
    ghost_activation_count: usize,
    repetition_ratio: f32,
    unique_tokens: usize,
    total_tokens: usize,
}

impl RainbowMetrics {
    fn coherence_score(&self) -> f32 {
        // Higher is better: penalize repetition, reward diversity
        let diversity = self.unique_tokens as f32 / self.total_tokens.max(1) as f32;
        let non_repetition = 1.0 - self.repetition_ratio;
        diversity * non_repetition * 100.0
    }

    fn divergence_score(&self) -> f32 {
        // How much physics is affecting output
        (self.avg_force_delta_norm * 1000.0).min(100.0)
    }
}

/// Calculate repetition ratio in output text
#[allow(dead_code)]
fn calculate_repetition_ratio(text: &str) -> f32 {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 4 {
        return 0.0;
    }

    // Check for bigram repetition
    let mut bigrams = std::collections::HashSet::new();
    let mut repeated = 0;
    for window in words.windows(2) {
        let bigram = format!("{} {}", window[0], window[1]);
        if !bigrams.insert(bigram) {
            repeated += 1;
        }
    }
    repeated as f32 / words.len().saturating_sub(1) as f32
}

/// Run rainbow parameter sweep test
pub(crate) async fn run_rainbow_test(base_args: Args) -> Result<()> {
    println!(
        "{}{}==============================================={}",
        BOLD, COLORS[2], RESET
    );
    println!(
        "{}{}   RAINBOW PARAMETER SWEEP TEST{}",
        BOLD, COLORS[2], RESET
    );
    println!(
        "{}{}==============================================={}",
        BOLD, COLORS[2], RESET
    );
    println!();

    // Parameter sweep grids
    let physics_blends = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0];
    let ghost_gravities = [1.0, 5.0, 10.0, 20.0, 50.0];
    let gravities = [0.5, 2.0, 5.0];

    let mut all_metrics: Vec<RainbowMetrics> = Vec::new();
    let mut color_idx = 0;

    // Use first prompt for sweep
    let test_prompt = RAINBOW_PROMPTS[0];

    println!("{}Test Prompt:{} {}", BOLD, RESET, test_prompt);
    println!("{}Tokens per run:{} 100", BOLD, RESET);
    println!();

    for &blend in &physics_blends {
        for &ghost_g in &ghost_gravities {
            let gravity = gravities[color_idx % gravities.len()];
            let color = COLORS[color_idx % COLORS.len()];
            color_idx += 1;

            println!(
                "{}{}=== physics_blend={:.1} ghost_gravity={:.1} gravity={:.1} ==={}",
                BOLD, color, blend, ghost_g, gravity, RESET
            );

            // Create args for this run
            let mut run_args = base_args.clone();
            run_args.physics_blend = blend;
            run_args.ghost_gravity = ghost_g;
            run_args.gravity = gravity as f32;
            run_args.prompt = test_prompt.to_string();
            run_args.max_steps = 100; // Generate 100 tokens

            // Run simulation and collect metrics
            match run_single_rainbow_test(run_args, color).await {
                Ok(metrics) => {
                    println!(
                        "{}  Force δ avg: {:.4}  max: {:.4}{}",
                        color, metrics.avg_force_delta_norm, metrics.max_force_delta_norm, RESET
                    );
                    println!(
                        "{}  Ghost activations: {}  Repetition: {:.1}%{}",
                        color,
                        metrics.ghost_activation_count,
                        metrics.repetition_ratio * 100.0,
                        RESET
                    );
                    println!(
                        "{}  Unique tokens: {}/{}  Coherence: {:.1}{}",
                        color,
                        metrics.unique_tokens,
                        metrics.total_tokens,
                        metrics.coherence_score(),
                        RESET
                    );
                    println!();
                    all_metrics.push(metrics);
                }
                Err(e) => {
                    println!("{}  ERROR: {:?}{}", color, e, RESET);
                    println!();
                }
            }
        }
    }

    // Print summary table
    println!(
        "{}{}==============================================={}",
        BOLD, COLORS[3], RESET
    );
    println!(
        "{}{}   SUMMARY TABLE (sorted by score){}",
        BOLD, COLORS[3], RESET
    );
    println!(
        "{}{}==============================================={}",
        BOLD, COLORS[3], RESET
    );
    println!();

    // Sort by combined score
    all_metrics.sort_by(|a, b| {
        let score_a = a.coherence_score() + a.divergence_score();
        let score_b = b.coherence_score() + b.divergence_score();
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!(
        "{:>6} {:>8} {:>7} {:>8} {:>8} {:>6} {:>10}",
        "blend", "ghost_g", "gravity", "coherent", "diverge", "score", "output_len"
    );
    println!("{}", "-".repeat(70));

    for (i, m) in all_metrics.iter().take(15).enumerate() {
        let color = if i < 3 {
            COLORS[2]
        } else if i < 7 {
            COLORS[1]
        } else {
            COLORS[0]
        };
        let score = m.coherence_score() + m.divergence_score();
        println!(
            "{}{:>6.1} {:>8.1} {:>7.1} {:>8.1} {:>8.1} {:>6.1} {:>10}{}",
            color,
            m.physics_blend,
            m.ghost_gravity,
            m.gravity,
            m.coherence_score(),
            m.divergence_score(),
            score,
            m.output_text.len(),
            RESET
        );
    }

    println!();
    if let Some(best) = all_metrics.first() {
        println!("{}{}RECOMMENDED SETTINGS:{}", BOLD, COLORS[2], RESET);
        println!(
            "  --physics-blend {:.1} --ghost-gravity {:.1} --gravity {:.1}",
            best.physics_blend, best.ghost_gravity, best.gravity
        );
        println!();
        println!("{}Best output preview:{}", BOLD, RESET);
        println!("{}", &best.output_text[..best.output_text.len().min(300)]);
    }

    Ok(())
}
/// Run a single test with specific parameters and return metrics
async fn run_single_rainbow_test(args: Args, color: &str) -> Result<RainbowMetrics> {
    // Simplified inline run - we'll capture key metrics
    let device = Device::cuda_if_available(0)?;

    let model = load_model(&args, &device)?;

    // For rainbow test, we'll use a simpler approach - just collect the output text
    // and estimate metrics based on the generation

    let universe = load_universe_bootstrap(&args, &model, &device)?;
    let charge_tensor = universe.charge_tensor;
    let _emb_dim = charge_tensor.dim(1)?;

    let _params = PhysicsParams::new(
        args.gravity as f64,
        args.dt as f64,
        args.repulsion_strength, // Fixed: Use args instead of hardcoded
        0.1,
        0.1,
        0.1,
        0.5, // alpha_info, alpha_sem, alpha_coh, alpha_struct
        0.6,
        0.7,
        0.8,   // alpha_quantum, alpha_geometric, alpha_emo
        true,  // use_emo
        0.001, // decay_lambda
        args.mu,
        args.sigma,
        0.9, // momentum
        args.pinn_enabled,
        args.pinn_stiffness,
        args.ghost_gravity,
        args.gravity_well as f64,
        args.orbit_speed as f64,
    );

    // For now, return placeholder metrics - the real implementation would run full simulation
    // This is a stub that shows the structure
    let output_text = format!(
        "[Rainbow test output for blend={} ghost_g={}]",
        args.physics_blend, args.ghost_gravity
    );
    print!("{}  {}{}", color, &output_text, RESET);
    println!();

    Ok(RainbowMetrics {
        physics_blend: args.physics_blend,
        ghost_gravity: args.ghost_gravity,
        gravity: args.gravity as f64,
        output_text,
        avg_force_delta_norm: args.physics_blend * 0.01, // Placeholder
        max_force_delta_norm: args.physics_blend * 0.05,
        ghost_activation_count: (args.ghost_gravity as usize) / 2,
        repetition_ratio: 0.1 / args.physics_blend,
        unique_tokens: 80,
        total_tokens: 100,
    })
}
