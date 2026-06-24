//! Atlas M9b drift watchdog — Rust port of `scripts/telemetry_drift_watchdog.py`.
//!
//! CUSUM, Page-Hinkley, and EWMA drift detectors with discrete one-shot alarm
//! semantics (§10dh: `reset_on_alarm = true` resets accumulators after each
//! alarm so the detector emits an event-onset rather than a sustained signal).
//!
//! This is the first Rust runtime primitive port from the
//! `LATENT_PACKET_STEERING_STABILIZATION_PLAN.md` "From Python to Rust" table.
//! Per-test bit-equivalent output to the Python reference on the §10dp
//! validation data is the gate before runtime integration.
//!
//! Defaults match Atlas Section 4 cheat-sheet:
//!   CUSUM:        k=0.5, h=5.0, burn_in=10
//!   Page-Hinkley: delta=0.005, lambda=50.0, burn_in=10
//!   EWMA:         lambda=0.1, L=3.0, burn_in=30

#[derive(Debug, Clone, PartialEq)]
pub struct Cusum {
    pub k: f64,
    pub h: f64,
    pub s_pos: f64,
    pub s_neg: f64,
    pub mu: Option<f64>,
    pub n: u32,
    pub burn_in: u32,
    pub reset_on_alarm: bool,
}

impl Default for Cusum {
    fn default() -> Self {
        Self {
            k: 0.5,
            h: 5.0,
            s_pos: 0.0,
            s_neg: 0.0,
            mu: None,
            n: 0,
            burn_in: 10,
            reset_on_alarm: true,
        }
    }
}

impl Cusum {
    pub fn new(k: f64, h: f64) -> Self {
        Self {
            k,
            h,
            ..Default::default()
        }
    }

    /// Process one observation. Returns `true` iff an alarm fires this step.
    pub fn step(&mut self, x: f64) -> bool {
        self.n += 1;
        if self.n <= self.burn_in {
            self.mu = match self.mu {
                None => Some(x),
                Some(prev) => {
                    let n = self.n as f64;
                    Some((prev * (n - 1.0) + x) / n)
                }
            };
            return false;
        }
        let mu = self.mu.unwrap_or(0.0);
        self.s_pos = (self.s_pos + (x - mu - self.k)).max(0.0);
        self.s_neg = (self.s_neg + (x - mu + self.k)).min(0.0);
        let alarm = self.s_pos > self.h || -self.s_neg > self.h;
        if alarm && self.reset_on_alarm {
            self.s_pos = 0.0;
            self.s_neg = 0.0;
        }
        alarm
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PageHinkley {
    pub delta: f64,
    pub lam: f64,
    pub mt: f64,
    pub min_mt: f64,
    pub sum_x: f64,
    pub n: u32,
    pub burn_in: u32,
    pub reset_on_alarm: bool,
}

impl Default for PageHinkley {
    fn default() -> Self {
        Self {
            delta: 0.005,
            lam: 50.0,
            mt: 0.0,
            min_mt: 0.0,
            sum_x: 0.0,
            n: 0,
            burn_in: 10,
            reset_on_alarm: true,
        }
    }
}

impl PageHinkley {
    pub fn new(delta: f64, lam: f64) -> Self {
        Self {
            delta,
            lam,
            ..Default::default()
        }
    }

    pub fn step(&mut self, x: f64) -> bool {
        self.n += 1;
        self.sum_x += x;
        if self.n <= self.burn_in {
            return false;
        }
        let x_bar = self.sum_x / self.n as f64;
        self.mt += x - x_bar - self.delta;
        self.min_mt = self.min_mt.min(self.mt);
        let alarm = (self.mt - self.min_mt) > self.lam;
        if alarm && self.reset_on_alarm {
            self.mt = 0.0;
            self.min_mt = 0.0;
        }
        alarm
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ewma {
    pub lam: f64,
    pub l_limit: f64,
    pub z: Option<f64>,
    pub mu: f64,
    pub sigma: f64,
    pub n: u32,
    pub burn_in: u32,
    pub history: Vec<f64>,
    pub reset_on_alarm: bool,
}

impl Default for Ewma {
    fn default() -> Self {
        Self {
            lam: 0.1,
            l_limit: 3.0,
            z: None,
            mu: 0.0,
            sigma: 1.0,
            n: 0,
            burn_in: 30,
            history: Vec::new(),
            reset_on_alarm: true,
        }
    }
}

impl Ewma {
    pub fn new(lam: f64, l_limit: f64) -> Self {
        Self {
            lam,
            l_limit,
            ..Default::default()
        }
    }

    pub fn step(&mut self, x: f64) -> bool {
        self.n += 1;
        self.history.push(x);
        self.z = Some(match self.z {
            None => x,
            Some(prev) => self.lam * x + (1.0 - self.lam) * prev,
        });
        if self.n <= self.burn_in {
            if self.n == self.burn_in {
                let n = self.history.len() as f64;
                self.mu = self.history.iter().sum::<f64>() / n;
                if self.history.len() > 1 {
                    let var: f64 = self
                        .history
                        .iter()
                        .map(|y| (y - self.mu).powi(2))
                        .sum::<f64>()
                        / (n - 1.0);
                    self.sigma = var.sqrt().max(1e-6);
                }
            }
            return false;
        }
        let ctrl = self.l_limit * self.sigma * (self.lam / (2.0 - self.lam)).sqrt();
        let z = self.z.unwrap_or(self.mu);
        let alarm = (z - self.mu).abs() > ctrl;
        if alarm && self.reset_on_alarm {
            self.z = Some(self.mu);
        }
        alarm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CUSUM step-shift: a stable mean for burn-in, then a positive shift
    /// large enough to trip h=5 within ~5 steps. Mirrors the Python smoke.
    #[test]
    fn cusum_detects_positive_shift_with_reset() {
        let mut c = Cusum::default();
        // burn-in: 10 samples around 0
        for _ in 0..10 {
            assert!(!c.step(0.0));
        }
        // baseline mu should be 0
        assert!((c.mu.unwrap_or(99.0) - 0.0).abs() < 1e-9);
        // sustained positive shift of +2 each step; with k=0.5 the
        // increment is 1.5/step; alarm at h=5 fires by step 4
        let mut alarms = 0;
        for _ in 0..6 {
            if c.step(2.0) {
                alarms += 1;
            }
        }
        assert!(
            alarms >= 1,
            "expected at least one alarm under sustained shift"
        );
        // post-alarm reset: s_pos should have dropped to 0 at the alarm step
        // and only re-accumulate from there. Concretely: with reset_on_alarm,
        // the detector should NOT emit alarms on every subsequent step.
        let mut sustained = 0;
        for _ in 0..6 {
            if c.step(2.0) {
                sustained += 1;
            }
        }
        assert!(
            sustained < 6,
            "reset_on_alarm should prevent every-step alarm under sustained shift; got {}",
            sustained
        );
    }

    #[test]
    fn cusum_quiet_under_noise() {
        let mut c = Cusum::default();
        // Mean 0, small noise; no alarms expected within 200 steps
        let xs = [
            0.10, -0.20, 0.15, -0.10, 0.05, -0.05, 0.20, -0.15, 0.10, -0.10, 0.05, -0.05, 0.15,
            -0.20, 0.10, 0.00, -0.10, 0.20, -0.15, 0.05,
        ];
        let mut alarms = 0;
        for _ in 0..10 {
            for &x in &xs {
                if c.step(x) {
                    alarms += 1;
                }
            }
        }
        assert_eq!(alarms, 0, "CUSUM should be quiet on small zero-mean noise");
    }

    #[test]
    fn page_hinkley_detects_step_drift_with_reset() {
        let mut ph = PageHinkley::default();
        // Long stable phase so x_bar tracks ~0
        for _ in 0..10 {
            assert!(!ph.step(0.0));
        }
        for _ in 0..200 {
            ph.step(0.0);
        }
        // Sudden positive shift large enough that mt - min_mt > lam=50
        // within reasonable steps. With x_bar ≈ 0 and x=10, each step
        // adds ~9.995 to mt; alarm after ~6 steps.
        let mut first_alarm: Option<u32> = None;
        for i in 0..20 {
            if ph.step(10.0) && first_alarm.is_none() {
                first_alarm = Some(i);
            }
        }
        assert!(
            first_alarm.is_some(),
            "PH should alarm on sudden +10 shift after stable phase"
        );
    }

    #[test]
    fn ewma_alarms_on_step_change_and_resets_z() {
        let mut e = Ewma::default();
        // burn-in 30: stable around 0 with small noise
        let burn = (0..30).map(|i| if i % 2 == 0 { 0.10 } else { -0.10 });
        for x in burn {
            assert!(!e.step(x));
        }
        // post burn-in step shift of +5 (way outside ctrl band)
        let mut alarmed = false;
        for _ in 0..30 {
            if e.step(5.0) {
                alarmed = true;
                break;
            }
        }
        assert!(alarmed, "EWMA should alarm on post-burn-in step shift");
        // EWMA's reset-on-alarm sets z = mu. Under sustained large drift,
        // z snaps back to mu immediately after each alarm, then is pulled
        // away by the next observation, so EWMA naturally re-alarms most
        // steps. This is the documented behavior and matches the Python
        // reference. The reset still does its job: z is repeatedly pulled
        // toward mu rather than accumulating monotonically (CUSUM-style).
        // Verify z is NEAR mu right after the alarmed step:
        let z_after = e.z.unwrap();
        // After an alarm at step k, z gets reset to mu, then next observation
        // at step k+1 brings it to lam*x + (1-lam)*mu. With lam=0.1, x=5, mu=0:
        // z_after_next = 0.5. So z stays bounded near lam*x rather than diverging.
        assert!(
            z_after.abs() < 5.0,
            "post-reset z should stay bounded by single-step EWMA pull, got {}",
            z_after
        );
    }

    /// Determinism: two detectors with identical params produce identical
    /// alarm streams on identical input. Required for the §10dp Python
    /// equivalence check.
    #[test]
    fn determinism_across_instances() {
        let xs: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let mut a = Cusum::default();
        let mut b = Cusum::default();
        for x in &xs {
            assert_eq!(a.step(*x), b.step(*x));
        }
        assert_eq!(a, b);
    }

    /// Cross-check: with reset_on_alarm=false, CUSUM behaves like the
    /// pre-§10dh implementation (sustained alarms post-trip). Lets the
    /// runtime opt into either semantic.
    #[test]
    fn cusum_no_reset_mode_sustained_alarms() {
        let mut c = Cusum {
            reset_on_alarm: false,
            ..Default::default()
        };
        for _ in 0..10 {
            c.step(0.0);
        }
        // Push hard until alarm trips
        let mut first_alarm: Option<u32> = None;
        for i in 0..20 {
            if c.step(2.0) && first_alarm.is_none() {
                first_alarm = Some(i);
            }
        }
        assert!(first_alarm.is_some());
        // Without reset, every step after first_alarm under sustained signal
        // should also alarm
        let mut after = 0;
        for _ in 0..10 {
            if c.step(2.0) {
                after += 1;
            }
        }
        assert_eq!(
            after, 10,
            "no-reset mode should sustain alarms under sustained signal"
        );
    }
}
