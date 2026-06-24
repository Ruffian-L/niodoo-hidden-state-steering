use clap::ValueEnum;
use regex::Regex;
use std::sync::OnceLock;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum LockStopPolicy {
    Off,
    Taper,
    Immediate,
}

impl LockStopPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Taper => "taper",
            Self::Immediate => "immediate",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FinalizationDecision {
    pub should_stop: bool,
    pub reason: Option<String>,
}

impl FinalizationDecision {
    fn continue_generation() -> Self {
        Self {
            should_stop: false,
            reason: None,
        }
    }

    fn stop(reason: impl Into<String>) -> Self {
        Self {
            should_stop: true,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FinalizationSnapshot {
    pub lock_detected: bool,
    pub lock_detected_step: Option<usize>,
    pub lock_text: Option<String>,
    pub lock_stop_policy: String,
    pub lock_taper_remaining: Option<usize>,
    pub lock_stop_triggered: bool,
    pub lock_stop_reason: Option<String>,
    pub tokens_after_lock: Option<usize>,
}

pub struct FinalizationController {
    policy: LockStopPolicy,
    taper_tokens: usize,
    lock_stop_on_final_answer: bool,
    lock_detected_step: Option<usize>,
    lock_detected_text_len: Option<usize>,
    lock_text: Option<String>,
    taper_remaining: Option<usize>,
    stop_triggered: bool,
    stop_reason: Option<String>,
    tokens_after_lock: Option<usize>,
}

impl FinalizationController {
    pub fn new(
        policy: LockStopPolicy,
        taper_tokens: usize,
        lock_stop_on_final_answer: bool,
    ) -> Self {
        Self {
            policy,
            taper_tokens,
            lock_stop_on_final_answer,
            lock_detected_step: None,
            lock_detected_text_len: None,
            lock_text: None,
            taper_remaining: None,
            stop_triggered: false,
            stop_reason: None,
            tokens_after_lock: None,
        }
    }

    pub fn observe_token(
        &mut self,
        step: usize,
        recent_text: &str,
        token_text: &str,
    ) -> FinalizationDecision {
        if self.stop_triggered {
            return FinalizationDecision::stop(
                self.stop_reason
                    .clone()
                    .unwrap_or_else(|| "lock_stop_already_triggered".to_string()),
            );
        }

        if self.lock_detected_step.is_none() {
            if let Some(lock_text) = find_lock_like_surface(recent_text) {
                self.lock_detected_step = Some(step);
                self.lock_detected_text_len = Some(recent_text.len());
                self.lock_text = Some(lock_text);
                self.tokens_after_lock = Some(0);

                match self.policy {
                    LockStopPolicy::Off => return FinalizationDecision::continue_generation(),
                    LockStopPolicy::Immediate => {
                        return self.trigger_stop("lock_immediate");
                    }
                    LockStopPolicy::Taper => {
                        self.taper_remaining = Some(self.taper_tokens);
                        if self.taper_tokens == 0 {
                            return self.trigger_stop("lock_taper_exhausted");
                        }
                        return FinalizationDecision::continue_generation();
                    }
                }
            }
        } else {
            self.tokens_after_lock = self.tokens_after_lock.map(|tokens| tokens + 1);
        }

        if self.policy != LockStopPolicy::Taper || self.lock_detected_step.is_none() {
            return FinalizationDecision::continue_generation();
        }

        let post_lock_text = self
            .lock_detected_text_len
            .and_then(|idx| recent_text.get(idx..))
            .unwrap_or(recent_text);
        if taper_boundary_seen(post_lock_text, token_text, self.lock_stop_on_final_answer) {
            return self.trigger_stop("lock_taper_boundary");
        }

        let remaining = self.taper_remaining.unwrap_or(self.taper_tokens);
        if remaining == 0 {
            return self.trigger_stop("lock_taper_exhausted");
        }
        self.taper_remaining = Some(remaining.saturating_sub(1));
        if remaining <= 1 {
            return self.trigger_stop("lock_taper_exhausted");
        }

        FinalizationDecision::continue_generation()
    }

    pub fn snapshot(&self) -> FinalizationSnapshot {
        FinalizationSnapshot {
            lock_detected: self.lock_detected_step.is_some(),
            lock_detected_step: self.lock_detected_step,
            lock_text: self.lock_text.clone(),
            lock_stop_policy: self.policy.as_str().to_string(),
            lock_taper_remaining: self.taper_remaining,
            lock_stop_triggered: self.stop_triggered,
            lock_stop_reason: self.stop_reason.clone(),
            tokens_after_lock: self.tokens_after_lock,
        }
    }

    pub fn veto_current_stop(&mut self) {
        self.stop_triggered = false;
        self.stop_reason = None;
        if self.policy == LockStopPolicy::Taper {
            self.taper_remaining = Some(self.taper_tokens.max(1));
        }
    }

    fn trigger_stop(&mut self, reason: impl Into<String>) -> FinalizationDecision {
        let reason = reason.into();
        self.stop_triggered = true;
        self.stop_reason = Some(reason.clone());
        FinalizationDecision::stop(reason)
    }
}

fn find_lock_like_surface(text: &str) -> Option<String> {
    let window = tail_chars(text, 512);
    for line in window.lines().rev() {
        if is_lock_like_line(line) {
            return Some(compact_line(line, 160));
        }
    }
    if is_lock_like_line(&window) {
        return Some(compact_line(&window, 160));
    }
    None
}

fn is_lock_like_line(line: &str) -> bool {
    let compact = line.trim();
    if compact.is_empty() {
        return false;
    }
    let upper = compact.to_ascii_uppercase();
    upper.contains("[REQUEST: LOCK]")
        || upper.contains("[REQUEST:LOCK]")
        || upper.contains("[LOCK]")
        || upper.contains("AGENCY HANDS: LOCK")
        || upper.contains("ANSWER IS NOW LOCKED")
        || upper.contains("ANSWER IS LOCKED")
        || upper.starts_with("LOCK ")
        || (upper.contains(" LOCK ") && upper.contains('='))
}

fn taper_boundary_seen(recent_text: &str, token_text: &str, stop_on_final_answer: bool) -> bool {
    if token_text.contains('\n') {
        return true;
    }

    let upper = tail_chars(recent_text, 256).to_ascii_uppercase();
    // Stop when the post-LOCK window contains another LOCK-like surface. The
    // model sometimes re-emits [REQUEST: LOCK] or echoes a LOCK=payload line
    // after the initial lock detection. Catching it here prevents the taper
    // from running past the clean finalization signal (ANSWER_PRESENT_LOCK_MISS
    // repair: gate34_restore_lock* evidence 2026-05-07).
    if upper.contains("[REQUEST: LOCK]") || upper.contains("[REQUEST:LOCK]") {
        return true;
    }

    upper.contains("<|START_HEADER_ID|>")
        || upper.contains("USER TURN:")
        || upper.contains("\nUSER:")
        || upper.contains("\nASSISTANT:")
        || upper.contains("[INTERNAL MONITOR")
        || upper.contains("[REQUEST: EXPLORE]")
        || upper.contains("[REQUEST: SPIKE]")
        || upper.contains("[REQUEST: RESET]")
        || (stop_on_final_answer
            && (upper.contains("FINAL ANSWER:") || upper.contains("WORKING ANSWER:")))
}

fn compact_line(line: &str, max_chars: usize) -> String {
    line.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max_chars)
        .collect::<String>()
        .trim()
        .to_string()
}

fn tail_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars().rev().take(max_chars).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AnswerBoundaryKind {
    Literal,
    WordReverse,
    Arithmetic,
}

impl AnswerBoundaryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::WordReverse => "word_reverse",
            Self::Arithmetic => "arithmetic",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnswerBoundaryExpectation {
    pub kind: AnswerBoundaryKind,
    pub source_term: String,
    pub expected_answer: String,
    pub operation: Option<String>,
    pub lhs: Option<i64>,
    pub rhs: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct AnswerBoundarySnapshot {
    pub enabled: bool,
    pub kind: Option<String>,
    pub source_term: Option<String>,
    pub expected_answer: Option<String>,
    pub operation: Option<String>,
    pub lhs: Option<i64>,
    pub rhs: Option<i64>,
    pub fired: bool,
    pub answer_seen: bool,
    pub observed_step: Option<usize>,
    pub replacement_answer: Option<String>,
    pub stop_reason: Option<String>,
}

pub struct AnswerBoundaryFinalizer {
    enabled: bool,
    expectation: Option<AnswerBoundaryExpectation>,
    fired: bool,
    answer_seen: bool,
    observed_step: Option<usize>,
    replacement_answer: Option<String>,
    stop_reason: Option<String>,
}

impl AnswerBoundaryFinalizer {
    pub fn from_prompt(enabled: bool, prompt: &str) -> Self {
        let expectation = if enabled {
            detect_answer_boundary_expectation(prompt)
        } else {
            None
        };
        Self {
            enabled,
            expectation,
            fired: false,
            answer_seen: false,
            observed_step: None,
            replacement_answer: None,
            stop_reason: None,
        }
    }

    pub fn expectation(&self) -> Option<&AnswerBoundaryExpectation> {
        self.expectation.as_ref()
    }

    pub fn observe_text(&mut self, step: usize, recent_text: &str) -> FinalizationDecision {
        if !self.enabled || self.fired {
            return FinalizationDecision::continue_generation();
        }
        let Some(exp) = self.expectation.as_ref() else {
            return FinalizationDecision::continue_generation();
        };
        if !answer_surface_seen(exp, recent_text) {
            return FinalizationDecision::continue_generation();
        }
        self.fired = true;
        self.answer_seen = true;
        self.observed_step = Some(step);
        self.replacement_answer = Some(exp.expected_answer.clone());
        let reason = format!("answer_boundary_{}_seen", exp.kind.as_str());
        self.stop_reason = Some(reason.clone());
        FinalizationDecision::stop(reason)
    }

    pub fn snapshot(&self) -> AnswerBoundarySnapshot {
        AnswerBoundarySnapshot {
            enabled: self.enabled,
            kind: self
                .expectation
                .as_ref()
                .map(|e| e.kind.as_str().to_string()),
            source_term: self.expectation.as_ref().map(|e| e.source_term.clone()),
            expected_answer: self.expectation.as_ref().map(|e| e.expected_answer.clone()),
            operation: self.expectation.as_ref().and_then(|e| e.operation.clone()),
            lhs: self.expectation.as_ref().and_then(|e| e.lhs),
            rhs: self.expectation.as_ref().and_then(|e| e.rhs),
            fired: self.fired,
            answer_seen: self.answer_seen,
            observed_step: self.observed_step,
            replacement_answer: self.replacement_answer.clone(),
            stop_reason: self.stop_reason.clone(),
        }
    }
}

pub fn detect_answer_boundary_expectation(prompt: &str) -> Option<AnswerBoundaryExpectation> {
    detect_literal(prompt)
        .or_else(|| detect_word_reverse(prompt))
        .or_else(|| detect_arithmetic(prompt))
}

static LITERAL_PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
static REVERSE_PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
static ARITHMETIC_PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();

fn literal_patterns() -> &'static [Regex] {
    LITERAL_PATTERNS.get_or_init(|| {
        [
            r#"(?i)output\s+exactly\s+['"]?([A-Za-z][A-Za-z0-9\-]*)['"]?\s+and\s+nothing\s+else\.?\s*$"#,
            r#"(?i)output\s+exactly\s+(?:the\s+)?(?:single\s+)?word\s*:\s*['"]?([A-Za-z][A-Za-z0-9\-]*)['"]?\.?\s*$"#,
            r#"(?i)reply\s+with\s*:?\s*['"]?([A-Za-z][A-Za-z0-9\-]*)['"]?\.?\s*$"#,
            r#"(?i)say\s+only\s*:?\s*['"]?([A-Za-z][A-Za-z0-9\-]*)['"]?\.?\s*$"#,
            r#"(?i)respond\s+with\s+(?:the\s+word\s*:?\s*)?['"]?([A-Za-z][A-Za-z0-9\-]*)['"]?\.?\s*$"#,
            r#"(?i)\bprint\s*:\s*['"]?([A-Za-z][A-Za-z0-9\-]*)['"]?\.?\s*$"#,
        ]
        .iter()
        .map(|p| Regex::new(p).expect("literal pattern"))
        .collect()
    })
}

fn reverse_patterns() -> &'static [Regex] {
    REVERSE_PATTERNS.get_or_init(|| {
        [
            r#"(?i)\breverse\s+(?:the\s+)?(?:word|string|letters\s+of)\s+['"]?([A-Za-z][A-Za-z\-]*)['"]?"#,
            r#"(?i)\bspell\s+['"]?([A-Za-z][A-Za-z\-]*)['"]?\s+backward"#,
            r#"(?i)\b(?:write|type|render|print)\s+(?:the\s+)?(?:(?:word|string)\s+)?['"]?([A-Za-z][A-Za-z\-]*)['"]?\s+backwards?\b"#,
            r#"(?i)\bwhat\s+is\s+['"]?([A-Za-z][A-Za-z\-]*)['"]?\s+reversed\b"#,
            r#"(?i)\b['"]?([A-Za-z][A-Za-z\-]*)['"]?\s+(?:written|typed|rendered|printed)\s+backwards?\b"#,
        ]
        .iter()
        .map(|p| Regex::new(p).expect("reverse pattern"))
        .collect()
    })
}

fn arithmetic_patterns() -> &'static [Regex] {
    ARITHMETIC_PATTERNS.get_or_init(|| {
        [
            r"(?i)\bwhat\s+is\s+(-?\d+)\s+(plus|minus|times|multiplied\s+by|divided\s+by)\s+(-?\d+)\b",
            r"(?i)\bcalculate\s+(-?\d+)\s+(plus|minus|times|multiplied\s+by|divided\s+by)\s+(-?\d+)\b",
            r"(?i)\b(-?\d+)\s+(plus|minus|times|multiplied\s+by|divided\s+by)\s+(-?\d+)\b",
        ]
        .iter()
        .map(|p| Regex::new(p).expect("arithmetic pattern"))
        .collect()
    })
}

fn detect_literal(prompt: &str) -> Option<AnswerBoundaryExpectation> {
    let trimmed = prompt.trim();
    for rx in literal_patterns() {
        if let Some(caps) = rx.captures(trimmed) {
            let word = caps.get(1)?.as_str().to_string();
            return Some(AnswerBoundaryExpectation {
                kind: AnswerBoundaryKind::Literal,
                source_term: word.clone(),
                expected_answer: word,
                operation: None,
                lhs: None,
                rhs: None,
            });
        }
    }
    None
}

fn detect_word_reverse(prompt: &str) -> Option<AnswerBoundaryExpectation> {
    let trimmed = prompt.trim();
    for rx in reverse_patterns() {
        if let Some(caps) = rx.captures(trimmed) {
            let source = caps.get(1)?.as_str().to_string();
            let reversed: String = source.chars().rev().collect();
            return Some(AnswerBoundaryExpectation {
                kind: AnswerBoundaryKind::WordReverse,
                source_term: source,
                expected_answer: reversed,
                operation: None,
                lhs: None,
                rhs: None,
            });
        }
    }
    None
}

fn detect_arithmetic(prompt: &str) -> Option<AnswerBoundaryExpectation> {
    let trimmed = prompt.trim();
    for rx in arithmetic_patterns() {
        if let Some(caps) = rx.captures(trimmed) {
            let a: i64 = caps.get(1)?.as_str().parse().ok()?;
            let op_raw = caps.get(2)?.as_str().to_lowercase();
            let op = op_raw.split_whitespace().collect::<Vec<_>>().join(" ");
            let b: i64 = caps.get(3)?.as_str().parse().ok()?;
            let answer = match op.as_str() {
                "plus" => a + b,
                "minus" => a - b,
                "times" | "multiplied by" => a * b,
                "divided by" => {
                    if b == 0 || a % b != 0 {
                        return None;
                    }
                    a / b
                }
                _ => return None,
            };
            return Some(AnswerBoundaryExpectation {
                kind: AnswerBoundaryKind::Arithmetic,
                source_term: format!("{} {} {}", a, op, b),
                expected_answer: answer.to_string(),
                operation: Some(op),
                lhs: Some(a),
                rhs: Some(b),
            });
        }
    }
    None
}

fn answer_surface_seen(exp: &AnswerBoundaryExpectation, text: &str) -> bool {
    let answer = &exp.expected_answer;
    match exp.kind {
        AnswerBoundaryKind::Literal => word_boundary_seen(text, answer),
        AnswerBoundaryKind::WordReverse => {
            if word_boundary_seen(text, answer) {
                return true;
            }
            hyphenated_letters_seen(text, answer)
        }
        AnswerBoundaryKind::Arithmetic => digit_boundary_seen(text, answer),
    }
}

fn word_boundary_seen(text: &str, term: &str) -> bool {
    if term.is_empty() {
        return false;
    }
    let lower_text = text.to_ascii_lowercase();
    let lower_term = term.to_ascii_lowercase();
    let bytes = lower_text.as_bytes();
    let term_bytes = lower_term.as_bytes();
    let mut start = 0;
    while let Some(idx) = lower_text[start..].find(&lower_term as &str) {
        let pos = start + idx;
        let before_ok =
            pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric() && bytes[pos - 1] != b'_';
        let end = pos + term_bytes.len();
        let after_ok =
            end == bytes.len() || !bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_';
        if before_ok && after_ok {
            return true;
        }
        start = pos + 1;
    }
    false
}

fn digit_boundary_seen(text: &str, term: &str) -> bool {
    if term.is_empty() {
        return false;
    }
    let bytes = text.as_bytes();
    let term_bytes = term.as_bytes();
    let mut start = 0;
    while let Some(idx) = text[start..].find(term) {
        let pos = start + idx;
        let before_ok = pos == 0 || !bytes[pos - 1].is_ascii_digit();
        let end = pos + term_bytes.len();
        let after_ok = end == bytes.len() || !bytes[end].is_ascii_digit();
        if before_ok && after_ok {
            return true;
        }
        start = pos + 1;
    }
    false
}

fn hyphenated_letters_seen(text: &str, term: &str) -> bool {
    if term.chars().count() < 2 {
        return false;
    }
    let term_chars: Vec<char> = term.chars().map(|c| c.to_ascii_lowercase()).collect();
    let bytes: Vec<char> = text.chars().collect();
    if bytes.is_empty() {
        return false;
    }
    let mut start = 0;
    while start < bytes.len() {
        if let Some(end) = match_hyphenated_at(&bytes, start, &term_chars) {
            let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphabetic();
            let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphabetic();
            if before_ok && after_ok {
                return true;
            }
        }
        start += 1;
    }
    false
}

fn match_hyphenated_at(text: &[char], start: usize, term: &[char]) -> Option<usize> {
    let mut pos = start;
    for (i, expected) in term.iter().enumerate() {
        if i > 0 {
            // Require at least one '-' separator, optionally surrounded by whitespace.
            let mut saw_dash = false;
            while pos < text.len() && (text[pos].is_whitespace() || text[pos] == '-') {
                if text[pos] == '-' {
                    if saw_dash {
                        return None;
                    }
                    saw_dash = true;
                }
                pos += 1;
            }
            if !saw_dash {
                return None;
            }
        }
        if pos >= text.len() {
            return None;
        }
        if text[pos].to_ascii_lowercase() != *expected {
            return None;
        }
        pos += 1;
    }
    Some(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_lock_surface_variants() {
        assert!(find_lock_like_surface("[REQUEST: LOCK] deadline=Friday").is_some());
        assert!(find_lock_like_surface("[LOCK] owner=Jason").is_some());
        assert!(find_lock_like_surface("AGENCY HANDS: LOCK budget=60k").is_some());
        assert!(find_lock_like_surface("LOCK budget=60k").is_some());
        assert!(find_lock_like_surface("The answer is now locked in.").is_some());
    }

    #[test]
    fn off_policy_detects_without_stopping() {
        let mut controller = FinalizationController::new(LockStopPolicy::Off, 8, false);
        let decision = controller.observe_token(4, "WORKING ANSWER: Friday [REQUEST: LOCK]", "]");
        let snapshot = controller.snapshot();
        assert!(snapshot.lock_detected);
        assert!(!decision.should_stop);
        assert!(!snapshot.lock_stop_triggered);
    }

    #[test]
    fn immediate_policy_stops_on_lock() {
        let mut controller = FinalizationController::new(LockStopPolicy::Immediate, 8, false);
        let decision = controller.observe_token(4, "WORKING ANSWER: Friday [REQUEST: LOCK]", "]");
        assert!(decision.should_stop);
        assert_eq!(decision.reason.as_deref(), Some("lock_immediate"));
    }

    #[test]
    fn taper_policy_stops_after_budget() {
        let mut controller = FinalizationController::new(LockStopPolicy::Taper, 2, false);
        assert!(
            !controller
                .observe_token(4, "WORKING ANSWER: Friday [REQUEST: LOCK]", "]")
                .should_stop
        );
        assert!(!controller.observe_token(5, "x", "x").should_stop);
        let decision = controller.observe_token(6, "xy", "y");
        assert!(decision.should_stop);
        assert_eq!(decision.reason.as_deref(), Some("lock_taper_exhausted"));
        assert_eq!(controller.snapshot().tokens_after_lock, Some(2));
    }

    #[test]
    fn veto_current_stop_allows_generation_after_guard() {
        let mut controller = FinalizationController::new(LockStopPolicy::Taper, 1, false);
        assert!(
            !controller
                .observe_token(0, "WORKING ANSWER: 50 hours [REQUEST: LOCK]", "]")
                .should_stop
        );
        assert!(
            controller
                .observe_token(1, "WORKING ANSWER: 50 hours [REQUEST: LOCK]\n", "\n")
                .should_stop
        );
        controller.veto_current_stop();
        let snapshot = controller.snapshot();
        assert!(!snapshot.lock_stop_triggered);
        assert_eq!(snapshot.lock_taper_remaining, Some(1));
    }

    #[test]
    fn taper_policy_stops_on_newline_boundary() {
        let mut controller = FinalizationController::new(LockStopPolicy::Taper, 8, false);
        assert!(
            !controller
                .observe_token(4, "WORKING ANSWER: Friday [REQUEST: LOCK]", "]")
                .should_stop
        );
        let decision = controller.observe_token(5, "WORKING ANSWER: Friday\n", "\n");
        assert!(decision.should_stop);
        assert_eq!(decision.reason.as_deref(), Some("lock_taper_boundary"));
    }

    #[test]
    fn taper_stops_on_lock_reemission_in_post_lock_window() {
        // Repair for ANSWER_PRESENT_LOCK_MISS: if the model re-emits [REQUEST: LOCK]
        // in the taper window, stop immediately rather than running until exhaustion.
        let mut controller = FinalizationController::new(LockStopPolicy::Taper, 16, false);
        assert!(
            !controller
                .observe_token(4, "WORKING ANSWER: 60000 [REQUEST: LOCK]", "]")
                .should_stop
        );
        // Model echoes LOCK payload then re-emits the LOCK tag — should stop.
        let post_lock = "WORKING ANSWER: 60000 [REQUEST: LOCK] Budget=60000 [REQUEST: LOCK]";
        let decision = controller.observe_token(5, post_lock, "]");
        assert!(
            decision.should_stop,
            "taper should stop on LOCK re-emission"
        );
        assert_eq!(decision.reason.as_deref(), Some("lock_taper_boundary"));
    }

    #[test]
    fn taper_ignores_request_tags_before_lock_surface() {
        let mut controller = FinalizationController::new(LockStopPolicy::Taper, 8, false);
        let before_lock = "[REQUEST: SPIKE]\n[REQUEST: EXPLORE]\n[REQUEST: FOCUS]\n[REQUEST: RESET]\n[REQUEST: LOCK]";
        assert!(!controller.observe_token(4, before_lock, "]").should_stop);

        let with_payload = format!("{before_lock} done");
        let decision = controller.observe_token(5, &with_payload, " done");
        assert!(!decision.should_stop);

        let with_newline = format!("{with_payload}\n");
        let decision = controller.observe_token(6, &with_newline, "\n");
        assert!(decision.should_stop);
        assert_eq!(decision.reason.as_deref(), Some("lock_taper_boundary"));
    }

    #[test]
    fn answer_boundary_literal_detects_and_fires() {
        let mut fin =
            AnswerBoundaryFinalizer::from_prompt(true, "Output exactly violet and nothing else.");
        let exp = fin.expectation().expect("should detect");
        assert_eq!(exp.kind, AnswerBoundaryKind::Literal);
        assert_eq!(exp.expected_answer, "violet");

        let early = fin.observe_text(2, "Sure, the color is");
        assert!(!early.should_stop);

        let later = fin.observe_text(8, "Sure, the color is violet.");
        assert!(later.should_stop);
        assert_eq!(
            later.reason.as_deref(),
            Some("answer_boundary_literal_seen")
        );
        let snap = fin.snapshot();
        assert!(snap.fired);
        assert_eq!(snap.replacement_answer.as_deref(), Some("violet"));
        assert_eq!(snap.observed_step, Some(8));
    }

    #[test]
    fn answer_boundary_literal_word_boundary_rejects_substring() {
        let mut fin = AnswerBoundaryFinalizer::from_prompt(true, "Reply with: red");
        assert_eq!(
            fin.expectation().map(|e| e.expected_answer.clone()),
            Some("red".to_string())
        );
        let decision = fin.observe_text(3, "I rendered the answer");
        assert!(!decision.should_stop, "must not match inside `rendered`");
        let decision = fin.observe_text(5, "I picked red.");
        assert!(decision.should_stop);
    }

    #[test]
    fn answer_boundary_word_reverse_detects_reversed_or_hyphenated() {
        let mut fin = AnswerBoundaryFinalizer::from_prompt(true, "Spell red backward.");
        let exp = fin.expectation().expect("should detect");
        assert_eq!(exp.kind, AnswerBoundaryKind::WordReverse);
        assert_eq!(exp.source_term, "red");
        assert_eq!(exp.expected_answer, "der");

        assert!(!fin.observe_text(0, "Thinking").should_stop);
        let decision = fin.observe_text(2, "The answer is der.");
        assert!(decision.should_stop);
    }

    #[test]
    fn answer_boundary_word_reverse_accepts_hyphenated_letters() {
        let mut fin = AnswerBoundaryFinalizer::from_prompt(true, "What is hello reversed");
        let exp = fin.expectation().expect("should detect");
        assert_eq!(exp.expected_answer, "olleh");
        let decision = fin.observe_text(4, "Letter by letter: o-l-l-e-h");
        assert!(decision.should_stop);
    }

    #[test]
    fn answer_boundary_arithmetic_detects_and_gates_on_seen_answer() {
        let mut fin = AnswerBoundaryFinalizer::from_prompt(true, "What is 7 plus 5?");
        let exp = fin.expectation().expect("should detect");
        assert_eq!(exp.kind, AnswerBoundaryKind::Arithmetic);
        assert_eq!(exp.expected_answer, "12");
        assert_eq!(exp.lhs, Some(7));
        assert_eq!(exp.rhs, Some(5));
        assert_eq!(exp.operation.as_deref(), Some("plus"));

        // 12 inside "120" must NOT fire (digit boundary).
        let decision = fin.observe_text(3, "There are 120 ways");
        assert!(!decision.should_stop);

        let decision = fin.observe_text(7, "The answer is 12.");
        assert!(decision.should_stop);
        assert_eq!(
            decision.reason.as_deref(),
            Some("answer_boundary_arithmetic_seen")
        );
    }

    #[test]
    fn answer_boundary_arithmetic_handles_all_operations() {
        for (prompt, expected) in [
            ("What is 8 minus 3?", "5"),
            ("Calculate 6 times 4.", "24"),
            ("What is 4 multiplied by 5?", "20"),
            ("What is 20 divided by 4?", "5"),
        ] {
            let fin = AnswerBoundaryFinalizer::from_prompt(true, prompt);
            assert_eq!(
                fin.expectation().map(|e| e.expected_answer.clone()),
                Some(expected.to_string()),
                "prompt: {prompt}"
            );
        }
    }

    #[test]
    fn answer_boundary_arithmetic_rejects_non_integer_division() {
        let fin = AnswerBoundaryFinalizer::from_prompt(true, "What is 7 divided by 2?");
        assert!(fin.expectation().is_none());
    }

    #[test]
    fn answer_boundary_disabled_never_fires() {
        let mut fin = AnswerBoundaryFinalizer::from_prompt(false, "Reply with: violet");
        assert!(fin.expectation().is_none());
        let decision = fin.observe_text(5, "violet");
        assert!(!decision.should_stop);
        assert!(!fin.snapshot().fired);
    }

    #[test]
    fn answer_boundary_no_match_yields_no_expectation() {
        let fin = AnswerBoundaryFinalizer::from_prompt(true, "Tell me about the weather.");
        assert!(fin.expectation().is_none());
    }

    #[test]
    fn answer_boundary_fires_only_once() {
        let mut fin = AnswerBoundaryFinalizer::from_prompt(true, "Reply with: blue");
        let first = fin.observe_text(2, "I think blue.");
        assert!(first.should_stop);
        let second = fin.observe_text(3, "I think blue and also red.");
        // Subsequent observations should NOT re-fire (already done).
        assert!(!second.should_stop);
    }
}
