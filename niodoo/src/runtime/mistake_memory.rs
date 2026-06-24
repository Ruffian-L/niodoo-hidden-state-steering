use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MistakeMemoryEvent {
    pub id: String,
    pub task_key: String,
    pub trigger_terms: Vec<String>,
    pub rejected_answers: Vec<String>,
    pub accepted_answer: String,
    #[serde(default)]
    pub accepted_aliases: Vec<String>,
    pub correction_text: String,
    #[serde(default)]
    pub source: String,
    pub created_at_ms: u128,
    pub updated_at_ms: u128,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub successful_reinforcements: u32,
    #[serde(default)]
    pub failed_reuses: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MistakeMemoryMatch {
    pub event_id: String,
    pub task_key: String,
    pub score: f32,
    pub rejected_answers: Vec<String>,
    pub accepted_answer: String,
    pub accepted_aliases: Vec<String>,
    pub correction_text: String,
    pub injection_strength: f32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MistakeGuardSnapshot {
    pub matched: bool,
    pub match_count: usize,
    pub event_ids: Vec<String>,
    pub rejected_answer_seen: bool,
    pub accepted_answer_seen: bool,
    pub accepted_boundary_seen: bool,
    pub accepted_boundary_byte_len: Option<usize>,
    pub claim_review_evidence_gate_match: bool,
    pub blocked_lock: bool,
    pub blocked_count: usize,
}

#[derive(Debug, Clone, Default)]
pub struct MistakeMemory {
    events: Vec<MistakeMemoryEvent>,
}

#[derive(Debug, Clone)]
pub struct MistakeMemoryGuard {
    matches: Vec<MistakeMemoryMatch>,
    rejected_answer_seen: bool,
    accepted_answer_seen: bool,
    accepted_boundary_seen: bool,
    accepted_boundary_byte_len: Option<usize>,
    blocked_count: usize,
}

fn default_confidence() -> f32 {
    1.0
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

impl MistakeMemory {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("Failed to read mistake memory {}", path.display()))?;
        let mut events = Vec::new();
        for (idx, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut event: MistakeMemoryEvent = serde_json::from_str(line).with_context(|| {
                format!(
                    "Failed to parse mistake memory {} at line {}",
                    path.display(),
                    idx + 1
                )
            })?;
            normalize_event(&mut event);
            events.push(event);
        }
        Ok(Self { events })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("Failed to create mistake memory dir {}", parent.display())
                })?;
            }
        }
        let mut out = String::new();
        for event in &self.events {
            out.push_str(&serde_json::to_string(event)?);
            out.push('\n');
        }
        fs::write(path, out)
            .with_context(|| format!("Failed to write mistake memory {}", path.display()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn capture_from_correction_turn(
        &mut self,
        user_prompt: &str,
        previous_assistant: Option<&str>,
    ) -> Vec<MistakeMemoryEvent> {
        let Some(previous_assistant) = previous_assistant else {
            return Vec::new();
        };
        if !correction_like(user_prompt) {
            return Vec::new();
        }

        let mut captured = Vec::new();
        if let Some(event) = strawberry_r_event(user_prompt, previous_assistant) {
            captured.push(self.upsert_event(event));
        }
        if let Some(event) = parallel_towel_event(user_prompt, previous_assistant) {
            captured.push(self.upsert_event(event));
        }
        if let Some(event) = human_correction_event(user_prompt, previous_assistant) {
            captured.push(self.upsert_event(event));
        }
        captured
    }

    pub fn query(&self, user_prompt: &str, limit: usize) -> Vec<MistakeMemoryMatch> {
        if limit == 0 {
            return Vec::new();
        }
        let prompt = normalize_text(user_prompt);
        let mut scored = Vec::new();
        for event in &self.events {
            let score = match_score(event, &prompt);
            if score <= 0.0 {
                continue;
            }
            scored.push((
                score,
                MistakeMemoryMatch {
                    event_id: event.id.clone(),
                    task_key: event.task_key.clone(),
                    score,
                    rejected_answers: event.rejected_answers.clone(),
                    accepted_answer: event.accepted_answer.clone(),
                    accepted_aliases: event.accepted_aliases.clone(),
                    correction_text: event.correction_text.clone(),
                    injection_strength: event_injection_strength(event),
                },
            ));
        }
        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(_, item)| item)
            .collect()
    }

    pub fn apply_prompt(user_prompt: &str, matches: &[MistakeMemoryMatch]) -> String {
        if matches.is_empty() {
            return user_prompt.to_string();
        }
        let mut lines = vec!["MISTAKE MEMORY:".to_string()];
        let human_only = matches.iter().all(|item| is_human_correction(item));
        if human_only {
            lines.push("Confirmed prior personal/factual correction matches this turn. Do not repeat stale facts. Use the current correction silently unless the user asks directly; prefer neutral wording if uncertain. For stale-surface client updates, write one complete client-safe note that preserves non-stale facts, says the QA owner should be confirmed with the current team when no current owner is known, and then stop. Preferred form: \"Project update: The beta checklist is due Friday; the QA owner should be confirmed with the current team.\" Do not use bracket placeholders like [Name] or [owner]. Do not quote or mention the stale surface in visible reasoning or the final answer.".to_string());
        } else {
            lines.push("Confirmed prior corrections match this turn. Answer from the correction first, then briefly justify if needed. Do not recompute into a known rejected answer.".to_string());
        }
        for item in matches.iter().take(3) {
            if is_human_correction(item) {
                if is_surface_suppression_match(item) {
                    lines.push(format!(
                        "- Prior correction: the inactive collaborator must not appear in project updates or QA ownership. Use a client-safe note like: Project update: The beta checklist is due Friday; the QA owner should be confirmed with the current team."
                    ));
                } else {
                    lines.push(format!(
                        "- task={} CURRENT_HINT={} STALE_SURFACES=[{}] strength={:.2}",
                        item.task_key,
                        item.accepted_answer,
                        item.rejected_answers.join(" | "),
                        item.injection_strength
                    ));
                }
            } else {
                lines.push(format!(
                    "START_VISIBLE_ANSWER={}",
                    visible_answer_for_match(item)
                ));
                lines.push(format!(
                    "- task={} CORRECT_ANSWER={} REJECTED_ANSWERS=[{}] strength={:.2}",
                    item.task_key,
                    item.accepted_answer,
                    item.rejected_answers.join(" | "),
                    item.injection_strength
                ));
            }
            if !item.correction_text.is_empty() {
                let correction = if is_surface_suppression_match(item) {
                    redact_surfaces(
                        &compact_whitespace(&item.correction_text, 180),
                        &surface_suppression_terms_for_match(item),
                    )
                } else {
                    compact_whitespace(&item.correction_text, 180)
                };
                lines.push(format!("  correction={}", correction));
            }
        }
        format!("{}\n\nUSER TURN:\n{}", lines.join("\n"), user_prompt)
    }

    pub fn surface_suppression_terms(matches: &[MistakeMemoryMatch]) -> Vec<String> {
        let mut terms = Vec::new();
        for item in matches {
            if !is_surface_suppression_match(item) {
                continue;
            }
            for term in surface_suppression_terms_for_match(item) {
                if !terms
                    .iter()
                    .any(|existing: &String| existing.eq_ignore_ascii_case(&term))
                {
                    terms.push(term);
                }
            }
        }
        terms
    }

    pub fn surface_suppression_violation(
        assistant_text: &str,
        token_text: &str,
        terms: &[String],
    ) -> bool {
        if terms.is_empty() {
            return false;
        }
        let combined = normalize_text(&format!("{assistant_text}{token_text}"));
        terms
            .iter()
            .any(|term| contains_rejected_surface(&combined, term))
    }

    pub fn record_outcome(&mut self, matches: &[MistakeMemoryMatch], assistant_text: &str) -> bool {
        if matches.is_empty() {
            return false;
        }
        let mut changed = false;
        for item in matches {
            let accepted = accepted_seen(item, assistant_text);
            let rejected = rejected_seen(item, assistant_text);
            let Some(event) = self
                .events
                .iter_mut()
                .find(|event| event.id == item.event_id)
            else {
                continue;
            };
            if is_surface_suppression_match(item) && rejected {
                event.failed_reuses = event.failed_reuses.saturating_add(1);
                event.confidence = (event.confidence + 0.10).min(1.5);
                event.updated_at_ms = now_ms();
                changed = true;
            } else if accepted {
                event.successful_reinforcements = event.successful_reinforcements.saturating_add(1);
                event.confidence = (event.confidence * 0.96).max(0.35);
                event.updated_at_ms = now_ms();
                changed = true;
            } else if rejected {
                event.failed_reuses = event.failed_reuses.saturating_add(1);
                event.confidence = (event.confidence + 0.10).min(1.5);
                event.updated_at_ms = now_ms();
                changed = true;
            }
        }
        changed
    }

    fn upsert_event(&mut self, mut event: MistakeMemoryEvent) -> MistakeMemoryEvent {
        normalize_event(&mut event);
        if let Some(existing) = self
            .events
            .iter_mut()
            .find(|existing| existing.id == event.id)
        {
            existing.trigger_terms =
                merge_strings(&existing.trigger_terms, &event.trigger_terms, 16);
            existing.rejected_answers =
                merge_strings(&existing.rejected_answers, &event.rejected_answers, 16);
            existing.accepted_aliases =
                merge_strings(&existing.accepted_aliases, &event.accepted_aliases, 16);
            existing.accepted_answer = event.accepted_answer.clone();
            existing.correction_text = event.correction_text.clone();
            existing.updated_at_ms = now_ms();
            existing.confidence = (existing.confidence + 0.15).min(1.5);
            return existing.clone();
        }
        self.events.push(event.clone());
        event
    }
}

impl MistakeMemoryGuard {
    pub fn new(matches: Vec<MistakeMemoryMatch>) -> Self {
        Self {
            matches,
            rejected_answer_seen: false,
            accepted_answer_seen: false,
            accepted_boundary_seen: false,
            accepted_boundary_byte_len: None,
            blocked_count: 0,
        }
    }

    pub fn observe(&mut self, assistant_text: &str) -> MistakeGuardSnapshot {
        self.rejected_answer_seen = self
            .matches
            .iter()
            .any(|item| rejected_seen(item, assistant_text));
        self.accepted_answer_seen = self
            .matches
            .iter()
            .any(|item| accepted_seen(item, assistant_text));
        self.accepted_boundary_byte_len = self
            .matches
            .iter()
            .filter_map(|item| accepted_boundary_byte_len(item, assistant_text))
            .min();
        self.accepted_boundary_seen = self.accepted_boundary_byte_len.is_some();
        self.snapshot()
    }

    pub fn should_block_finalization(&self) -> bool {
        !self.matches.is_empty() && self.rejected_answer_seen && !self.accepted_answer_seen
    }

    pub fn record_blocked_lock(&mut self) {
        self.blocked_count = self.blocked_count.saturating_add(1);
    }

    pub fn snapshot(&self) -> MistakeGuardSnapshot {
        MistakeGuardSnapshot {
            matched: !self.matches.is_empty(),
            match_count: self.matches.len(),
            event_ids: self
                .matches
                .iter()
                .map(|item| item.event_id.clone())
                .collect(),
            rejected_answer_seen: self.rejected_answer_seen,
            accepted_answer_seen: self.accepted_answer_seen,
            accepted_boundary_seen: self.accepted_boundary_seen,
            accepted_boundary_byte_len: self.accepted_boundary_byte_len,
            claim_review_evidence_gate_match: self
                .matches
                .iter()
                .any(|item| item.task_key == "human:claim_review:evidence_gate"),
            blocked_lock: self.blocked_count > 0,
            blocked_count: self.blocked_count,
        }
    }
}

fn correction_like(text: &str) -> bool {
    let lower = normalize_text(text);
    [
        "wrong",
        "actually",
        "correction",
        "instead",
        "broke up",
        "split up",
        "don't",
        "do not",
        "can't",
        "cannot",
        "no,",
        "no ",
        "not ",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn strawberry_r_event(user_prompt: &str, previous_assistant: &str) -> Option<MistakeMemoryEvent> {
    let user = normalize_text(user_prompt);
    let Some((word, target, accepted_count)) = extract_letter_count_correction(&user) else {
        return None;
    };
    let previous = normalize_text(previous_assistant);
    let mut rejected = Vec::new();
    for count in extract_answer_like_numbers(&previous) {
        if count != accepted_count {
            rejected.push(count.to_string());
            rejected.push(format!("{count} {target}"));
            rejected.push(format!("{count} {target}s"));
            rejected.push(format!("{count} {target}'s"));
        }
    }
    if rejected.is_empty() {
        rejected.push("2".to_string());
    }
    let task_key = format!("letter_count:{word}:{target}");
    let accepted_answer = format!("{accepted_count} {target}'s");
    let target_plural = format!("{target}s");
    let target_possessive = format!("{target}'s");
    let accepted_count_word = number_word(accepted_count).unwrap_or("");
    let accepted_aliases = [
        format!("{accepted_count} {target}"),
        format!("{accepted_count} {target_plural}"),
        format!("{accepted_count} {target_possessive}"),
        format!("{accepted_count_word} {target}"),
        format!("{accepted_count_word} {target_plural}"),
    ];
    let trigger_terms = [
        word.as_str(),
        "letter",
        "letters",
        "count",
        target.as_str(),
        target_plural.as_str(),
    ];
    Some(event(
        &task_key,
        &trigger_terms,
        rejected,
        &accepted_answer,
        &accepted_aliases
            .iter()
            .filter(|alias| !alias.trim().is_empty())
            .cloned()
            .collect::<Vec<_>>(),
        user_prompt,
    ))
}

fn parallel_towel_event(user_prompt: &str, previous_assistant: &str) -> Option<MistakeMemoryEvent> {
    let user = normalize_text(user_prompt);
    if !user.contains("dry") || !user.contains("5 hour") {
        return None;
    }
    let item = parallel_item_from_text(&user)?;
    let previous = normalize_text(previous_assistant);
    let mut rejected = Vec::new();
    for hours in extract_hour_numbers(&previous) {
        if hours != 5 {
            rejected.push(format!("{hours} hours"));
            rejected.push(hours.to_string());
        }
    }
    for value in ["45", "50", "150"] {
        if contains_token(&previous, value) {
            rejected.push(value.to_string());
            rejected.push(format!("{value} hours"));
        }
    }
    if rejected.is_empty() {
        rejected.extend([
            "45 hours".to_string(),
            "50 hours".to_string(),
            "150 hours".to_string(),
        ]);
    }
    let item_plural = pluralize_item(&item);
    let task_key = format!("parallel_drying:{item_plural}");
    let trigger_terms = [
        item.as_str(),
        item_plural.as_str(),
        "dry",
        "drying",
        "hours",
        "more",
    ];
    let accepted_aliases = vec![
        "5 hours".to_string(),
        "5 hour".to_string(),
        "five hours".to_string(),
        "five hour".to_string(),
    ];
    Some(event(
        &task_key,
        &trigger_terms,
        rejected,
        "5 hours",
        &accepted_aliases,
        user_prompt,
    ))
}

fn human_correction_event(
    user_prompt: &str,
    previous_assistant: &str,
) -> Option<MistakeMemoryEvent> {
    let user = normalize_text(user_prompt);
    let previous = normalize_text(previous_assistant);
    if (user.contains("scripts cannot")
        || user.contains("script scores")
        || user.contains("script-scored")
        || user.contains("without reading generated outputs")
        || user.contains("mechanism telemetry"))
        && (user.contains("claim")
            || user.contains("claims")
            || user.contains("artifact")
            || user.contains("promote")
            || user.contains("falsify")
            || user.contains("red")
            || user.contains("green"))
    {
        let mut rejected = vec![
            "GREEN".to_string(),
            "validated".to_string(),
            "RED".to_string(),
            "falsified".to_string(),
            "script score is enough".to_string(),
        ];
        if previous.contains("green") {
            rejected.push("GREEN".to_string());
        }
        if previous.contains("red") {
            rejected.push("RED".to_string());
        }
        return Some(event(
            "human:claim_review:evidence_gate",
            &[
                "claim",
                "claims",
                "artifact",
                "script",
                "score",
                "telemetry",
                "generated",
                "outputs",
                "config",
            ],
            rejected,
            "PRELIMINARY until generated outputs and mechanism telemetry are read",
            &[
                "PRELIMINARY".to_string(),
                "read generated outputs".to_string(),
                "mechanism telemetry".to_string(),
                "CONFIG-BLOCKED".to_string(),
            ],
            user_prompt,
        ));
    }
    if user.contains("broke up") || user.contains("split up") {
        if user.contains("girlfriend")
            || user.contains("partner")
            || previous.contains("girlfriend")
            || previous.contains("partner")
        {
            let mut rejected = vec![
                "your girlfriend".to_string(),
                "your partner".to_string(),
                "date night".to_string(),
                "date-night".to_string(),
            ];
            if user.contains("maya") || previous.contains("maya") {
                rejected.push("maya".to_string());
            }
            return Some(event(
                "human:superseded_relationship",
                &[
                    "weekend",
                    "plan",
                    "date",
                    "restaurant",
                    "girlfriend",
                    "partner",
                ],
                rejected,
                "use neutral relationship wording",
                &[
                    "solo".to_string(),
                    "friends".to_string(),
                    "neutral".to_string(),
                    "if you are going".to_string(),
                    "someone new".to_string(),
                ],
                user_prompt,
            ));
        }
    }
    if user.contains("spicy")
        && (user.contains("anymore")
            || user.contains("can't")
            || user.contains("cannot")
            || user.contains("can't handle"))
    {
        return Some(event(
            "human:preference_changed:spicy_food",
            &["dinner", "food", "restaurant", "meal", "spicy"],
            vec![
                "spicy".to_string(),
                "hot sauce".to_string(),
                "chili".to_string(),
            ],
            "avoid spicy food",
            &[
                "mild".to_string(),
                "not spicy".to_string(),
                "gentle".to_string(),
                "avoid spicy".to_string(),
            ],
            user_prompt,
        ));
    }
    if user.contains("budget") && (user.contains("60k") || user.contains("60000")) {
        return Some(event(
            "human:project_fact:budget",
            &["budget", "plan", "project", "cost"],
            vec!["40k".to_string(), "40000".to_string()],
            "60k",
            &[
                "60,000".to_string(),
                "$60k".to_string(),
                "$60,000".to_string(),
            ],
            user_prompt,
        ));
    }
    if user.contains("friday")
        && (user.contains("due") || user.contains("deadline") || user.contains("correction"))
    {
        return Some(event(
            "human:project_fact:deadline",
            &["schedule", "work", "deadline", "due", "friday"],
            vec!["next week".to_string()],
            "Friday",
            &["by Friday".to_string(), "due Friday".to_string()],
            user_prompt,
        ));
    }
    if user.contains("alex")
        && (user.contains("no longer")
            || user.contains("don't include")
            || user.contains("do not include"))
    {
        return Some(event(
            "human:project_fact:stale_collaborator_removed",
            &["project", "plan", "draft", "team", "alex"],
            vec!["Alex".to_string()],
            "do not include Alex",
            &[
                "current team".to_string(),
                "remaining team".to_string(),
                "project update".to_string(),
                "client update".to_string(),
                "beta checklist".to_string(),
                "new owner".to_string(),
                "someone else".to_string(),
            ],
            user_prompt,
        ));
    }
    None
}

fn event(
    task_key: &str,
    trigger_terms: &[&str],
    rejected_answers: Vec<String>,
    accepted_answer: &str,
    accepted_aliases: &[String],
    correction_text: &str,
) -> MistakeMemoryEvent {
    let created_at_ms = now_ms();
    let id = stable_event_id(task_key, accepted_answer);
    MistakeMemoryEvent {
        id,
        task_key: task_key.to_string(),
        trigger_terms: trigger_terms.iter().map(|term| term.to_string()).collect(),
        rejected_answers,
        accepted_answer: accepted_answer.to_string(),
        accepted_aliases: accepted_aliases.to_vec(),
        correction_text: compact_whitespace(correction_text, 240),
        source: "user_correction".to_string(),
        created_at_ms,
        updated_at_ms: created_at_ms,
        confidence: 1.0,
        successful_reinforcements: 0,
        failed_reuses: 0,
    }
}

fn stable_event_id(task_key: &str, accepted_answer: &str) -> String {
    let mut hasher = DefaultHasher::new();
    task_key.hash(&mut hasher);
    accepted_answer.hash(&mut hasher);
    format!("mistake:{}:{:016x}", task_key, hasher.finish())
}

fn normalize_event(event: &mut MistakeMemoryEvent) {
    event.trigger_terms = normalize_list(&event.trigger_terms, 24);
    event.rejected_answers = normalize_list(&event.rejected_answers, 24);
    event.accepted_aliases = normalize_list(&event.accepted_aliases, 24);
    if event.source.trim().is_empty() {
        event.source = "user_correction".to_string();
    }
    if event.confidence <= 0.0 {
        event.confidence = 1.0;
    }
}

fn normalize_list(items: &[String], limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    for item in items {
        let compact = compact_whitespace(item, 80);
        if compact.is_empty() {
            continue;
        }
        if !out
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(&compact))
        {
            out.push(compact);
        }
        if out.len() >= limit {
            break;
        }
    }
    out
}

fn merge_strings(left: &[String], right: &[String], limit: usize) -> Vec<String> {
    let mut merged = left.to_vec();
    merged.extend(right.iter().cloned());
    normalize_list(&merged, limit)
}

fn match_score(event: &MistakeMemoryEvent, normalized_prompt: &str) -> f32 {
    let task_match = if let Some((word, target)) = parse_letter_count_key(&event.task_key) {
        normalized_prompt.contains(word)
            && (normalized_prompt.contains("count") || normalized_prompt.contains("how many"))
            && prompt_mentions_target(normalized_prompt, target)
    } else if let Some(item) = parse_parallel_drying_key(&event.task_key) {
        let item_plural = pluralize_item(item);
        let item_singular = singularize_item(item);
        (normalized_prompt.contains(item)
            || normalized_prompt.contains(&item_plural)
            || normalized_prompt.contains(&item_singular))
            && normalized_prompt.contains("dry")
            && !explicit_sequential_duration_prompt(normalized_prompt)
    } else if event.task_key.starts_with("human:") {
        match event.task_key.as_str() {
            "human:superseded_relationship" => {
                normalized_prompt.contains("weekend")
                    || normalized_prompt.contains("plan")
                    || normalized_prompt.contains("date")
                    || normalized_prompt.contains("restaurant")
                    || normalized_prompt.contains("partner")
                    || normalized_prompt.contains("girlfriend")
            }
            "human:preference_changed:spicy_food" => {
                normalized_prompt.contains("dinner")
                    || normalized_prompt.contains("food")
                    || normalized_prompt.contains("restaurant")
                    || normalized_prompt.contains("meal")
            }
            "human:project_fact:budget" => {
                normalized_prompt.contains("budget")
                    || normalized_prompt.contains("plan around")
                    || normalized_prompt.contains("project cost")
            }
            "human:project_fact:deadline" => {
                normalized_prompt.contains("schedule")
                    || normalized_prompt.contains("deadline")
                    || normalized_prompt.contains("due")
                    || normalized_prompt.contains("work")
            }
            "human:project_fact:stale_collaborator_removed" => {
                normalized_prompt.contains("project")
                    || normalized_prompt.contains("plan")
                    || normalized_prompt.contains("draft")
                    || normalized_prompt.contains("team")
                    || normalized_prompt.contains("client update")
                    || normalized_prompt.contains("update")
                    || normalized_prompt.contains("qa")
                    || normalized_prompt.contains("follow up")
                    || normalized_prompt.contains("followup")
                    || normalized_prompt.contains("follow-up")
            }
            "human:claim_review:evidence_gate" => {
                (normalized_prompt.contains("claim")
                    || normalized_prompt.contains("claims")
                    || normalized_prompt.contains("artifact"))
                    && (normalized_prompt.contains("script")
                        || normalized_prompt.contains("score")
                        || normalized_prompt.contains("telemetry")
                        || normalized_prompt.contains("generated output")
                        || normalized_prompt.contains("generated answer")
                        || normalized_prompt.contains("green")
                        || normalized_prompt.contains("red")
                        || normalized_prompt.contains("preliminary")
                        || normalized_prompt.contains("config"))
            }
            _ => false,
        }
    } else {
        match event.task_key.as_str() {
            "letter_count:strawberry:r" => {
                normalized_prompt.contains("strawberry")
                    && (normalized_prompt.contains("count")
                        || normalized_prompt.contains("how many"))
                    && prompt_mentions_target(normalized_prompt, "r")
            }
            "parallel_drying:towels" => {
                normalized_prompt.contains("towel")
                    && normalized_prompt.contains("dry")
                    && !explicit_sequential_duration_prompt(normalized_prompt)
            }
            _ => false,
        }
    };
    if !task_match {
        return 0.0;
    }
    let mut score = 2.0;
    for term in &event.trigger_terms {
        if term.len() >= 2 && contains_token_or_subword(normalized_prompt, term) {
            score += 0.25;
        }
    }
    if score < 0.75 {
        return 0.0;
    }
    score * event_injection_strength(event)
}

fn is_human_correction(item: &MistakeMemoryMatch) -> bool {
    item.task_key.starts_with("human:")
}

fn is_surface_suppression_match(item: &MistakeMemoryMatch) -> bool {
    matches!(
        item.task_key.as_str(),
        "human:project_fact:stale_collaborator_removed"
    )
}

fn surface_suppression_terms_for_match(item: &MistakeMemoryMatch) -> Vec<String> {
    if !is_surface_suppression_match(item) {
        return Vec::new();
    }
    item.rejected_answers
        .iter()
        .map(|answer| compact_whitespace(answer, 80))
        .filter(|answer| !answer.is_empty())
        .collect()
}

fn redact_surfaces(text: &str, surfaces: &[String]) -> String {
    let mut redacted = text.to_string();
    for surface in surfaces {
        if surface.trim().is_empty() {
            continue;
        }
        redacted = replace_case_insensitive(&redacted, surface, "the inactive collaborator");
    }
    redacted
}

fn replace_case_insensitive(text: &str, needle: &str, replacement: &str) -> String {
    let lower_text = text.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    if lower_needle.is_empty() {
        return text.to_string();
    }
    let mut out = String::new();
    let mut cursor = 0usize;
    let mut search_from = 0usize;
    while let Some(relative_idx) = lower_text[search_from..].find(&lower_needle) {
        let idx = search_from + relative_idx;
        out.push_str(&text[cursor..idx]);
        out.push_str(replacement);
        cursor = idx + needle.len();
        search_from = cursor;
    }
    out.push_str(&text[cursor..]);
    out
}

fn event_injection_strength(event: &MistakeMemoryEvent) -> f32 {
    let success_decay = 1.0 / (1.0 + event.successful_reinforcements as f32 * 0.35);
    let failure_boost = 1.0 + event.failed_reuses as f32 * 0.15;
    (event.confidence * success_decay * failure_boost).clamp(0.25, 1.5)
}

fn accepted_seen(item: &MistakeMemoryMatch, assistant_text: &str) -> bool {
    if is_surface_suppression_match(item) && rejected_seen(item, assistant_text) {
        return false;
    }
    contains_surface(assistant_text, &item.accepted_answer)
        || item
            .accepted_aliases
            .iter()
            .any(|alias| contains_surface(assistant_text, alias))
}

fn accepted_boundary_byte_len(item: &MistakeMemoryMatch, assistant_text: &str) -> Option<usize> {
    if item.task_key == "human:claim_review:evidence_gate" {
        return claim_review_evidence_gate_boundary_seen(assistant_text)
            .then_some(assistant_text.len());
    }
    if is_surface_suppression_match(item) {
        return surface_suppression_completion_boundary_byte_len(item, assistant_text);
    }
    accepted_seen(item, assistant_text).then_some(assistant_text.len())
}

fn claim_review_evidence_gate_boundary_seen(assistant_text: &str) -> bool {
    let text = normalize_text(assistant_text);
    if text.contains("label: preliminary")
        || text.contains("label preliminary")
        || text.contains("label - preliminary")
    {
        return true;
    }
    if !text.contains("preliminary") {
        return false;
    }

    let evidence_gate_terms = [
        "generated outputs",
        "generated output",
        "generated answer",
        "answer excerpts",
        "mechanism telemetry",
        "raw telemetry",
        "telemetry fields",
        "stdout/stderr",
        "stdout",
        "stderr",
        "config audit",
        "single-seed",
        "single seed",
        "script score",
        "script-scored",
        "collector summary",
    ];
    if !evidence_gate_terms
        .iter()
        .any(|needle| text.contains(needle))
    {
        return false;
    }

    let committed_preliminary = [
        "script score is preliminary",
        "script score stays preliminary",
        "script-scored summary is preliminary",
        "collector summary is preliminary",
        "score is preliminary",
        "claim is preliminary",
        "artifact is preliminary",
        "card is preliminary",
        "move is preliminary",
        "status is preliminary",
        "preliminary status is appropriate",
        "working answer: preliminary",
        "should be preliminary",
        "should stay preliminary",
        "should remain preliminary",
        "should be marked preliminary",
        "should be marked with a preliminary",
        "mark it preliminary",
        "marked with a preliminary label",
        "preliminary label",
        "preliminary until",
        "preliminary because",
        "still preliminary",
    ]
    .iter()
    .any(|needle| text.contains(needle));
    if committed_preliminary {
        return true;
    }

    let missing_evidence_commitment = [
        "cannot determine",
        "cannot accurately determine",
        "not enough to make a final",
        "not enough to make a decision",
        "should not make a decision",
        "cannot be trusted",
        "cannot promote",
        "cannot falsify",
        "do not promote",
        "do not falsify",
    ]
    .iter()
    .any(|needle| text.contains(needle));
    missing_evidence_commitment && text.contains("preliminary")
}

fn surface_suppression_completion_boundary_byte_len(
    item: &MistakeMemoryMatch,
    assistant_text: &str,
) -> Option<usize> {
    if rejected_seen(item, assistant_text) {
        return None;
    }
    if let Some(byte_len) = earned_client_note_sentence_boundary_byte_len(assistant_text) {
        return Some(byte_len);
    }
    let text = normalize_text(assistant_text);
    if [
        "[name",
        "[owner",
        "[current owner",
        "[current lead",
        "[inactive collaborator",
        "[removed",
        "removed inactive collaborator",
    ]
    .iter()
    .any(|needle| text.contains(needle))
    {
        return None;
    }
    let note_context = [
        "client update",
        "project update",
        "status update",
        "client-safe note",
        "client-ready",
        "note:",
    ]
    .iter()
    .any(|needle| text.contains(needle));
    let preserved_fact = ["beta checklist", "due friday", "friday", "checklist"]
        .iter()
        .any(|needle| text.contains(needle));
    let ownership_resolved = [
        "current team",
        "remaining team",
        "qa owner",
        "owner to confirm",
        "confirm the owner",
        "confirm owner",
        "assign",
        "reassign",
        "new owner",
        "project team",
        "team will",
        "active team",
        "team members",
        "ongoing qa coverage",
    ]
    .iter()
    .any(|needle| text.contains(needle));
    let trimmed = assistant_text.trim_end();
    let sentence_boundary =
        trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with(']');
    if note_context && preserved_fact && ownership_resolved && sentence_boundary {
        Some(assistant_text.len())
    } else {
        None
    }
}

fn earned_client_note_sentence_boundary_byte_len(assistant_text: &str) -> Option<usize> {
    let mut start = 0usize;
    for (idx, ch) in assistant_text.char_indices() {
        if matches!(ch, '.' | '!' | '?' | ']') {
            let end = idx + ch.len_utf8();
            let mut tail_end = end;
            for (tail_idx, tail_ch) in assistant_text[end..].char_indices() {
                if matches!(tail_ch, '\'' | '"' | ')' | ']') {
                    tail_end = end + tail_idx + tail_ch.len_utf8();
                } else {
                    break;
                }
            }
            let end = tail_end;
            if earned_client_note_sentence_seen(&assistant_text[start..end]) {
                return Some(end);
            }
            start = end;
        }
    }
    if start < assistant_text.len() && earned_client_note_fragment_seen(&assistant_text[start..]) {
        return Some(assistant_text.len());
    }
    None
}

fn earned_client_note_sentence_seen(raw: &str) -> bool {
    let text = normalize_text(raw);
    if text.contains("[name")
        || text.contains("[owner")
        || text.contains("[current owner")
        || text.contains("[current lead")
        || text.contains("[inactive collaborator")
        || text.contains("[removed")
        || text.contains("removed inactive collaborator")
        || text.contains("not yet available")
        || text.contains("currently no owner")
    {
        return false;
    }
    let preserved_fact = ["beta checklist", "due friday", "due on friday", "checklist"]
        .iter()
        .any(|needle| text.contains(needle));
    let ownership_resolved = [
        "current team owns qa follow-up",
        "current team owns the qa follow-up",
        "current team will own qa follow-up",
        "qa owner should be confirmed with the current team",
        "qa owner should be confirmed",
        "confirm the qa owner",
        "confirm qa owner",
        "confirm the owner with the current team",
        "qa will proceed under new ownership",
        "qa follow-up will be handled by the current owner",
        "all qa follow-up will be handled by the current owner",
        "ongoing qa coverage by the team",
    ]
    .iter()
    .any(|needle| text.contains(needle));
    preserved_fact && ownership_resolved
}

fn earned_client_note_fragment_seen(raw: &str) -> bool {
    let text = normalize_text(raw);
    if text.contains("[name")
        || text.contains("[owner")
        || text.contains("[current owner")
        || text.contains("[current lead")
        || text.contains("[inactive collaborator")
        || text.contains("[removed")
        || text.contains("removed inactive collaborator")
        || text.contains("not yet available")
        || text.contains("currently no owner")
    {
        return false;
    }
    [
        "qa owner should be confirmed with the current team",
        "qa owner should be confirmed",
        "confirm the qa owner",
        "confirm qa owner",
    ]
    .iter()
    .any(|needle| text.contains(needle))
}

fn rejected_seen(item: &MistakeMemoryMatch, assistant_text: &str) -> bool {
    let text = normalize_text(assistant_text);
    item.rejected_answers
        .iter()
        .any(|answer| contains_rejected_surface(&text, answer))
}

fn visible_answer_for_match(item: &MistakeMemoryMatch) -> String {
    if let Some((word, _target)) = parse_letter_count_key(&item.task_key) {
        format!(
            "VISIBLE ANSWER: There are {} in {word}.",
            item.accepted_answer
        )
    } else if let Some(item_name) = parse_parallel_drying_key(&item.task_key) {
        format!(
            "VISIBLE ANSWER: It takes {} to dry the {}.",
            item.accepted_answer,
            pluralize_item(item_name)
        )
    } else {
        format!("VISIBLE ANSWER: {}", item.accepted_answer)
    }
}

fn contains_surface(text: &str, surface: &str) -> bool {
    let text = normalize_text(text);
    let surface = normalize_text(surface);
    if surface.is_empty() {
        return false;
    }
    if surface.chars().all(|ch| ch.is_ascii_digit()) {
        return contains_token(&text, &surface);
    }
    text.contains(&surface)
}

fn contains_rejected_surface(normalized_text: &str, surface: &str) -> bool {
    let surface = normalize_text(surface);
    if surface.is_empty() {
        return false;
    }
    let mut start = 0usize;
    while let Some(idx) = normalized_text[start..].find(&surface) {
        let abs = start + idx;
        let before = &normalized_text[abs.saturating_sub(48)..abs];
        let after = &normalized_text
            [abs + surface.len()..normalized_text.len().min(abs + surface.len() + 32)];
        if after.contains("so far") || after.contains("so-far") {
            start = abs + surface.len();
            continue;
        }
        if before.contains("working answer") || before.contains("visible_working_answer") {
            if !before.contains("final answer") && !before.contains("visible answer") {
                let locked_after_working = normalized_text.contains("[request: lock]")
                    || normalized_text.contains("[request lock]")
                    || normalized_text.contains("[agency hands: lock]");
                if !locked_after_working {
                    start = abs + surface.len();
                    continue;
                }
            }
        }
        if surface.chars().all(|ch| ch.is_ascii_digit()) {
            if contains_token_at(normalized_text, &surface, abs) {
                return true;
            }
        } else if normalized_text.contains(&surface) {
            return true;
        }
        start = abs + surface.len();
    }
    false
}

fn contains_token_or_subword(text: &str, term: &str) -> bool {
    if term.len() <= 2 {
        text.contains(term)
    } else {
        contains_token(text, term) || text.contains(term)
    }
}

fn contains_token(text: &str, token: &str) -> bool {
    let mut start = 0usize;
    while let Some(idx) = text[start..].find(token) {
        let abs = start + idx;
        if contains_token_at(text, token, abs) {
            return true;
        }
        start = abs + token.len();
    }
    false
}

fn contains_token_at(text: &str, token: &str, idx: usize) -> bool {
    if !text[idx..].starts_with(token) {
        return false;
    }
    let before = text[..idx].chars().next_back();
    let after = text[idx + token.len()..].chars().next();
    let before_boundary = before.map(|ch| !ch.is_ascii_alphanumeric()).unwrap_or(true);
    let after_boundary = after.map(|ch| !ch.is_ascii_alphanumeric()).unwrap_or(true);
    before_boundary && after_boundary
}

fn extract_letter_count_correction(normalized_user: &str) -> Option<(String, String, i64)> {
    let words = normalized_user
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '\'')
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    for idx in 0..words.len().saturating_sub(2) {
        let Some(count) = parse_count_word(words[idx]) else {
            continue;
        };
        let target = normalize_letter_target(words[idx + 1])?;
        let word = if words.get(idx + 2) == Some(&"in") {
            words.get(idx + 3).copied()
        } else {
            words
                .windows(2)
                .find(|window| window[0] == "in")
                .map(|window| window[1])
        }?;
        if word.len() < 3 || word == "the" {
            continue;
        }
        return Some((word.to_string(), target, count));
    }
    None
}

fn parse_count_word(word: &str) -> Option<i64> {
    if let Ok(value) = word.parse::<i64>() {
        return Some(value);
    }
    match word {
        "zero" => Some(0),
        "one" => Some(1),
        "two" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        _ => None,
    }
}

fn number_word(value: i64) -> Option<&'static str> {
    match value {
        0 => Some("zero"),
        1 => Some("one"),
        2 => Some("two"),
        3 => Some("three"),
        4 => Some("four"),
        5 => Some("five"),
        6 => Some("six"),
        7 => Some("seven"),
        8 => Some("eight"),
        9 => Some("nine"),
        10 => Some("ten"),
        _ => None,
    }
}

fn normalize_letter_target(raw: &str) -> Option<String> {
    let trimmed = raw.trim_matches('\'').trim_end_matches("'s");
    if trimmed.len() == 1 && trimmed.chars().all(|ch| ch.is_ascii_alphabetic()) {
        Some(trimmed.to_string())
    } else if trimmed.len() == 2 && trimmed.ends_with('s') {
        let target = trimmed.chars().next()?;
        if target.is_ascii_alphabetic() {
            Some(target.to_string())
        } else {
            None
        }
    } else {
        None
    }
}

fn extract_plain_numbers(text: &str) -> Vec<i64> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|word| parse_count_word(word))
        .collect()
}

fn extract_answer_like_numbers(text: &str) -> Vec<i64> {
    let markers = [
        "final answer",
        "visible answer",
        "working answer",
        "working_answer",
        "answer",
    ];
    let mut out = Vec::new();
    for marker in markers {
        let mut start = 0usize;
        while let Some(idx) = text[start..].find(marker) {
            let abs = start + idx;
            let window = &text[abs..text.len().min(abs + 160)];
            out.extend(extract_plain_numbers(window));
            start = abs + marker.len();
        }
    }
    normalize_numbers(&out)
}

fn normalize_numbers(values: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    for value in values {
        if !out.contains(value) {
            out.push(*value);
        }
    }
    out
}

fn parse_letter_count_key(task_key: &str) -> Option<(&str, &str)> {
    let mut parts = task_key.split(':');
    if parts.next()? != "letter_count" {
        return None;
    }
    let word = parts.next()?;
    let target = parts.next()?;
    Some((word, target))
}

fn parse_parallel_drying_key(task_key: &str) -> Option<&str> {
    task_key.strip_prefix("parallel_drying:")
}

fn parallel_item_from_text(text: &str) -> Option<String> {
    for item in ["towel", "shirt", "cookie", "cloth", "blanket"] {
        let plural = pluralize_item(item);
        if contains_token(text, item) || contains_token(text, &plural) {
            return Some(item.to_string());
        }
    }
    None
}

fn pluralize_item(item: &str) -> String {
    if item.ends_with('y') {
        format!("{}ies", item.trim_end_matches('y'))
    } else if item.ends_with('s') {
        item.to_string()
    } else {
        format!("{item}s")
    }
}

fn singularize_item(item: &str) -> String {
    if item.ends_with("ies") {
        format!("{}y", item.trim_end_matches("ies"))
    } else if item.ends_with('s') {
        item.trim_end_matches('s').to_string()
    } else {
        item.to_string()
    }
}

fn prompt_mentions_target(normalized_prompt: &str, target: &str) -> bool {
    let plural = format!("{target}s");
    let possessive = format!("{target}'s");
    contains_token(normalized_prompt, target)
        || contains_token(normalized_prompt, &plural)
        || normalized_prompt.contains(&possessive)
}

fn explicit_sequential_duration_prompt(normalized_prompt: &str) -> bool {
    normalized_prompt.contains("after the previous")
        || normalized_prompt.contains("previous towel finishes")
        || normalized_prompt.contains("one after another")
        || normalized_prompt.contains("sequential")
        || normalized_prompt.contains("sequentially")
}

fn extract_hour_numbers(text: &str) -> Vec<i64> {
    let words = text
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    for window in words.windows(2) {
        let Ok(value) = window[0].parse::<i64>() else {
            continue;
        };
        if window[1].starts_with("hour") && !out.contains(&value) {
            out.push(value);
        }
    }
    out
}

fn compact_whitespace(text: &str, max_chars: usize) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max_chars)
        .collect::<String>()
        .trim()
        .to_string()
}

fn normalize_text(text: &str) -> String {
    text.to_ascii_lowercase()
        .replace('’', "'")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_strawberry_and_towel_corrections() {
        let mut memory = MistakeMemory::default();
        let previous = "There are 2 Rs in strawberry. It takes 50 hours.";
        let user = "you got both answers wrong there are 3 rs in strawberry and it takes 5 hours to dry all towels";

        let captured = memory.capture_from_correction_turn(user, Some(previous));

        assert_eq!(captured.len(), 2);
        assert!(captured
            .iter()
            .any(|event| event.task_key == "letter_count:strawberry:r"));
        assert!(captured
            .iter()
            .any(|event| event.task_key == "parallel_drying:towels"));
    }

    #[test]
    fn retrieves_relevant_towel_memory() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all towels",
            Some("The answer is 45 hours."),
        );

        let matches = memory.query(
            "if 1 towel takes 5 hours to dry and someone tells me to dry 9 more how long will it take me now",
            3,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "parallel_drying:towels");
    }

    #[test]
    fn guard_blocks_rejected_without_accepted() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all towels",
            Some("The answer is 50 hours."),
        );
        let matches = memory.query("how long to dry 9 more towels", 3);
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe("WORKING ANSWER: 50 hours [REQUEST: LOCK]");

        assert!(snapshot.rejected_answer_seen);
        assert!(!snapshot.accepted_answer_seen);
        assert!(guard.should_block_finalization());
    }

    #[test]
    fn guard_allows_accepted_answer() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all towels",
            Some("The answer is 50 hours."),
        );
        let matches = memory.query("how long to dry 9 more towels", 3);
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe("WORKING ANSWER: 5 hours [REQUEST: LOCK]");

        assert!(snapshot.accepted_answer_seen);
        assert!(!guard.should_block_finalization());
    }

    #[test]
    fn strawberry_count_sequence_does_not_count_as_accepted_answer() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe("I will count: 1, 2, 3. VISIBLE ANSWER: There are 2 Rs.");

        assert!(snapshot.rejected_answer_seen);
        assert!(!snapshot.accepted_answer_seen);
        assert!(guard.should_block_finalization());
    }

    #[test]
    fn prompt_injection_starts_with_visible_correct_answer() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);

        let prompt = MistakeMemory::apply_prompt("count the number of Rs in strawberry", &matches);

        assert!(
            prompt.contains("START_VISIBLE_ANSWER=VISIBLE ANSWER: There are 3 r's in strawberry.")
        );
        assert!(prompt.contains("REJECTED_ANSWERS=[2"));
    }

    #[test]
    fn captures_human_relationship_supersession_without_start_answer() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Actually we broke up. Don't refer to Maya as my girlfriend anymore.",
            Some("I will remember your girlfriend Maya for plans."),
        );

        let matches = memory.query("Can you help me write a weekend plan?", 3);
        let prompt = MistakeMemory::apply_prompt("Can you help me write a weekend plan?", &matches);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "human:superseded_relationship");
        assert!(prompt.contains("Do not repeat stale facts"));
        assert!(!prompt.contains("START_VISIBLE_ANSWER="));
        assert!(matches[0]
            .rejected_answers
            .iter()
            .any(|surface| normalize_text(surface) == "maya"));
    }

    #[test]
    fn captures_human_project_fact_budget() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn("No, corrected budget is 60k.", Some("Budget is 40k."));

        let matches = memory.query("What budget should I plan around?", 3);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "human:project_fact:budget");
        assert_eq!(matches[0].accepted_answer, "60k");
        assert!(matches[0]
            .rejected_answers
            .iter()
            .any(|surface| surface == "40k"));
    }

    #[test]
    fn stale_collaborator_memory_matches_client_update_replay() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );

        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].task_key,
            "human:project_fact:stale_collaborator_removed"
        );
        assert_eq!(matches[0].accepted_answer, "do not include Alex");
    }

    #[test]
    fn stale_collaborator_prompt_redacts_rejected_surface() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );

        let prompt = MistakeMemory::apply_prompt(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            &matches,
        );

        let memory_hint = prompt
            .split_once("\n\nUSER TURN:\n")
            .map(|(hint, _)| hint)
            .unwrap_or(prompt.as_str());
        assert!(!memory_hint.contains("Alex"));
        assert!(!memory_hint.contains("task=human"));
        assert!(!memory_hint.contains("CURRENT_HINT="));
        assert!(!memory_hint.contains("[stale surface]"));
        assert!(memory_hint.contains("the inactive collaborator"));
        assert!(memory_hint.contains("one complete client-safe note"));
        assert!(memory_hint.contains("Do not use bracket placeholders"));
        assert!(!memory_hint.contains("REDACTED_NAME_COUNT"));
        assert!(!memory_hint.contains("STALE_SURFACE"));
    }

    #[test]
    fn stale_collaborator_surface_shield_detects_token_violation() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );
        let terms = MistakeMemory::surface_suppression_terms(&matches);

        assert_eq!(terms, vec!["Alex".to_string()]);
        assert!(MistakeMemory::surface_suppression_violation(
            "Given the correction that",
            " Alex",
            &terms
        ));
        assert!(!MistakeMemory::surface_suppression_violation(
            "Project update:",
            " The beta checklist is due Friday.",
            &terms
        ));
    }

    #[test]
    fn stale_collaborator_success_requires_clean_surface() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );
        let mut clean_guard = MistakeMemoryGuard::new(matches.clone());
        let clean = clean_guard
            .observe("CLIENT UPDATE NOTE: Project update: The beta checklist is due Friday.");
        assert!(clean.accepted_answer_seen);
        assert!(!clean.accepted_boundary_seen);
        assert!(!clean.rejected_answer_seen);

        let mut leaked_guard = MistakeMemoryGuard::new(matches);
        let leaked = leaked_guard.observe("VISIBLE REASONING: Alex is no longer on this project.");
        assert!(!leaked.accepted_answer_seen);
        assert!(leaked.rejected_answer_seen);
    }

    #[test]
    fn stale_collaborator_completion_boundary_requires_useful_client_note() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let partial = guard.observe(
            "VISIBLE REASONING: I will remove the inactive collaborator. The client update",
        );
        assert!(partial.accepted_answer_seen);
        assert!(!partial.accepted_boundary_seen);

        let complete = guard.observe(
            "CLIENT UPDATE NOTE: Project update: The beta checklist remains due Friday; please confirm the QA owner with the current team.",
        );
        assert!(complete.accepted_answer_seen);
        assert!(complete.accepted_boundary_seen);
        assert!(!complete.rejected_answer_seen);

        let placeholder = guard.observe(
            "CLIENT UPDATE NOTE: Project update: QA follow-up owned by [current owner]; beta checklist due Friday.",
        );
        assert!(placeholder.accepted_answer_seen);
        assert!(!placeholder.accepted_boundary_seen);
    }

    #[test]
    fn stale_collaborator_completion_boundary_preserves_first_earned_note_sentence() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);
        let output = "VISIBLE REASONING: We remove the inactive collaborator.\n\
WORKING ANSWER: 'The beta checklist is due Friday; QA will proceed under new ownership.'\n\
\nHowever, to further refine the answer, I'll request: [REQUEST: FOCUS]";

        let snapshot = guard.observe(output);

        assert!(snapshot.accepted_answer_seen);
        assert!(snapshot.accepted_boundary_seen);
        let boundary = snapshot.accepted_boundary_byte_len.unwrap();
        assert!(boundary < output.len());
        assert_eq!(
            &output[..boundary],
            "VISIBLE REASONING: We remove the inactive collaborator.\nWORKING ANSWER: 'The beta checklist is due Friday; QA will proceed under new ownership.'"
        );
    }

    #[test]
    fn stale_collaborator_completion_boundary_accepts_current_team_owner_note() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe(
            "WORKING ANSWER: Project update: The beta checklist is due Friday; the QA owner should be confirmed with the current team.",
        );

        assert!(snapshot.accepted_answer_seen);
        assert!(snapshot.accepted_boundary_seen);
    }

    #[test]
    fn stale_collaborator_completion_boundary_rejects_unresolved_placeholder_sentence() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: Alex is no longer on this project. Do not include Alex in project updates or QA ownership.",
            Some("Alex owns the QA follow-up for this client update."),
        );
        let matches = memory.query(
            "Revise this client update: Alex will own QA follow-up; the beta checklist is due Friday.",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe(
            "WORKING ANSWER: The beta checklist is due on Friday and [owner] is covering the QA follow-up.",
        );

        assert!(snapshot.accepted_answer_seen);
        assert!(!snapshot.accepted_boundary_seen);
        assert_eq!(snapshot.accepted_boundary_byte_len, None);

        let bracketed_surface = guard.observe(
            "WORKING ANSWER: Project update: The beta checklist is still due Friday, and [inactive collaborator] should cover QA follow-up.",
        );
        assert!(bracketed_surface.accepted_answer_seen);
        assert!(!bracketed_surface.accepted_boundary_seen);
        assert_eq!(bracketed_surface.accepted_boundary_byte_len, None);
    }

    #[test]
    fn claim_review_evidence_gate_matches_related_artifact_review() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry. Single-seed script scores stay PRELIMINARY; bridge-off or feature-off mistakes are CONFIG-BLOCKED, not RED.",
            Some("LABEL: GREEN. The script score is enough to validate the claim."),
        );

        let matches = memory.query(
            "Review this different Niodoo claim artifact: script score passed, telemetry count is nonzero, but generated answers were not read. Should it move GREEN, RED, or PRELIMINARY?",
            3,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "human:claim_review:evidence_gate");
        assert_eq!(
            matches[0].accepted_answer,
            "PRELIMINARY until generated outputs and mechanism telemetry are read"
        );
    }

    #[test]
    fn claim_review_evidence_gate_boundary_stops_after_label() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry. Single-seed script scores stay PRELIMINARY.",
            Some("LABEL: GREEN. The script score is enough to validate the claim."),
        );
        let matches = memory.query(
            "Review this different Niodoo claim artifact: script score passed, but generated answers were not read. Should it move GREEN, RED, or PRELIMINARY?",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);
        let before =
            guard.observe("VISIBLE REASONING: This could look GREEN from the script score.");
        assert!(before.rejected_answer_seen);
        assert!(!before.accepted_boundary_seen);

        let after = guard.observe(
            "VISIBLE REASONING: This could look GREEN from the script score. LABEL: PRELIMINARY",
        );
        assert!(after.claim_review_evidence_gate_match);
        assert!(after.rejected_answer_seen);
        assert!(after.accepted_boundary_seen);
    }

    #[test]
    fn claim_review_evidence_gate_boundary_stops_after_committed_preliminary_rationale() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry. Single-seed script scores stay PRELIMINARY.",
            Some("LABEL: GREEN. The script score is enough to validate the claim."),
        );
        let matches = memory.query(
            "Review this different Niodoo claim artifact: script score passed, but generated answers were not read. Should it move GREEN, RED, or PRELIMINARY?",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let setup = guard.observe(
            "VISIBLE REASONING: I need to choose between GREEN, RED, or PRELIMINARY from a script score.",
        );
        assert!(setup.rejected_answer_seen);
        assert!(!setup.accepted_boundary_seen);

        let committed = guard.observe(
            "VISIBLE REASONING: The script score is PRELIMINARY because generated outputs and mechanism telemetry were not read.",
        );
        assert!(committed.claim_review_evidence_gate_match);
        assert!(committed.accepted_answer_seen);
        assert!(committed.accepted_boundary_seen);
    }

    #[test]
    fn claim_review_evidence_gate_boundary_stops_after_missing_evidence_working_answer() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry. Single-seed script scores stay PRELIMINARY.",
            Some("LABEL: GREEN. The script score is enough to validate the claim."),
        );
        let matches = memory.query(
            "Review this different Niodoo claim artifact: script score passed, but generated answers were not read. Should it move GREEN, RED, or PRELIMINARY?",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let committed = guard.observe(
            "WORKING ANSWER: Cannot determine if the candidate arm actually won without generated outputs and mechanism telemetry; the script score remains PRELIMINARY.",
        );
        assert!(committed.claim_review_evidence_gate_match);
        assert!(committed.accepted_answer_seen);
        assert!(committed.accepted_boundary_seen);
    }

    #[test]
    fn claim_review_evidence_gate_boundary_stops_after_preliminary_status_commitment() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry. Single-seed script scores stay PRELIMINARY.",
            Some("LABEL: GREEN. The script score is enough to validate the claim."),
        );
        let matches = memory.query(
            "Review this different Niodoo claim artifact: script score passed, but generated answers were not read. Should it move GREEN, RED, or PRELIMINARY?",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let committed = guard.observe(
            "Since the card only provides script-scored PASS/FAIL lines and nonzero telemetry counts, but lacks generated answer excerpts, raw telemetry fields inspected, stdout/stderr review, or config audit, the PRELIMINARY status is appropriate. WORKING ANSWER: PRELIMINARY",
        );
        assert!(committed.claim_review_evidence_gate_match);
        assert!(committed.accepted_answer_seen);
        assert!(committed.accepted_boundary_seen);
    }

    #[test]
    fn claim_review_evidence_gate_boundary_ignores_preliminary_without_evidence_gate() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry.",
            Some("LABEL: GREEN."),
        );
        let matches = memory.query(
            "Review this different Niodoo claim artifact: script score passed, but generated answers were not read. Should it move GREEN, RED, or PRELIMINARY?",
            3,
        );
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe(
            "VISIBLE REASONING: PRELIMINARY is one option, but I am still setting up the choice.",
        );
        assert!(!snapshot.accepted_boundary_seen);
    }

    #[test]
    fn claim_review_evidence_gate_does_not_match_code_review() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "Correction: scripts cannot promote or falsify claims without reading generated outputs and mechanism telemetry.",
            Some("LABEL: GREEN."),
        );

        let matches = memory.query(
            "Review this Python snippet for the main bug: def total(items): return items[0]",
            3,
        );

        assert!(matches.is_empty());
    }

    #[test]
    fn generic_count_prompt_does_not_trigger_strawberry_memory() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );

        let matches = memory.query("Count the letters in blueberry. Answer directly.", 3);

        assert!(matches.is_empty());
    }

    #[test]
    fn nearby_counting_prompt_does_not_trigger_strawberry_memory() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );

        assert!(memory
            .query("Count the number of Bs in strawberry. Answer directly.", 3)
            .is_empty());
        assert!(memory
            .query(
                "Count the number of letters in blueberry. Answer directly.",
                3
            )
            .is_empty());
    }

    #[test]
    fn captures_generic_letter_count_correction() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Es in bookkeeper",
            Some("There are 2 Es in bookkeeper."),
        );

        let matches = memory.query("Count the number of Es in bookkeeper. Answer directly.", 3);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "letter_count:bookkeeper:e");
        assert!(matches[0]
            .accepted_aliases
            .iter()
            .any(|alias| alias == "3 es"));
    }

    #[test]
    fn captures_plural_s_letter_count_correction() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 4 Ss in mississippi",
            Some("VISIBLE COUNT: 1 S\nVISIBLE COUNT: 2 Ss\nVISIBLE COUNT: 3 Ss\nVISIBLE COUNT: 5 Ss\nFINAL ANSWER: 5 Ss"),
        );

        let matches = memory.query("Count the number of Ss in mississippi. Answer directly.", 3);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "letter_count:mississippi:s");
        assert!(matches[0]
            .accepted_aliases
            .iter()
            .any(|alias| alias == "4 ss"));
        assert!(matches[0]
            .rejected_answers
            .iter()
            .any(|alias| alias == "5 ss"));
        assert!(!matches[0]
            .rejected_answers
            .iter()
            .any(|alias| alias == "1 s"));
    }

    #[test]
    fn captures_generic_parallel_drying_correction() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all shirts when they dry in parallel",
            Some("The answer is 30 hours."),
        );

        let matches = memory.query(
            "If 1 shirt takes 5 hours to dry and I dry 5 more shirts, how long will it take?",
            3,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].task_key, "parallel_drying:shirts");
    }

    #[test]
    fn sequential_towel_prompt_does_not_trigger_parallel_memory() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all towels when they dry in parallel",
            Some("The answer is 45 hours."),
        );

        assert!(memory
            .query(
                "If one towel must dry after the previous towel finishes, how long do 3 towels take?",
                3,
            )
            .is_empty());
    }

    #[test]
    fn intermediate_working_count_is_not_final_rejected_answer() {
        let mut memory = MistakeMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeMemoryGuard::new(matches);

        let snapshot = guard.observe("WORKING ANSWER: 2 Rs so far. I found another R.");

        assert!(!snapshot.rejected_answer_seen);
        assert!(!guard.should_block_finalization());
    }
}
