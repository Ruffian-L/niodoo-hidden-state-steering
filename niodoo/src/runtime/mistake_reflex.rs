use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReflexTaskSchema {
    SymbolicCounting {
        word: String,
        target_char: String,
        expected_count: i64,
    },
    ParallelDuration {
        item: String,
        single_duration: i64,
        unit: String,
        total_items: Option<i64>,
        capacity_mode: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MistakeReflexEvent {
    pub id: String,
    pub domain: String,
    pub trigger_terms: Vec<String>,
    pub bad_reflex: String,
    pub corrected_reflex: String,
    pub evidence_requirement: String,
    pub rejected_surfaces: Vec<String>,
    #[serde(default)]
    pub accepted_surfaces: Vec<String>,
    #[serde(default)]
    pub schema: Option<ReflexTaskSchema>,
    pub allowed_actions: Vec<String>,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default = "default_action_level")]
    pub action_level: u8,
    #[serde(default = "default_decay_rate")]
    pub decay_rate: f32,
    #[serde(default)]
    pub success_count: u32,
    #[serde(default)]
    pub repeat_mistake_count: u32,
    #[serde(default)]
    pub episodic_correction: Option<String>,
    #[serde(default)]
    pub example_anchor: Option<String>,
    #[serde(default)]
    pub procedural_rule: Option<String>,
    #[serde(default)]
    pub evidence_gate: Option<String>,
    #[serde(default)]
    pub symbolic_key: Option<String>,
    #[serde(default)]
    pub hidden_full_path: Option<String>,
    #[serde(default)]
    pub hidden_dim: Option<usize>,
    #[serde(default)]
    pub route_64d: Option<Vec<f32>>,
    #[serde(default)]
    pub route_motif_id: Option<String>,
    #[serde(default)]
    pub unicode_packet_id: Option<String>,
    #[serde(default)]
    pub unicode_escape: Option<String>,
    #[serde(default)]
    pub decoded_route_id: Option<String>,
    #[serde(default)]
    pub route_preserved: Option<bool>,
    #[serde(default = "default_resolution_level")]
    pub current_resolution_level: u8,
    #[serde(default = "default_resolution_level")]
    pub last_required_resolution_level: u8,
    #[serde(default)]
    pub success_streak: u32,
    #[serde(default)]
    pub failure_count: u32,
    #[serde(default)]
    pub false_positive_count: u32,
    pub created_at_ms: u128,
    pub updated_at_ms: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MistakeReflexMatch {
    pub event_id: String,
    pub domain: String,
    pub trigger_terms: Vec<String>,
    pub score: f32,
    pub corrected_reflex: String,
    pub evidence_requirement: String,
    pub rejected_surfaces: Vec<String>,
    pub accepted_surfaces: Vec<String>,
    pub schema: Option<ReflexTaskSchema>,
    pub allowed_actions: Vec<String>,
    pub confidence: f32,
    pub action_level: u8,
    pub current_resolution_level: u8,
    pub vector_slice_available: bool,
    pub unicode_packet_id: Option<String>,
    pub unicode_escape: Option<String>,
    pub route_motif_id: Option<String>,
    pub decoded_route_id: Option<String>,
    pub route_preserved: Option<bool>,
    pub procedural_rule: Option<String>,
    pub evidence_gate: Option<String>,
    pub symbolic_key: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MistakeReflexSnapshot {
    pub matched: bool,
    pub match_count: usize,
    pub event_ids: Vec<String>,
    pub domains: Vec<String>,
    pub action_level: u8,
    pub resolution_level: u8,
    pub vector_slice_available: bool,
    pub unicode_packet_ids: Vec<String>,
    pub route_preserved: Option<bool>,
    pub unfold_reason: Option<String>,
    pub decay_reason: Option<String>,
    pub evidence_seen: bool,
    pub accepted_answer_candidate_seen: bool,
    pub old_mistake_seen: bool,
    pub old_path_after_earned: bool,
    pub earned_answer_seen: bool,
    pub earned_answer_text: Option<String>,
    pub earned_boundary_step: Option<usize>,
    pub earned_boundary_byte_len: Option<usize>,
    pub blocked_lock: bool,
    pub blocked_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GmmsObserveOnlySummary {
    pub event_id: String,
    pub family_id: String,
    pub mode: String,
    pub score: f32,
    pub trigger_hits: Vec<String>,
    pub rejected_path_hit: bool,
    pub accepted_surface_count: usize,
    pub final_answer_injection_allowed: bool,
    pub allowed_action_max: String,
    pub action_level: u8,
    pub vector_slice_available: bool,
    pub route_unicode_sidecar_attached: bool,
    pub unicode_packet_id: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct MistakeReflexMemory {
    events: Vec<MistakeReflexEvent>,
}

#[derive(Debug, Clone)]
pub struct MistakeReflexGuard {
    matches: Vec<MistakeReflexMatch>,
    evidence_seen: bool,
    accepted_answer_candidate_seen: bool,
    old_mistake_seen: bool,
    old_path_after_earned: bool,
    earned_answer_seen: bool,
    earned_answer_text: Option<String>,
    earned_boundary_step: Option<usize>,
    earned_boundary_byte_len: Option<usize>,
    blocked_count: usize,
}

fn default_confidence() -> f32 {
    0.75
}

fn default_action_level() -> u8 {
    1
}

fn default_decay_rate() -> f32 {
    0.15
}

fn default_resolution_level() -> u8 {
    2
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

impl MistakeReflexMemory {
    // Expose the events so Qdrant can sync them
    pub fn events(&self) -> &Vec<MistakeReflexEvent> {
        &self.events
    }

    // Allow Qdrant to load the fetched memories into the live runtime
    pub fn replace_events(&mut self, new_events: Vec<MistakeReflexEvent>) {
        self.events = new_events;
    }

    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("Failed to read mistake reflex memory {}", path.display()))?;
        let mut events = Vec::new();
        for (idx, line) in raw.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut event: MistakeReflexEvent = serde_json::from_str(line).with_context(|| {
                format!(
                    "Failed to parse mistake reflex memory {} at line {}",
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
                    format!("Failed to create mistake reflex dir {}", parent.display())
                })?;
            }
        }
        let mut out = String::new();
        for event in &self.events {
            out.push_str(&serde_json::to_string(event)?);
            out.push('\n');
        }
        fs::write(path, out)
            .with_context(|| format!("Failed to write mistake reflex memory {}", path.display()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn observe_gmms_applicability(
        &self,
        user_prompt: &str,
        limit: usize,
    ) -> Vec<GmmsObserveOnlySummary> {
        if limit == 0 {
            return Vec::new();
        }
        let normalized_prompt = normalize_text(user_prompt);
        let prompt_terms = gmms_terms(user_prompt);
        let mut scored = Vec::new();
        for event in self
            .events
            .iter()
            .filter(|event| event.domain == "gmms:semantic_correction_slice")
        {
            let trigger_hits = event
                .trigger_terms
                .iter()
                .filter(|term| {
                    let normalized = normalize_text(term);
                    normalized_prompt.contains(&normalized)
                        || gmms_terms(term)
                            .iter()
                            .any(|term| prompt_terms.contains(term))
                })
                .cloned()
                .collect::<Vec<_>>();
            let rejected_path_hit = event.rejected_surfaces.iter().any(|surface| {
                contains_rejected_surface(&normalized_prompt, surface)
                    || gmms_terms(surface)
                        .iter()
                        .any(|term| prompt_terms.contains(term))
            });
            let family_terms = event
                .symbolic_key
                .as_deref()
                .map(gmms_terms)
                .unwrap_or_default();
            let family_hits = family_terms
                .iter()
                .filter(|term| prompt_terms.contains(*term))
                .count();
            let score = trigger_hits.len() as f32 * 0.20
                + family_hits as f32 * 0.05
                + if rejected_path_hit { 0.20 } else { 0.0 };
            if score <= 0.0 {
                continue;
            }
            let mode = gmms_mode_for_event(event);
            let final_answer_injection_allowed =
                mode != "skill_reflex" && !event.accepted_surfaces.is_empty();
            let route_unicode_sidecar_attached = event
                .unicode_packet_id
                .as_ref()
                .is_some_and(|packet| !packet.trim().is_empty());
            scored.push(GmmsObserveOnlySummary {
                event_id: event.id.clone(),
                family_id: event
                    .symbolic_key
                    .clone()
                    .unwrap_or_else(|| event.domain.clone()),
                mode,
                score,
                trigger_hits,
                rejected_path_hit,
                accepted_surface_count: event.accepted_surfaces.len(),
                final_answer_injection_allowed,
                allowed_action_max: gmms_allowed_action_max(event),
                action_level: event.action_level,
                vector_slice_available: event_has_usable_vector_slice(event),
                route_unicode_sidecar_attached,
                unicode_packet_id: event.unicode_packet_id.clone(),
            });
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.into_iter().take(limit).collect()
    }

    pub fn attach_vector_slices_from_packet_index(&mut self, path: &Path) -> Result<usize> {
        let text = fs::read_to_string(path).with_context(|| {
            format!(
                "Failed to read mistake reflex packet index {}",
                path.display()
            )
        })?;
        let mut attached = 0usize;
        for event in &mut self.events {
            if event.unicode_packet_id.is_some() {
                continue;
            }
            if let Some(slice) = find_packet_slice_for_event(&text, event)? {
                event.hidden_full_path = slice.hidden_full_path;
                event.hidden_dim = slice.hidden_dim;
                event.route_64d = slice.route_64d;
                event.route_motif_id = slice.route_motif_id;
                event.unicode_packet_id = slice.unicode_packet_id;
                event.unicode_escape = slice.unicode_escape;
                event.decoded_route_id = slice.decoded_route_id;
                event.route_preserved = slice.route_preserved;
                event.updated_at_ms = now_ms();
                attached += 1;
            }
        }
        Ok(attached)
    }

    pub fn capture_from_correction_turn(
        &mut self,
        user_prompt: &str,
        previous_assistant: Option<&str>,
    ) -> Vec<MistakeReflexEvent> {
        let Some(previous_assistant) = previous_assistant else {
            return Vec::new();
        };
        if !correction_like(user_prompt) {
            return Vec::new();
        }
        let mut captured = Vec::new();
        if let Some(event) = symbolic_counting_reflex(user_prompt, previous_assistant) {
            captured.push(self.upsert_event(event));
        }
        if let Some(event) = parallel_duration_reflex(user_prompt, previous_assistant) {
            captured.push(self.upsert_event(event));
        }
        captured
    }

    pub fn query(&self, user_prompt: &str, limit: usize) -> Vec<MistakeReflexMatch> {
        if limit == 0 {
            return Vec::new();
        }
        let prompt = normalize_text(user_prompt);
        let mut scored = Vec::new();
        for event in &self.events {
            let schema_override = prompt_schema_for_memory_slice(event, &prompt);
            let schema_accepted_override = schema_override
                .as_ref()
                .map(accepted_surfaces_for_prompt_schema)
                .unwrap_or_default();
            let prompt_process_accepted_override =
                accepted_surfaces_for_prompt_process_event(event, &prompt);
            let score = match_score_with_schema(event, &prompt, schema_override.as_ref());
            if score <= 0.0 {
                continue;
            }
            let accepted_surfaces = if !schema_accepted_override.is_empty() {
                schema_accepted_override
            } else if !prompt_process_accepted_override.is_empty() {
                prompt_process_accepted_override
            } else {
                event.accepted_surfaces.clone()
            };
            scored.push((
                score,
                MistakeReflexMatch {
                    event_id: event.id.clone(),
                    domain: event.domain.clone(),
                    trigger_terms: event.trigger_terms.clone(),
                    score,
                    corrected_reflex: event.corrected_reflex.clone(),
                    evidence_requirement: event.evidence_requirement.clone(),
                    rejected_surfaces: event.rejected_surfaces.clone(),
                    accepted_surfaces,
                    schema: schema_override.or_else(|| event.schema.clone()),
                    allowed_actions: event.allowed_actions.clone(),
                    confidence: event.confidence,
                    action_level: event.action_level,
                    current_resolution_level: event.current_resolution_level,
                    vector_slice_available: event_has_usable_vector_slice(event),
                    unicode_packet_id: event.unicode_packet_id.clone(),
                    unicode_escape: event.unicode_escape.clone(),
                    route_motif_id: event.route_motif_id.clone(),
                    decoded_route_id: event.decoded_route_id.clone(),
                    route_preserved: event.route_preserved,
                    procedural_rule: event.procedural_rule.clone(),
                    evidence_gate: event.evidence_gate.clone(),
                    symbolic_key: event.symbolic_key.clone(),
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

    pub fn apply_prompt(
        user_prompt: &str,
        matches: &[MistakeReflexMatch],
        action_mode: &str,
    ) -> String {
        if matches.is_empty() {
            return user_prompt.to_string();
        }
        if action_mode == "hidden-control" {
            return user_prompt.to_string();
        }
        let mut lines = vec![
            "MISTAKE REFLEX:".to_string(),
            "A prior correction matches this task region. The runtime is selecting the smallest active slice needed; do the move yourself.".to_string(),
            "For skill reflexes, do not use this note as a final answer.".to_string(),
        ];
        let expose_route_packet_surface = action_mode != "text-hint-hidden-packet";
        let normalized_user_prompt = normalize_text(user_prompt);
        for item in matches.iter().take(3) {
            lines.push(format!("- region={}", item.domain));
            if item.vector_slice_available && expose_route_packet_surface {
                lines.push(format!(
                    "  route_slice=available packet={} route={} preserved={}",
                    item.unicode_packet_id.as_deref().unwrap_or("-"),
                    item.route_motif_id.as_deref().unwrap_or("-"),
                    item.route_preserved
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                ));
            }
            if action_mode == "evidence-gate" || item.current_resolution_level <= 1 {
                lines.push(format!(
                    "  evidence_gate={}",
                    compact_whitespace(
                        item.evidence_gate
                            .as_deref()
                            .unwrap_or(&item.evidence_requirement),
                        180
                    )
                ));
                continue;
            }
            if item.current_resolution_level >= 2 {
                if gmms_should_redact_unmentioned_removed_entity(item, &normalized_user_prompt) {
                    lines.push(
                        "  procedure=do not introduce or assign removed collaborators from the matched correction; if the active owner is unclear, ask for the owner"
                            .to_string(),
                    );
                } else {
                    lines.push(format!(
                        "  procedure={}",
                        compact_whitespace(
                            item.procedural_rule
                                .as_deref()
                                .unwrap_or(&item.corrected_reflex),
                            180
                        )
                    ));
                }
            }
            if gmms_should_redact_unmentioned_removed_entity(item, &normalized_user_prompt) {
                lines.push(
                    "  evidence_required=does not assign work to a removed collaborator; asks for the active owner if uncertain; does not name removed people unless the user did"
                        .to_string(),
                );
            } else {
                lines.push(format!(
                    "  evidence_required={}",
                    compact_whitespace(&item.evidence_requirement, 180)
                ));
            }
            if is_verification_policy_process_match(item) {
                lines.push(
                    "  claim_decision_guard=missing evidence blocks GREEN but does not prove RED/FALSIFIED; keep the claim unpromoted or reject GREEN unless raw per-token JSONL/generated-output telemetry is reviewed"
                        .to_string(),
                );
                lines.push(
                    "  answer_boundary=one line should name the no-GREEN decision and the raw per-token JSONL/generated-output evidence requirement"
                        .to_string(),
                );
            }
            if item.domain == "symbolic_counting:letter_count" {
                lines.push(
                    "  trap_note=scan from the first character; do not only count the final adjacent pair"
                        .to_string(),
                );
                if let Some(ReflexTaskSchema::SymbolicCounting {
                    word, target_char, ..
                }) = &item.schema
                {
                    lines.push(format!(
                        "  task_binding=word `{word}`; target letter `{target_char}`; start count at 0"
                    ));
                    let example = symbolic_scan_prefix_example(word, target_char);
                    if !example.is_empty() {
                        lines.push(format!(
                            "  scan_entry_example={example}; continue across every remaining character yourself"
                        ));
                    }
                }
                lines.push(
                    "  evidence_format=use one short line only: VISIBLE SCAN: <actual-letter>=<running-total> | <actual-letter>=<running-total> ...; never write placeholder words like char or running_total"
                        .to_string(),
                );
                lines.push(
                    "  boundary=after the scan, write exactly one ANSWER line as: ANSWER: <number> <target-letter>s; do not use letter=number; do not explain the procedure"
                        .to_string(),
                );
            } else if item.domain == "parallel_duration:drying" {
                lines.push(
                    "  trap_note=if enough rack/line/space is present, classify as parallel drying, not sequential drying"
                        .to_string(),
                );
                lines.push(
                    "  evidence_format=state PARALLEL CHECK before the answer; when capacity is enough, answer immediately with the single-item duration and unit"
                        .to_string(),
                );
            }
        }
        format!("{}\n\nUSER TURN:\n{}", lines.join("\n"), user_prompt)
    }

    pub fn record_outcome(
        &mut self,
        matches: &[MistakeReflexMatch],
        snapshot: &MistakeReflexSnapshot,
    ) -> bool {
        if matches.is_empty() {
            return false;
        }
        let mut changed = false;
        for item in matches {
            let Some(event) = self
                .events
                .iter_mut()
                .find(|event| event.id == item.event_id)
            else {
                continue;
            };
            if snapshot.old_mistake_seen {
                event.repeat_mistake_count = event.repeat_mistake_count.saturating_add(1);
                event.failure_count = event.failure_count.saturating_add(1);
                event.success_streak = 0;
                event.confidence = (event.confidence + 0.10).min(1.75);
                event.action_level = event.action_level.saturating_add(1).min(3);
                event.last_required_resolution_level = event.current_resolution_level;
                event.current_resolution_level =
                    event.current_resolution_level.saturating_add(1).min(5);
                event.updated_at_ms = now_ms();
                changed = true;
            } else if snapshot.evidence_seen {
                event.success_count = event.success_count.saturating_add(1);
                event.success_streak = event.success_streak.saturating_add(1);
                event.confidence = (event.confidence * (1.0 - event.decay_rate * 0.25)).max(0.35);
                event.action_level = event.action_level.saturating_sub(1).max(1);
                if event.success_streak >= 2 {
                    event.last_required_resolution_level = event.current_resolution_level;
                    event.current_resolution_level =
                        event.current_resolution_level.saturating_sub(1);
                    event.success_streak = 0;
                }
                event.updated_at_ms = now_ms();
                changed = true;
            }
        }
        changed
    }

    fn upsert_event(&mut self, mut event: MistakeReflexEvent) -> MistakeReflexEvent {
        normalize_event(&mut event);
        if let Some(existing) = self
            .events
            .iter_mut()
            .find(|existing| existing.id == event.id)
        {
            existing.trigger_terms =
                merge_strings(&existing.trigger_terms, &event.trigger_terms, 24);
            existing.rejected_surfaces =
                merge_strings(&existing.rejected_surfaces, &event.rejected_surfaces, 24);
            existing.accepted_surfaces =
                merge_strings(&existing.accepted_surfaces, &event.accepted_surfaces, 24);
            existing.corrected_reflex = event.corrected_reflex.clone();
            existing.evidence_requirement = event.evidence_requirement.clone();
            existing.episodic_correction = event.episodic_correction.clone();
            existing.example_anchor = event.example_anchor.clone();
            existing.procedural_rule = event.procedural_rule.clone();
            existing.evidence_gate = event.evidence_gate.clone();
            existing.symbolic_key = event.symbolic_key.clone();
            existing.confidence = (existing.confidence + 0.15).min(1.75);
            existing.action_level = existing.action_level.saturating_add(1).min(3);
            existing.last_required_resolution_level = existing.current_resolution_level;
            existing.current_resolution_level =
                existing.current_resolution_level.saturating_add(1).min(5);
            existing.updated_at_ms = now_ms();
            return existing.clone();
        }
        self.events.push(event.clone());
        event
    }
}

fn gmms_should_redact_unmentioned_removed_entity(
    item: &MistakeReflexMatch,
    normalized_user_prompt: &str,
) -> bool {
    if item.domain != "gmms:semantic_correction_slice"
        || !item
            .symbolic_key
            .as_deref()
            .is_some_and(|key| key.contains("entity_membership"))
    {
        return false;
    }
    let stale_text = normalize_text(&format!(
        "{} {} {}",
        item.corrected_reflex,
        item.evidence_requirement,
        item.rejected_surfaces.join(" ")
    ));
    item.trigger_terms
        .iter()
        .flat_map(|term| gmms_informative_terms(term))
        .filter(|term| stale_text.contains(term))
        .filter(|term| {
            !matches!(
                term.as_str(),
                "assign"
                    | "assignment"
                    | "assignee"
                    | "collaborator"
                    | "owner"
                    | "project"
                    | "qa"
                    | "task"
                    | "stale"
                    | "removal"
                    | "removed"
            )
        })
        .all(|term| !normalized_user_prompt.contains(&term))
}

impl MistakeReflexGuard {
    pub fn new(matches: Vec<MistakeReflexMatch>) -> Self {
        Self {
            matches,
            evidence_seen: false,
            accepted_answer_candidate_seen: false,
            old_mistake_seen: false,
            old_path_after_earned: false,
            earned_answer_seen: false,
            earned_answer_text: None,
            earned_boundary_step: None,
            earned_boundary_byte_len: None,
            blocked_count: 0,
        }
    }

    pub fn observe(&mut self, step: usize, assistant_text: &str) -> MistakeReflexSnapshot {
        let raw_old_path_seen = self
            .matches
            .iter()
            .any(|item| old_path_seen(item, assistant_text));
        self.evidence_seen = self
            .matches
            .iter()
            .any(|item| evidence_seen(item, assistant_text));
        self.accepted_answer_candidate_seen = self
            .matches
            .iter()
            .any(|item| accepted_answer_candidate_seen(item, assistant_text));
        self.earned_answer_text = self
            .matches
            .iter()
            .find_map(|item| earned_answer_seen(item, assistant_text));
        self.earned_answer_seen = self.earned_answer_text.is_some();
        if self.earned_answer_seen && self.earned_boundary_step.is_none() {
            self.earned_boundary_step = Some(step);
            self.earned_boundary_byte_len = Some(assistant_text.len());
        }
        self.old_path_after_earned = self
            .earned_boundary_byte_len
            .and_then(|byte_len| assistant_text.get(byte_len..))
            .map(|tail| self.matches.iter().any(|item| old_path_seen(item, tail)))
            .unwrap_or(false);
        self.old_mistake_seen = raw_old_path_seen && self.earned_boundary_step.is_none();
        self.snapshot()
    }

    pub fn should_block_finalization(&self) -> bool {
        if self.matches.is_empty() {
            return false;
        }
        let max_action = self
            .matches
            .iter()
            .map(|item| item.action_level)
            .max()
            .unwrap_or_default();
        self.old_mistake_seen || (max_action >= 2 && !self.evidence_seen)
    }

    pub fn record_blocked_lock(&mut self) {
        self.blocked_count = self.blocked_count.saturating_add(1);
    }

    pub fn snapshot(&self) -> MistakeReflexSnapshot {
        MistakeReflexSnapshot {
            matched: !self.matches.is_empty(),
            match_count: self.matches.len(),
            event_ids: self
                .matches
                .iter()
                .map(|item| item.event_id.clone())
                .collect(),
            domains: self
                .matches
                .iter()
                .map(|item| item.domain.clone())
                .collect(),
            action_level: self
                .matches
                .iter()
                .map(|item| item.action_level)
                .max()
                .unwrap_or_default(),
            resolution_level: self
                .matches
                .iter()
                .map(|item| item.current_resolution_level)
                .max()
                .unwrap_or_default(),
            vector_slice_available: self.matches.iter().any(|item| item.vector_slice_available),
            unicode_packet_ids: self
                .matches
                .iter()
                .filter_map(|item| item.unicode_packet_id.clone())
                .collect(),
            route_preserved: route_preserved_summary(&self.matches),
            unfold_reason: if self.old_mistake_seen {
                Some("old_mistake_seen".to_string())
            } else {
                None
            },
            decay_reason: if self.evidence_seen && !self.old_mistake_seen {
                Some("evidence_seen".to_string())
            } else {
                None
            },
            evidence_seen: self.evidence_seen,
            accepted_answer_candidate_seen: self.accepted_answer_candidate_seen,
            old_mistake_seen: self.old_mistake_seen,
            old_path_after_earned: self.old_path_after_earned,
            earned_answer_seen: self.earned_answer_seen,
            earned_answer_text: self.earned_answer_text.clone(),
            earned_boundary_step: self.earned_boundary_step,
            earned_boundary_byte_len: self.earned_boundary_byte_len,
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
        "no,",
        "no ",
        "not ",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn symbolic_counting_reflex(
    user_prompt: &str,
    previous_assistant: &str,
) -> Option<MistakeReflexEvent> {
    let user = normalize_text(user_prompt);
    let Some((word, target, accepted_count)) = extract_letter_count_correction(&user) else {
        return None;
    };
    let previous = normalize_text(previous_assistant);
    let mut rejected = Vec::new();
    for count in extract_answer_like_numbers(&previous) {
        if count != accepted_count {
            rejected.push(format!("{count} {target}"));
            rejected.push(format!("{count} {target}s"));
            rejected.push(format!("{count} {target}'s"));
        }
    }
    if rejected.is_empty() {
        rejected.push(format!("2 {target}"));
    }
    let target_plural = format!("{target}s");
    let target_possessive = format!("{target}'s");
    let accepted_count_word = number_word(accepted_count).unwrap_or("");
    let accepted_surfaces = vec![
        format!("{accepted_count} {target_plural}"),
        format!("{accepted_count} {target_possessive}"),
        format!("{accepted_count} {target}"),
        accepted_count.to_string(),
        format!("{accepted_count_word} {target_plural}"),
        format!("there are {accepted_count}"),
    ];
    let trigger_terms = [
        "count",
        "letter",
        "letters",
        word.as_str(),
        target.as_str(),
        target_plural.as_str(),
    ];
    Some(event(
        "symbolic_counting:letter_count",
        &trigger_terms,
        "guess from word familiarity or stop after the first visible pair",
        "initialize count=0; scan each character once; if the character is the target, add 1; if it is not the target, leave count unchanged; only answer after the running count line",
        "visible running count evidence before numeric LOCK",
        rejected,
            accepted_surfaces
            .into_iter()
            .filter(|surface| !surface.trim().is_empty())
            .collect(),
        Some(ReflexTaskSchema::SymbolicCounting {
            word: word.clone(),
            target_char: target.clone(),
            expected_count: accepted_count,
        }),
        &[
            "inject_short_reflex_hint",
            "require_evidence_before_lock",
            "suppress_early_lock",
        ],
    ))
}

fn parallel_duration_reflex(
    user_prompt: &str,
    previous_assistant: &str,
) -> Option<MistakeReflexEvent> {
    let user = normalize_text(user_prompt);
    if !user.contains("dry") {
        return None;
    }
    let single_duration = extract_hour_numbers(&user).first().copied().unwrap_or(5);
    let item = parallel_item_from_text(&user)?;
    let total_items = extract_parallel_total_items(&user);
    let previous = normalize_text(previous_assistant);
    let mut rejected = Vec::new();
    for hours in extract_hour_numbers(&previous) {
        if hours != single_duration {
            rejected.push(format!("{hours} hours"));
            rejected.push(hours.to_string());
        }
    }
    if rejected.is_empty() {
        if let Some(total_items) = total_items {
            let multiplied = single_duration.saturating_mul(total_items);
            rejected.push(format!("{multiplied} hours"));
            rejected.push(multiplied.to_string());
        }
    }
    let item_plural = pluralize_item(&item);
    let trigger_terms = [
        item.as_str(),
        item_plural.as_str(),
        "dry",
        "drying",
        "hours",
        "more",
    ];
    Some(event(
        "parallel_duration:drying",
        &trigger_terms,
        "multiply elapsed duration by item count without checking parallel capacity",
        "first classify parallel vs sequential; if enough capacity allows parallel drying, elapsed time stays the single-item duration; multiply only when the prompt says sequential",
        "explicit parallel/sequential check before LOCK",
        rejected,
        vec![
            format!("{single_duration} hours"),
            format!("{single_duration} hour"),
            number_word(single_duration)
                .map(|word| format!("{word} hours"))
                .unwrap_or_default(),
        ],
        Some(ReflexTaskSchema::ParallelDuration {
            item: item.clone(),
            single_duration,
            unit: "hours".to_string(),
            total_items,
            capacity_mode: "parallel_if_capacity_present".to_string(),
        }),
        &[
            "inject_short_reflex_hint",
            "require_evidence_before_lock",
            "suppress_early_lock",
        ],
    ))
}

fn event(
    domain: &str,
    trigger_terms: &[&str],
    bad_reflex: &str,
    corrected_reflex: &str,
    evidence_requirement: &str,
    rejected_surfaces: Vec<String>,
    accepted_surfaces: Vec<String>,
    schema: Option<ReflexTaskSchema>,
    allowed_actions: &[&str],
) -> MistakeReflexEvent {
    let created_at_ms = now_ms();
    MistakeReflexEvent {
        id: stable_event_id(domain, trigger_terms),
        domain: domain.to_string(),
        trigger_terms: trigger_terms.iter().map(|term| term.to_string()).collect(),
        bad_reflex: bad_reflex.to_string(),
        corrected_reflex: corrected_reflex.to_string(),
        evidence_requirement: evidence_requirement.to_string(),
        rejected_surfaces,
        accepted_surfaces,
        schema,
        allowed_actions: allowed_actions
            .iter()
            .map(|action| action.to_string())
            .collect(),
        confidence: 0.85,
        action_level: 2,
        decay_rate: 0.15,
        success_count: 0,
        repeat_mistake_count: 0,
        episodic_correction: None,
        example_anchor: None,
        procedural_rule: Some(corrected_reflex.to_string()),
        evidence_gate: Some(evidence_requirement.to_string()),
        symbolic_key: Some(domain.to_string()),
        hidden_full_path: None,
        hidden_dim: None,
        route_64d: None,
        route_motif_id: None,
        unicode_packet_id: None,
        unicode_escape: None,
        decoded_route_id: None,
        route_preserved: None,
        current_resolution_level: 2,
        last_required_resolution_level: 2,
        success_streak: 0,
        failure_count: 0,
        false_positive_count: 0,
        created_at_ms,
        updated_at_ms: created_at_ms,
    }
}

fn stable_event_id(domain: &str, trigger_terms: &[&str]) -> String {
    let mut hasher = DefaultHasher::new();
    domain.hash(&mut hasher);
    for term in trigger_terms {
        term.hash(&mut hasher);
    }
    format!("mistake_reflex:{}:{:016x}", domain, hasher.finish())
}

fn normalize_event(event: &mut MistakeReflexEvent) {
    event.trigger_terms = normalize_list(&event.trigger_terms, 24);
    event.rejected_surfaces = normalize_list(&event.rejected_surfaces, 24);
    event.accepted_surfaces = normalize_list(&event.accepted_surfaces, 24);
    event.allowed_actions = normalize_list(&event.allowed_actions, 16);
    if event.confidence <= 0.0 {
        event.confidence = default_confidence();
    }
    if event.action_level == 0 {
        event.action_level = default_action_level();
    }
    if event.decay_rate <= 0.0 {
        event.decay_rate = default_decay_rate();
    }
    if event.current_resolution_level > 5 {
        event.current_resolution_level = 5;
    }
    if event.last_required_resolution_level > 5 {
        event.last_required_resolution_level = 5;
    }
    if event.procedural_rule.is_none() && !event.corrected_reflex.is_empty() {
        event.procedural_rule = Some(event.corrected_reflex.clone());
    }
    if event.evidence_gate.is_none() && !event.evidence_requirement.is_empty() {
        event.evidence_gate = Some(event.evidence_requirement.clone());
    }
    if event.symbolic_key.is_none() {
        event.symbolic_key = Some(event.domain.clone());
    }
    if event.schema.is_none() {
        event.schema = derive_schema_from_event(event);
    }
    if let Some(route_64d) = &mut event.route_64d {
        if route_64d.len() != 64 {
            event.route_64d = None;
        }
    }
}

fn event_has_usable_vector_slice(event: &MistakeReflexEvent) -> bool {
    if event
        .route_64d
        .as_ref()
        .is_some_and(|route| route.len() == 64)
    {
        return true;
    }
    event
        .unicode_packet_id
        .as_ref()
        .is_some_and(|value| !value.is_empty())
        && event
            .unicode_escape
            .as_ref()
            .is_some_and(|value| !value.is_empty())
        && event
            .route_motif_id
            .as_ref()
            .is_some_and(|value| !value.is_empty())
        && event
            .decoded_route_id
            .as_ref()
            .is_some_and(|value| !value.is_empty())
        && event.route_preserved == Some(true)
}

fn derive_schema_from_event(event: &MistakeReflexEvent) -> Option<ReflexTaskSchema> {
    match event.domain.as_str() {
        "symbolic_counting:letter_count" => {
            let (word, target_char) = event_word_and_target(event)?;
            let expected_count = event
                .accepted_surfaces
                .iter()
                .filter_map(|surface| {
                    normalize_text(surface)
                        .split_whitespace()
                        .next()
                        .and_then(parse_count_word)
                })
                .next()?;
            Some(ReflexTaskSchema::SymbolicCounting {
                word,
                target_char,
                expected_count,
            })
        }
        "parallel_duration:drying" => {
            let item = event_parallel_item(event)?;
            let single_duration = event
                .accepted_surfaces
                .iter()
                .find_map(|surface| {
                    extract_hour_numbers(&normalize_text(surface))
                        .first()
                        .copied()
                })
                .unwrap_or(5);
            Some(ReflexTaskSchema::ParallelDuration {
                item,
                single_duration,
                unit: "hours".to_string(),
                total_items: None,
                capacity_mode: "parallel_if_capacity_present".to_string(),
            })
        }
        _ => None,
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

fn gmms_terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for term in normalize_text(text)
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
    {
        if term.len() < 2 {
            continue;
        }
        let term = term.to_string();
        if !out.contains(&term) {
            out.push(term);
        }
    }
    out
}

fn gmms_informative_terms(text: &str) -> Vec<String> {
    gmms_terms(text)
        .into_iter()
        .filter(|term| !gmms_term_is_stopword(term))
        .collect()
}

fn gmms_term_is_stopword(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "be"
            | "by"
            | "can"
            | "do"
            | "does"
            | "for"
            | "from"
            | "give"
            | "how"
            | "if"
            | "in"
            | "is"
            | "it"
            | "me"
            | "not"
            | "of"
            | "on"
            | "only"
            | "or"
            | "should"
            | "the"
            | "this"
            | "to"
            | "use"
            | "what"
            | "when"
            | "with"
            | "you"
            | "your"
    )
}

fn gmms_mode_for_event(event: &MistakeReflexEvent) -> String {
    if event
        .allowed_actions
        .iter()
        .any(|action| action == "do_not_inject_final_answer")
    {
        return "skill_reflex".to_string();
    }
    let family = event.symbolic_key.as_deref().unwrap_or_default();
    if !event.accepted_surfaces.is_empty()
        || family.contains("personal_fact")
        || family.contains("preference")
        || family.contains("entity_membership")
    {
        return "fact_personal".to_string();
    }
    "unknown".to_string()
}

fn gmms_allowed_action_max(event: &MistakeReflexEvent) -> String {
    let actions = &event.allowed_actions;
    if actions.iter().any(|action| action == "ask_if_uncertain") {
        "ask_if_uncertain"
    } else if actions
        .iter()
        .any(|action| action == "preserve_earned_boundary")
    {
        "preserve_earned_boundary"
    } else if actions
        .iter()
        .any(|action| action == "attach_route_unicode_sidecar_if_semantic_gate_passed")
    {
        "light_route_unicode_sidecar_use"
    } else if actions.iter().any(|action| action == "suppress_stale_path") {
        "suppress_stale_path"
    } else if actions
        .iter()
        .any(|action| action == "require_evidence_before_lock")
    {
        "require_evidence"
    } else if actions
        .iter()
        .any(|action| action == "inject_short_reflex_hint")
    {
        "context_process_hint"
    } else {
        "observe_only"
    }
    .to_string()
}

fn match_score(event: &MistakeReflexEvent, normalized_prompt: &str) -> f32 {
    match_score_with_schema(event, normalized_prompt, None)
}

fn match_score_with_schema(
    event: &MistakeReflexEvent,
    normalized_prompt: &str,
    schema_override: Option<&ReflexTaskSchema>,
) -> f32 {
    let domain_match = match event.domain.as_str() {
        // §10eq gap: this branch hard-requires the prompt to contain the
        // literal source word (e.g. "mississippi"). A captured letter-count
        // reflex therefore matches only on prompts about the SAME word, not
        // on prompts in the same task class with a different word. The fix
        // is to parameterize this on the event/schema so generalized
        // captures (where word-in-prompt is NOT required) can match by
        // (count + target_char) only. See §10eq in CLAIMS.md and
        // artifacts/codex_capture_to_reload_20260506/ for the failing case.
        "symbolic_counting:letter_count" => {
            if let Some(ReflexTaskSchema::SymbolicCounting {
                word, target_char, ..
            }) = schema_override
            {
                normalized_prompt.contains(word)
                    && (normalized_prompt.contains("count")
                        || normalized_prompt.contains("how many"))
                    && prompt_mentions_target(normalized_prompt, target_char)
            } else {
                let Some((word, target)) = event_word_and_target(event) else {
                    return 0.0;
                };
                normalized_prompt.contains(&word)
                    && (normalized_prompt.contains("count")
                        || normalized_prompt.contains("how many"))
                    && prompt_mentions_target(normalized_prompt, &target)
            }
        }
        "parallel_duration:drying" => {
            if let Some(ReflexTaskSchema::ParallelDuration { item, .. }) = schema_override {
                let item_plural = pluralize_item(item);
                (normalized_prompt.contains(item) || normalized_prompt.contains(&item_plural))
                    && normalized_prompt.contains("dry")
                    && !explicit_sequential_duration_prompt(normalized_prompt)
            } else {
                let Some(item) = event_parallel_item(event) else {
                    return 0.0;
                };
                let item_plural = pluralize_item(&item);
                (normalized_prompt.contains(&item) || normalized_prompt.contains(&item_plural))
                    && normalized_prompt.contains("dry")
                    && !explicit_sequential_duration_prompt(normalized_prompt)
            }
        }
        "gmms:semantic_correction_slice" => {
            gmms_semantic_correction_match(event, normalized_prompt)
        }
        _ => false,
    };
    if !domain_match {
        return 0.0;
    }
    let mut score = 1.0;
    for term in &event.trigger_terms {
        if term.len() >= 2 && normalized_prompt.contains(&normalize_text(term)) {
            score += 0.20;
        }
    }
    score * event.confidence * (1.0 + event.action_level as f32 * 0.10)
}

fn gmms_semantic_correction_match(event: &MistakeReflexEvent, normalized_prompt: &str) -> bool {
    if requires_rejected_surface_anchor(event) {
        return event.rejected_surfaces.iter().any(|surface| {
            let normalized = normalize_text(surface);
            (!normalized.is_empty() && normalized_prompt.contains(&normalized))
                || gmms_rejected_surface_alias_match(event, surface, normalized_prompt)
        });
    }

    let prompt_terms = gmms_informative_terms(normalized_prompt);
    let trigger_hits = event
        .trigger_terms
        .iter()
        .flat_map(|term| gmms_informative_terms(term))
        .filter(|term| prompt_terms.contains(term))
        .take(3)
        .count();
    if trigger_hits >= 2 {
        return true;
    }
    event.example_anchor.as_deref().is_some_and(|anchor| {
        gmms_informative_terms(anchor)
            .iter()
            .filter(|term| prompt_terms.contains(*term))
            .count()
            >= 3
    }) || event.rejected_surfaces.iter().any(|surface| {
        let normalized = normalize_text(surface);
        !normalized.is_empty() && normalized_prompt.contains(&normalized)
    })
}

fn is_claims_section_tag_event(event: &MistakeReflexEvent) -> bool {
    event.symbolic_key.as_deref() == Some("oracle:claims_section_tag")
        || event.id.contains("claims_section_tag")
}

fn is_path_exists_event(event: &MistakeReflexEvent) -> bool {
    event.symbolic_key.as_deref() == Some("oracle:path_exists") || event.id.contains("path_exists")
}

fn requires_rejected_surface_anchor(event: &MistakeReflexEvent) -> bool {
    is_claims_section_tag_event(event) || is_path_exists_event(event)
}

fn gmms_rejected_surface_alias_match(
    event: &MistakeReflexEvent,
    surface: &str,
    normalized_prompt: &str,
) -> bool {
    if is_claims_section_tag_event(event) {
        let surface_terms = gmms_informative_terms(surface);
        if surface_terms.len() < 2 {
            return false;
        }
        let prompt_terms = gmms_informative_terms(normalized_prompt);
        let Some(section_pos) = prompt_terms
            .iter()
            .position(|term| term == &surface_terms[0])
        else {
            return false;
        };
        let window_end = (section_pos + 6).min(prompt_terms.len());
        return surface_terms[1..]
            .iter()
            .all(|term| prompt_terms[section_pos + 1..window_end].contains(term));
    }

    if is_path_exists_event(event) {
        let surface_terms = gmms_informative_terms(surface);
        if surface_terms.len() < 2 {
            return false;
        }
        let prompt_terms = gmms_informative_terms(normalized_prompt);
        let path_context = [
            "path",
            "repo",
            "file",
            "folder",
            "directory",
            "exists",
            "exist",
            "cited",
            "cite",
        ]
        .iter()
        .any(|term| prompt_terms.iter().any(|prompt_term| prompt_term == term));
        return path_context && surface_terms.iter().all(|term| prompt_terms.contains(term));
    }

    false
}

fn prompt_schema_for_memory_slice(
    event: &MistakeReflexEvent,
    normalized_prompt: &str,
) -> Option<ReflexTaskSchema> {
    match event.symbolic_key.as_deref() {
        Some("memory_slice_v1:symbolic_counting") => {
            let (word, target) = extract_letter_count_task(normalized_prompt)?;
            let target_char = target.chars().next()?.to_ascii_lowercase();
            let expected_count = word
                .chars()
                .filter(|ch| ch.to_ascii_lowercase() == target_char)
                .count() as i64;
            if !is_dominant_repeated_letter_task(&word, target_char, expected_count) {
                return None;
            }
            Some(ReflexTaskSchema::SymbolicCounting {
                word,
                target_char: target,
                expected_count,
            })
        }
        Some("memory_slice_v1:parallel_duration") => {
            if explicit_sequential_duration_prompt(normalized_prompt) {
                return None;
            }
            let item = parallel_item_from_text(normalized_prompt)?;
            let single_duration = extract_hour_numbers(normalized_prompt)
                .first()
                .copied()
                .unwrap_or(5);
            Some(ReflexTaskSchema::ParallelDuration {
                item,
                single_duration,
                unit: "hours".to_string(),
                total_items: extract_parallel_total_items(normalized_prompt),
                capacity_mode: "parallel_if_capacity_present".to_string(),
            })
        }
        _ => None,
    }
}

fn accepted_surfaces_for_prompt_schema(schema: &ReflexTaskSchema) -> Vec<String> {
    match schema {
        ReflexTaskSchema::SymbolicCounting {
            target_char,
            expected_count,
            ..
        } => {
            let plural = format!("{target_char}s");
            let possessive = format!("{target_char}'s");
            let word = number_word(*expected_count).unwrap_or("");
            [
                format!("{expected_count} {plural}"),
                format!("{expected_count} {possessive}"),
                format!("{expected_count} {target_char}"),
                expected_count.to_string(),
                format!("{word} {plural}"),
                format!("there are {expected_count}"),
            ]
            .into_iter()
            .filter(|surface| !surface.trim().is_empty())
            .collect()
        }
        ReflexTaskSchema::ParallelDuration {
            single_duration,
            unit,
            ..
        } => {
            let word = number_word(*single_duration).unwrap_or("");
            [
                format!("{single_duration} {unit}"),
                format!("{single_duration} hour"),
                format!("{word} {unit}"),
            ]
            .into_iter()
            .filter(|surface| !surface.trim().is_empty())
            .collect()
        }
    }
}

fn accepted_surfaces_for_prompt_process_event(
    event: &MistakeReflexEvent,
    normalized_prompt: &str,
) -> Vec<String> {
    if !is_shelf_order_process_event(event) {
        return Vec::new();
    }
    let Some(items) = extract_shelf_order_items(normalized_prompt) else {
        return Vec::new();
    };
    if items.len() < 2 || items.len() > 12 {
        return Vec::new();
    }
    let mut ordered = items
        .into_iter()
        .enumerate()
        .map(|(idx, item)| {
            (
                idx,
                item.chars().filter(|ch| ch.is_alphanumeric()).count(),
                item,
            )
        })
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(idx, len, _)| (*len, *idx));
    let sorted = ordered
        .into_iter()
        .map(|(_, _, item)| item)
        .collect::<Vec<_>>();
    let comma = sorted.join(", ");
    let slash = sorted.join(" / ");
    let arrow = sorted.join(" -> ");
    let space = sorted.join(" ");
    let newline = sorted.join("\n");
    [comma, slash, arrow, space, newline]
        .into_iter()
        .filter(|surface| !surface.trim().is_empty())
        .collect()
}

fn is_shelf_order_process_event(event: &MistakeReflexEvent) -> bool {
    [
        event.symbolic_key.as_deref().unwrap_or_default(),
        event.id.as_str(),
        event.corrected_reflex.as_str(),
        event.procedural_rule.as_deref().unwrap_or_default(),
        event.evidence_requirement.as_str(),
    ]
    .iter()
    .any(|value| {
        let normalized = normalize_text(value);
        normalized.contains("shelf-order")
            || normalized.contains("shelf order")
            || normalized.contains("word length")
            || normalized.contains("letter-count sort")
            || normalized.contains("letter count sort")
    })
}

fn extract_shelf_order_items(normalized_prompt: &str) -> Option<Vec<String>> {
    let marker = ["item list:", "items:"]
        .into_iter()
        .find_map(|marker| normalized_prompt.find(marker).map(|idx| (idx, marker)))?;
    let mut rest = normalized_prompt[marker.0 + marker.1.len()..].trim();
    for delimiter in [". return", ". output", ". give", ". answer", "?"] {
        if let Some(idx) = rest.find(delimiter) {
            rest = &rest[..idx];
            break;
        }
    }
    let rest = rest
        .replace(" and ", ", ")
        .replace(" then ", ", ")
        .replace('|', ",");
    let items = rest
        .split([',', '/', ';'])
        .map(|item| {
            item.trim_matches(|ch: char| {
                ch.is_whitespace()
                    || matches!(
                        ch,
                        '.' | ':' | '"' | '\'' | '`' | '[' | ']' | '(' | ')' | '{' | '}'
                    )
            })
            .trim()
            .to_string()
        })
        .filter(|item| {
            !item.is_empty()
                && item.len() <= 48
                && item.chars().any(|ch| ch.is_alphabetic())
                && !item.contains("return only")
                && !item.contains("ordered list")
        })
        .collect::<Vec<_>>();
    (!items.is_empty()).then_some(items)
}

fn old_path_seen(item: &MistakeReflexMatch, assistant_text: &str) -> bool {
    let text = normalize_text(assistant_text);
    if item.domain == "parallel_duration:drying" {
        if parallel_serial_multiplication_seen(&text) {
            return true;
        }
    }
    if item.domain == "symbolic_counting:letter_count" && symbolic_bad_path_seen(item, &text) {
        return true;
    }
    if gmms_verification_policy_invalid_claim_decision_seen(item, &text) {
        return true;
    }
    item.rejected_surfaces
        .iter()
        .any(|surface| contains_rejected_surface(&text, surface))
}

fn old_mistake_seen(item: &MistakeReflexMatch, assistant_text: &str) -> bool {
    let text = normalize_text(assistant_text);
    if item.domain == "symbolic_counting:letter_count"
        && symbolic_terminal_acceptance_after_wobble(item, &text)
    {
        return false;
    }
    if item.domain == "parallel_duration:drying"
        && parallel_terminal_acceptance_after_retry(item, &text)
    {
        return false;
    }
    old_path_seen(item, assistant_text)
}

fn evidence_seen(item: &MistakeReflexMatch, assistant_text: &str) -> bool {
    let text = normalize_text(assistant_text);
    match item.domain.as_str() {
        "symbolic_counting:letter_count" => symbolic_counting_evidence_seen(item, &text),
        "parallel_duration:drying" => {
            if parallel_terminal_acceptance_after_retry(item, &text) {
                return true;
            }
            if parallel_serial_multiplication_seen(&text) {
                return false;
            }
            text.contains("parallel")
                || text.contains("same time")
                || text.contains("same amount")
                || text.contains("same amount of time")
                || text.contains("simultaneous")
                || text.contains("sequential")
                || text.contains("capacity")
                || text.contains("additive")
                || text.contains("not multiplicative")
        }
        "gmms:semantic_correction_slice" => {
            if gmms_verification_policy_decision_seen(item, assistant_text) {
                return true;
            }
            gmms_process_contract_evidence_seen(item, assistant_text)
        }
        _ => false,
    }
}

fn gmms_process_contract_evidence_seen(item: &MistakeReflexMatch, assistant_text: &str) -> bool {
    if !(is_shelf_order_process_match(item) || is_verification_policy_process_match(item))
        || item.accepted_surfaces.is_empty()
    {
        return false;
    }
    if is_verification_policy_process_match(item) {
        return gmms_verification_policy_decision_seen(item, assistant_text);
    }
    let compact = compact_whitespace(assistant_text, 4096);
    let upper = compact.to_ascii_uppercase();
    let has_answer_marker = [
        "EXACT OUTPUT:",
        "FINAL ANSWER:",
        "VISIBLE ANSWER:",
        "WORKING ANSWER:",
        "VISIBLE WORKING ANSWER:",
        "ANSWER:",
    ]
    .iter()
    .any(|marker| upper.contains(marker));
    has_answer_marker
        && item
            .accepted_surfaces
            .iter()
            .any(|surface| contains_accepted_surface(&compact, surface))
}

fn gmms_verification_policy_decision_seen(item: &MistakeReflexMatch, assistant_text: &str) -> bool {
    if !is_verification_policy_process_match(item) || item.accepted_surfaces.is_empty() {
        return false;
    }
    let compact = compact_whitespace(assistant_text, 4096);
    let normalized = normalize_text(&compact);
    let has_accepted_surface = item
        .accepted_surfaces
        .iter()
        .any(|surface| contains_accepted_surface(&compact, surface));
    if !has_accepted_surface {
        return false;
    }
    let rejects_startup_only = gmms_verification_policy_rejects_startup_only(&normalized);
    let requires_telemetry_evidence =
        gmms_verification_policy_requires_telemetry_evidence(&normalized);
    rejects_startup_only && requires_telemetry_evidence
}

fn gmms_verification_policy_rejects_startup_only(normalized: &str) -> bool {
    [
        "cannot accept bridge-influence",
        "cannot accept bridge influence",
        "should not move green",
        "shouldn't move green",
        "claims should not move green",
        "should not be updated to bridge_influence=green",
        "should not be updated to bridge influence green",
        "claim should not be updated",
        "reject the claim",
        "reject the move",
        "reject moving",
        "reject startup",
        "not sufficient evidence",
        "not sufficient to justify",
        "startup logs are not sufficient",
        "startup logs are insufficient",
        "not enough evidence",
        "does not meet the required evidence",
        "doesn't meet the required evidence",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn gmms_verification_policy_requires_telemetry_evidence(normalized: &str) -> bool {
    [
        "raw per-token jsonl",
        "raw per token jsonl",
        "per-token jsonl",
        "per token jsonl",
        "generated-output telemetry",
        "generated output telemetry",
        "raw telemetry",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn gmms_verification_policy_invalid_claim_decision_seen(
    item: &MistakeReflexMatch,
    normalized_text: &str,
) -> bool {
    if !is_verification_policy_process_match(item) {
        return false;
    }
    let red_or_falsified = [
        "bridge_influence=red",
        "bridge influence=red",
        "bridge influence red",
        "claims: bridge_influence red",
        "claims: bridge influence red",
        "mark bridge influence red",
        "move bridge influence red",
        "claim is red",
        "claim should be red",
        "falsify bridge influence",
        "bridge influence falsified",
    ]
    .iter()
    .any(|needle| normalized_text.contains(needle));
    red_or_falsified && !gmms_verification_policy_requires_telemetry_evidence(normalized_text)
}

fn is_verification_policy_process_match(item: &MistakeReflexMatch) -> bool {
    [
        item.symbolic_key.as_deref().unwrap_or_default(),
        item.event_id.as_str(),
        item.corrected_reflex.as_str(),
        item.procedural_rule.as_deref().unwrap_or_default(),
        item.evidence_requirement.as_str(),
    ]
    .iter()
    .any(|value| {
        let normalized = normalize_text(value);
        (normalized.contains("raw telemetry")
            || normalized.contains("per token jsonl")
            || normalized.contains("per-token jsonl")
            || normalized.contains("startup log"))
            && (normalized.contains("claim")
                || normalized.contains("verification")
                || normalized.contains("bridge influence")
                || normalized.contains("evidence"))
    })
}

fn is_shelf_order_process_match(item: &MistakeReflexMatch) -> bool {
    [
        item.symbolic_key.as_deref().unwrap_or_default(),
        item.event_id.as_str(),
        item.corrected_reflex.as_str(),
        item.procedural_rule.as_deref().unwrap_or_default(),
        item.evidence_requirement.as_str(),
    ]
    .iter()
    .any(|value| {
        let normalized = normalize_text(value);
        normalized.contains("shelf-order")
            || normalized.contains("shelf order")
            || normalized.contains("word length")
            || normalized.contains("letter-count sort")
            || normalized.contains("letter count sort")
    })
}

fn accepted_answer_candidate_seen(item: &MistakeReflexMatch, assistant_text: &str) -> bool {
    let compact = compact_whitespace(assistant_text, 4096);
    item.accepted_surfaces
        .iter()
        .any(|surface| contains_accepted_surface(&compact, surface))
}

fn earned_answer_seen(item: &MistakeReflexMatch, assistant_text: &str) -> Option<String> {
    if !evidence_seen(item, assistant_text) || old_mistake_seen(item, assistant_text) {
        return None;
    }
    let compact = compact_whitespace(assistant_text, 4096);
    if gmms_verification_policy_decision_seen(item, assistant_text) {
        if let Some(surface) = item
            .accepted_surfaces
            .iter()
            .find(|surface| contains_accepted_surface(&compact, surface))
        {
            return Some(surface.clone());
        }
    }
    let upper = compact.to_ascii_uppercase();
    for marker in [
        "EXACT OUTPUT:",
        "FINAL ANSWER:",
        "VISIBLE ANSWER:",
        "WORKING ANSWER:",
        "VISIBLE WORKING ANSWER:",
        "ANSWER:",
    ] {
        let mut start = 0usize;
        while let Some(idx) = upper[start..].find(marker) {
            let abs = start + idx;
            let window = &compact[abs..compact.len().min(abs + 220)];
            if let Some(surface) = item.accepted_surfaces.iter().find(|surface| {
                if surface.chars().all(|ch| ch.is_ascii_digit())
                    && (marker == "WORKING ANSWER:" || marker == "VISIBLE WORKING ANSWER:")
                {
                    return false;
                }
                contains_accepted_surface(window, surface)
            }) {
                return Some(surface.clone());
            }
            start = abs + marker.len();
        }
    }
    None
}

fn symbolic_terminal_acceptance_after_wobble(
    item: &MistakeReflexMatch,
    normalized_text: &str,
) -> bool {
    if !symbolic_counting_evidence_seen(item, normalized_text) {
        return false;
    }
    let accepted_pos = item
        .accepted_surfaces
        .iter()
        .filter_map(|surface| normalized_text.rfind(&normalize_text(surface)))
        .max();
    let Some(accepted_pos) = accepted_pos else {
        return false;
    };
    let rejected_pos = item
        .rejected_surfaces
        .iter()
        .filter_map(|surface| normalized_text.rfind(&normalize_text(surface)))
        .max();
    rejected_pos.map(|pos| accepted_pos > pos).unwrap_or(true)
}

fn symbolic_counting_evidence_seen(item: &MistakeReflexMatch, normalized_text: &str) -> bool {
    let (word, target, expected_count) = match &item.schema {
        Some(ReflexTaskSchema::SymbolicCounting {
            word,
            target_char,
            expected_count,
        }) => (word.clone(), target_char.clone(), *expected_count),
        _ => {
            let (word, target) = match_word_and_target(item)
                .unwrap_or_else(|| ("strawberry".to_string(), "r".to_string()));
            let expected_count = item
                .accepted_surfaces
                .iter()
                .filter_map(|surface| {
                    normalize_text(surface)
                        .split_whitespace()
                        .next()
                        .and_then(parse_count_word)
                })
                .next()
                .unwrap_or(3);
            (word, target, expected_count)
        }
    };
    let spelled = word
        .chars()
        .map(|ch| ch.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let has_sequence = normalized_text.contains(&spelled);
    let visible_scan_count = count_target_from_visible_scan(normalized_text, &word, &target);
    if visible_scan_count == Some(expected_count) {
        return true;
    }
    let one = format!("{target}=1");
    let one_colon = format!("{target}:1");
    let one_arrow = format!("{target} -> 1");
    let two = format!("{target}=2");
    let two_colon = format!("{target}:2");
    let two_arrow = format!("{target} -> 2");
    let three = format!("{target}=3");
    let three_colon = format!("{target}:3");
    let three_arrow = format!("{target} -> 3");
    let has_running_count = (normalized_text.contains(&one)
        || normalized_text.contains(&one_colon)
        || normalized_text.contains(&one_arrow))
        && (normalized_text.contains(&two)
            || normalized_text.contains(&two_colon)
            || normalized_text.contains(&two_arrow))
        && (normalized_text.contains(&three)
            || normalized_text.contains(&three_colon)
            || normalized_text.contains(&three_arrow));
    let has_ordinal_count = (normalized_text.contains(&format!("first {target}"))
        || normalized_text.contains(&format!("one {target}")))
        && (normalized_text.contains(&format!("second {target}"))
            || normalized_text.contains(&format!("two {target}")))
        && (normalized_text.contains(&format!("third {target}"))
            || normalized_text.contains(&format!("three {target}")));
    let has_so_far_count = normalized_text.contains(&format!("2 {target}s so far"))
        && normalized_text.contains(&format!("3 {target}s so far"));
    let has_accepted_count_after_scan = item
        .accepted_surfaces
        .iter()
        .any(|surface| normalized_text.contains(&normalize_text(surface)))
        && (normalized_text.contains("so far")
            || normalized_text.contains("found another")
            || normalized_text.contains("count"));
    has_sequence
        && (has_running_count
            || has_ordinal_count
            || has_so_far_count
            || has_accepted_count_after_scan)
}

fn symbolic_bad_path_seen(item: &MistakeReflexMatch, normalized_text: &str) -> bool {
    let Some((word, _target)) = match_word_and_target(item) else {
        return false;
    };
    normalized_text.contains(&format!(
        "{} has no",
        word.chars().take(4).collect::<String>()
    )) || (normalized_text.contains("composed of both words")
        && normalized_text.contains("add the")
        && normalized_text.contains("together"))
}

fn count_target_from_visible_scan(normalized_text: &str, word: &str, target: &str) -> Option<i64> {
    let target_char = target.chars().next()?.to_ascii_lowercase();
    let spelled = word
        .chars()
        .map(|ch| ch.to_ascii_lowercase().to_string())
        .collect::<Vec<_>>()
        .join("-");
    if normalized_text.contains(&spelled) {
        return Some(
            word.chars()
                .filter(|ch| ch.to_ascii_lowercase() == target_char)
                .count() as i64,
        );
    }

    let explicit_scan_count = max_target_count_from_letter_scan(normalized_text, target_char);
    if explicit_scan_count.is_some() {
        return explicit_scan_count;
    }

    let mut enumerated = Vec::new();
    let chars = normalized_text.chars().collect::<Vec<_>>();
    let mut idx = 0usize;
    while idx < chars.len() {
        if !chars[idx].is_ascii_digit() {
            idx += 1;
            continue;
        }
        while idx < chars.len() && chars[idx].is_ascii_digit() {
            idx += 1;
        }
        while idx < chars.len() && chars[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= chars.len() || chars[idx] != '.' {
            continue;
        }
        idx += 1;
        while idx < chars.len() && chars[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < chars.len() && chars[idx].is_ascii_alphabetic() {
            enumerated.push(chars[idx].to_ascii_lowercase());
        }
    }
    if enumerated.len() >= word.len() {
        return Some(
            enumerated
                .iter()
                .filter(|letter| **letter == target_char)
                .count() as i64,
        );
    }
    None
}

fn max_target_count_from_letter_scan(normalized_text: &str, target_char: char) -> Option<i64> {
    let chars = normalized_text.chars().collect::<Vec<_>>();
    let mut idx = 0usize;
    let mut best: Option<i64> = None;
    while idx < chars.len() {
        if chars[idx].to_ascii_lowercase() != target_char {
            idx += 1;
            continue;
        }
        if idx > 0 && chars[idx - 1].is_ascii_alphanumeric() {
            idx += 1;
            continue;
        }
        idx += 1;
        if idx < chars.len() && chars[idx].is_ascii_alphanumeric() {
            continue;
        }
        while idx < chars.len() && chars[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx >= chars.len() || !['=', ':'].contains(&chars[idx]) {
            if idx + 1 < chars.len() && chars[idx] == '-' && chars[idx + 1] == '>' {
                idx += 2;
            } else {
                continue;
            }
        } else {
            idx += 1;
        }
        while idx < chars.len() && chars[idx].is_ascii_whitespace() {
            idx += 1;
        }
        let start = idx;
        while idx < chars.len() && chars[idx].is_ascii_digit() {
            idx += 1;
        }
        if start == idx {
            continue;
        }
        let value = chars[start..idx]
            .iter()
            .collect::<String>()
            .parse::<i64>()
            .ok()?;
        best = Some(best.map(|current| current.max(value)).unwrap_or(value));
    }
    best
}

fn is_dominant_repeated_letter_task(word: &str, target_char: char, expected_count: i64) -> bool {
    if expected_count < 3 {
        return false;
    }
    let mut counts = [0i64; 26];
    for ch in word.chars().filter(|ch| ch.is_ascii_alphabetic()) {
        let idx = (ch.to_ascii_lowercase() as u8 - b'a') as usize;
        counts[idx] += 1;
    }
    let target_idx = (target_char as u8).saturating_sub(b'a') as usize;
    let max_count = counts.iter().copied().max().unwrap_or_default();
    target_idx < counts.len() && counts[target_idx] == max_count
}

fn parallel_serial_multiplication_seen(normalized_text: &str) -> bool {
    normalized_text.contains("*")
        || normalized_text.contains(" x ")
        || normalized_text.contains("multiply")
        || normalized_text.contains("multiplying")
        || normalized_text.contains("add the drying time")
        || normalized_text.contains("additional shirts")
        || normalized_text.contains("additional towels")
        || normalized_text.contains("total drying time")
}

fn parallel_terminal_acceptance_after_retry(
    item: &MistakeReflexMatch,
    normalized_text: &str,
) -> bool {
    let has_parallel_evidence = normalized_text.contains("parallel")
        || normalized_text.contains("same time")
        || normalized_text.contains("enough rack")
        || normalized_text.contains("doesn't increase")
        || normalized_text.contains("does not increase")
        || normalized_text.contains("base time")
        || normalized_text.contains("remains at the base time")
        || normalized_text.contains("remaining shirts");
    if !has_parallel_evidence {
        return false;
    }
    let accepted_pos = if let Some(ReflexTaskSchema::ParallelDuration {
        single_duration,
        unit,
        ..
    }) = &item.schema
    {
        normalized_text.rfind(&format!("{single_duration} {unit}"))
    } else {
        item.accepted_surfaces
            .iter()
            .filter_map(|surface| normalized_text.rfind(&normalize_text(surface)))
            .max()
    };
    let Some(accepted_pos) = accepted_pos else {
        return false;
    };
    let rejected_pos = item
        .rejected_surfaces
        .iter()
        .filter_map(|surface| normalized_text.rfind(&normalize_text(surface)))
        .chain(
            [
                "*",
                " x ",
                "multiply",
                "multiplying",
                "add the drying time",
                "additional shirts",
                "additional towels",
                "total drying time",
            ]
            .iter()
            .filter_map(|surface| normalized_text.rfind(surface)),
        )
        .max();
    rejected_pos.map(|pos| accepted_pos > pos).unwrap_or(true)
}

fn extract_hour_numbers(text: &str) -> Vec<i64> {
    let words = text
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let mut out = Vec::new();
    for idx in 0..words.len() {
        let Some(value) = parse_count_word(words[idx]) else {
            continue;
        };
        let next = words.get(idx + 1).copied().unwrap_or_default();
        if next == "hour" || next == "hours" {
            out.push(value);
        }
    }
    normalize_numbers(&out)
}

fn extract_parallel_total_items(text: &str) -> Option<i64> {
    let words = text
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    let mut base = None;
    let mut more = None;
    for idx in 0..words.len() {
        let Some(value) = parse_count_word(words[idx]) else {
            continue;
        };
        let after = words.get(idx + 1).copied().unwrap_or_default();
        if parallel_item_from_text(after).is_some() {
            base.get_or_insert(value);
        }
        if words.get(idx + 1) == Some(&"more") {
            more = Some(value);
        }
    }
    Some(base.unwrap_or(1) + more?)
}

fn contains_accepted_surface(text: &str, surface: &str) -> bool {
    let text = normalize_text(text);
    let surface = normalize_text(surface);
    if surface.is_empty() {
        return false;
    }
    if surface.chars().all(|ch| ch.is_ascii_digit()) {
        return contains_token_anywhere(&text, &surface);
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
        if abs + surface.len() + 16 >= normalized_text.len()
            && (before.contains("working answer") || before.contains("working_answer"))
        {
            start = abs + surface.len();
            continue;
        }
        let after = &normalized_text
            [abs + surface.len()..normalized_text.len().min(abs + surface.len() + 32)];
        if after.contains("so far") || after.contains("so-far") {
            start = abs + surface.len();
            continue;
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

fn contains_token_anywhere(text: &str, token: &str) -> bool {
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

fn extract_letter_count_task(normalized_prompt: &str) -> Option<(String, String)> {
    let words = normalized_prompt
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '\'')
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>();
    for idx in 0..words.len() {
        let Some(target) = normalize_letter_target(words[idx]) else {
            continue;
        };
        let word = if words.get(idx + 1) == Some(&"in") {
            words.get(idx + 2).copied()
        } else {
            words
                .windows(2)
                .find(|window| window[0] == "in")
                .map(|window| window[1])
        }?;
        if word.len() < 3 || word == "the" {
            continue;
        }
        let before = words[..idx].join(" ");
        if before.contains("number of") || before.contains("count") || before.contains("many") {
            return Some((word.to_string(), target));
        }
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

fn event_word_and_target(event: &MistakeReflexEvent) -> Option<(String, String)> {
    word_and_target_from_terms(&event.trigger_terms)
}

fn match_word_and_target(item: &MistakeReflexMatch) -> Option<(String, String)> {
    word_and_target_from_terms(&item.trigger_terms)
}

fn word_and_target_from_terms(terms: &[String]) -> Option<(String, String)> {
    let target = terms
        .iter()
        .find(|term| term.len() == 1 && term.chars().all(|ch| ch.is_ascii_alphabetic()))?
        .clone();
    let word = terms
        .iter()
        .filter(|term| term.len() >= 3)
        .find(|term| !["count", "letter", "letters"].contains(&term.as_str()))?
        .clone();
    Some((word, target))
}

fn event_parallel_item(event: &MistakeReflexEvent) -> Option<String> {
    event
        .trigger_terms
        .iter()
        .filter(|term| term.len() >= 3)
        .find(|term| !["dry", "drying", "hours", "more"].contains(&term.as_str()))
        .cloned()
}

fn parallel_item_from_text(text: &str) -> Option<String> {
    for item in ["towel", "shirt", "cookie", "cloth", "blanket"] {
        let plural = pluralize_item(item);
        if contains_token_anywhere(text, item) || contains_token_anywhere(text, &plural) {
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

fn prompt_mentions_target(normalized_prompt: &str, target: &str) -> bool {
    let plural = format!("{target}s");
    let possessive = format!("{target}'s");
    contains_token_anywhere(normalized_prompt, target)
        || contains_token_anywhere(normalized_prompt, &plural)
        || normalized_prompt.contains(&possessive)
}

fn explicit_sequential_duration_prompt(normalized_prompt: &str) -> bool {
    normalized_prompt.contains("after the previous")
        || normalized_prompt.contains("previous towel finishes")
        || normalized_prompt.contains("one after another")
        || normalized_prompt.contains("sequential")
        || normalized_prompt.contains("sequentially")
}

fn symbolic_scan_prefix_example(word: &str, target: &str) -> String {
    let Some(target_char) = target.chars().next().map(|ch| ch.to_ascii_lowercase()) else {
        return String::new();
    };
    let mut running = 0i64;
    word.chars()
        .take(2)
        .map(|ch| {
            if ch.to_ascii_lowercase() == target_char {
                running += 1;
            }
            format!("{}={running}", ch.to_ascii_lowercase())
        })
        .collect::<Vec<_>>()
        .join(" | ")
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

#[derive(Debug, Clone, Default)]
struct PacketSlice {
    hidden_full_path: Option<String>,
    hidden_dim: Option<usize>,
    route_64d: Option<Vec<f32>>,
    route_motif_id: Option<String>,
    unicode_packet_id: Option<String>,
    unicode_escape: Option<String>,
    decoded_route_id: Option<String>,
    route_preserved: Option<bool>,
}

fn find_packet_slice_for_event(
    text: &str,
    event: &MistakeReflexEvent,
) -> Result<Option<PacketSlice>> {
    let mut best: Option<(i32, PacketSlice)> = None;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value: Value = serde_json::from_str(line).with_context(|| {
            format!(
                "Failed to parse Unicode memory packet JSONL line {}",
                line_no + 1
            )
        })?;
        let score = packet_event_score(&value, event);
        if score <= 0 {
            continue;
        }
        let slice = packet_slice_from_value(&value);
        if slice.unicode_packet_id.is_none() && slice.route_motif_id.is_none() {
            continue;
        }
        let replace = best
            .as_ref()
            .map(|(best_score, _)| score > *best_score)
            .unwrap_or(true);
        if replace {
            best = Some((score, slice));
        }
    }
    Ok(best.map(|(_, slice)| slice))
}

fn packet_event_score(value: &Value, event: &MistakeReflexEvent) -> i32 {
    let event_family = memory_family_for_event(event);
    let explicit_event_refs = string_values_at_paths(
        value,
        &[
            "/memory/event_id",
            "/selection/event_id",
            "/memory/gmms_event_id",
            "/selection/gmms_event_id",
        ],
    );
    if !explicit_event_refs.is_empty()
        && !explicit_event_refs
            .iter()
            .any(|candidate| event_ref_matches(candidate, &event.id))
    {
        return 0;
    }
    let explicit_symbolic_refs = string_values_at_paths(
        value,
        &[
            "/memory/symbolic_key",
            "/selection/symbolic_key",
            "/memory/family_id",
            "/selection/family_id",
        ],
    );
    if !explicit_symbolic_refs.is_empty()
        && !explicit_symbolic_refs.iter().any(|candidate| {
            event
                .symbolic_key
                .as_deref()
                .is_some_and(|symbolic_key| event_ref_matches(candidate, symbolic_key))
        })
    {
        return 0;
    }
    let explicit_slice_refs = string_values_at_paths(
        value,
        &[
            "/memory/slice_id",
            "/selection/slice_id",
            "/memory/gmms_slice_id",
            "/selection/gmms_slice_id",
        ],
    );
    if !explicit_slice_refs.is_empty()
        && !explicit_slice_refs
            .iter()
            .any(|candidate| event_ref_matches(candidate, &event.id))
    {
        return 0;
    }
    let metadata_domains = string_values_at_paths(
        value,
        &[
            "/memory/domain",
            "/memory/reflex_domain",
            "/selection/target_domain",
            "/selection/reflex_domain",
        ],
    );
    let metadata_families = string_values_at_paths(
        value,
        &[
            "/memory/family",
            "/memory/task_family",
            "/selection/target_family",
            "/selection/family",
            "/task_family",
        ],
    );
    if !metadata_domains.is_empty()
        && !metadata_domains
            .iter()
            .any(|domain| normalize_text(domain) == event.domain)
    {
        return 0;
    }
    if !metadata_families.is_empty()
        && !metadata_families
            .iter()
            .any(|family| normalize_family(family) == event_family)
    {
        return 0;
    }
    let mut score = 0;
    if metadata_domains
        .iter()
        .any(|domain| normalize_text(domain) == event.domain)
    {
        score += 100;
    }
    if metadata_families
        .iter()
        .any(|family| normalize_family(family) == event_family)
    {
        score += 50;
    }
    if explicit_event_refs
        .iter()
        .any(|candidate| event_ref_matches(candidate, &event.id))
    {
        score += 200;
    }
    if explicit_symbolic_refs.iter().any(|candidate| {
        event
            .symbolic_key
            .as_deref()
            .is_some_and(|symbolic_key| event_ref_matches(candidate, symbolic_key))
    }) {
        score += 150;
    }
    if explicit_slice_refs
        .iter()
        .any(|candidate| event_ref_matches(candidate, &event.id))
    {
        score += 125;
    }

    let route_text = string_values_at_paths(
        value,
        &[
            "/geometry/original_route/motif_id",
            "/geometry/decoded_route/motif_id",
            "/source_route_id",
            "/decoded_nearest_route",
        ],
    )
    .join(" ");
    if route_text_indicates_other_supported_family(&route_text, &event_family) {
        return 0;
    }
    if !route_text.is_empty() && route_text_indicates_family(&route_text, &event_family) {
        score += 20;
    }

    let prompt = value
        .pointer("/source/prompt")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let prompt = normalize_text(prompt);
    score
        + match event.domain.as_str() {
            "symbolic_counting:letter_count" => {
                let mut score = 0;
                if prompt.contains("count") || prompt.contains("how many") {
                    score += 2;
                }
                if prompt.contains("strawberry") {
                    score += 4;
                }
                if prompt.contains(" r") || prompt.contains("letter") {
                    score += 1;
                }
                score
            }
            "parallel_duration:drying" => {
                let mut score = 0;
                if prompt.contains("towel") || prompt.contains("shirt") || prompt.contains("cookie")
                {
                    score += 2;
                }
                if prompt.contains("dry") || prompt.contains("parallel") || prompt.contains("time")
                {
                    score += 3;
                }
                score
            }
            _ => 0,
        }
}

fn event_ref_matches(candidate: &str, expected: &str) -> bool {
    let candidate = normalize_event_ref(candidate);
    let expected = normalize_event_ref(expected);
    // Never treat blank / whitespace-only refs as wildcard matches (historical slice-id bug class).
    if candidate.is_empty() {
        return false;
    }
    candidate == expected
        || candidate
            .strip_prefix("gmms_compat_")
            .is_some_and(|stripped| stripped == expected)
        || expected
            .strip_prefix("gmms_compat_")
            .is_some_and(|stripped| stripped == candidate)
}

fn normalize_event_ref(value: &str) -> String {
    normalize_text(value)
        .replace('-', "_")
        .replace('/', "_")
        .replace(':', "_")
}

fn string_values_at_paths(value: &Value, paths: &[&str]) -> Vec<String> {
    paths
        .iter()
        .filter_map(|path| value.pointer(path).and_then(Value::as_str))
        .map(ToString::to_string)
        .collect()
}

fn memory_family_for_event(event: &MistakeReflexEvent) -> String {
    match event.domain.as_str() {
        "symbolic_counting:letter_count" => "symbolic_counting".to_string(),
        "parallel_duration:drying" => "parallel_duration".to_string(),
        other => normalize_family(other),
    }
}

fn normalize_family(value: &str) -> String {
    normalize_text(value)
        .replace('-', "_")
        .replace('/', "_")
        .replace(':', "_")
}

fn route_text_indicates_family(route_text: &str, family: &str) -> bool {
    let route = normalize_family(route_text);
    match family {
        "symbolic_counting" => route.contains("counting") || route.contains("counting_trap"),
        "parallel_duration" => {
            route.contains("parallel_duration")
                || route.contains("parallel_time")
                || route.contains("time_trap")
        }
        other => route.contains(other),
    }
}

fn route_text_indicates_other_supported_family(route_text: &str, event_family: &str) -> bool {
    if route_text.trim().is_empty() {
        return false;
    }
    ["symbolic_counting", "parallel_duration"]
        .iter()
        .any(|family| *family != event_family && route_text_indicates_family(route_text, family))
}

fn packet_slice_from_value(value: &Value) -> PacketSlice {
    PacketSlice {
        hidden_full_path: value
            .pointer("/source/source_artifact")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        hidden_dim: value
            .pointer("/source/hidden_dim")
            .and_then(Value::as_u64)
            .map(|value| value as usize),
        route_64d: value
            .pointer("/vectors/decoded_64d")
            .and_then(Value::as_array)
            .and_then(|items| {
                if items.len() != 64 {
                    return None;
                }
                Some(
                    items
                        .iter()
                        .map(|item| item.as_f64().unwrap_or_default() as f32)
                        .collect::<Vec<_>>(),
                )
            }),
        route_motif_id: value
            .pointer("/geometry/original_route/motif_id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        unicode_packet_id: value
            .get("packet_id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        unicode_escape: value
            .pointer("/codec/unicode_escape")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        decoded_route_id: value
            .pointer("/geometry/decoded_route/motif_id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        route_preserved: value
            .pointer("/geometry/route_preserved")
            .and_then(Value::as_bool),
    }
}

fn route_preserved_summary(matches: &[MistakeReflexMatch]) -> Option<bool> {
    let mut saw_any = false;
    let mut all_preserved = true;
    for item in matches {
        if let Some(preserved) = item.route_preserved {
            saw_any = true;
            all_preserved &= preserved;
        }
    }
    saw_any.then_some(all_preserved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_event_ref_candidates_do_not_match_events() {
        assert!(
            !event_ref_matches("", "gmms_skill_initials_v1"),
            "empty slice/event ref must not act as wildcard"
        );
        assert!(
            !event_ref_matches("  \t  ", "gmms_skill_initials_v1"),
            "whitespace-only ref normalizes empty and must not match"
        );
        assert!(event_ref_matches(
            "gmms_skill_initials_v1",
            "gmms_skill_initials_v1"
        ));
    }

    #[test]
    fn captures_skill_reflexes_without_answers() {
        let mut memory = MistakeReflexMemory::default();
        let captured = memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry and towels still take 5 hours to dry",
            Some("There are 2 Rs. The towels take 50 hours."),
        );

        assert_eq!(captured.len(), 2);
        assert!(captured
            .iter()
            .any(|event| event.domain == "symbolic_counting:letter_count"));
        assert!(captured
            .iter()
            .any(|event| event.domain == "parallel_duration:drying"));
    }

    #[test]
    fn prompt_hint_does_not_inject_final_answer() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);

        let prompt = MistakeReflexMemory::apply_prompt(
            "count the number of Rs in strawberry",
            &matches,
            "text-hint",
        );

        assert!(prompt.contains("MISTAKE REFLEX"));
        assert!(prompt.contains("running count"));
        assert!(!prompt.contains("CORRECT_ANSWER"));
        assert!(!prompt.contains("START_VISIBLE_ANSWER"));
        assert!(!prompt.contains("There are 3"));
    }

    #[test]
    fn gmms_compat_jsonl_loads_and_matches_generic_runtime_surface() {
        let stamp = now_ms();
        let skill = serde_json::json!({
            "id": "gmms_compat:gmms_skill_initials_v1",
            "domain": "gmms:semantic_correction_slice",
            "trigger_terms": ["initials", "acronym", "first", "letter", "phrase", "derive"],
            "bad_reflex": "invent expansion; reuse stale abbreviation meaning; answer without deriving from prompt words",
            "corrected_reflex": "derive initials by taking first letters in order; do not expose a memorized final answer",
            "evidence_requirement": "uses only words present in the current prompt; does not include an invented expansion; does not expose final answer text from memory",
            "rejected_surfaces": [
                "invent expansion; reuse stale abbreviation meaning; answer without deriving from prompt words",
                "invented abbreviation expansion or stale acronym meaning"
            ],
            "accepted_surfaces": [],
            "schema": null,
            "allowed_actions": ["inject_short_reflex_hint", "do_not_inject_final_answer"],
            "confidence": 0.75,
            "action_level": 2,
            "decay_rate": 0.15,
            "success_count": 0,
            "repeat_mistake_count": 0,
            "episodic_correction": "Correction: for initials, take the first letter of each provided word in order. Do not invent a meaning or replay an old expansion.",
            "example_anchor": "Give the initials for North Valley Transit.",
            "procedural_rule": "derive initials by taking first letters in order; do not expose a memorized final answer",
            "evidence_gate": "uses only words present in the current prompt; does not include an invented expansion; does not expose final answer text from memory",
            "symbolic_key": "correction_slice:procedure:derive_initials",
            "hidden_full_path": null,
            "hidden_dim": null,
            "route_64d": null,
            "route_motif_id": null,
            "unicode_packet_id": null,
            "unicode_escape": null,
            "decoded_route_id": null,
            "route_preserved": null,
            "current_resolution_level": 2,
            "last_required_resolution_level": 2,
            "success_streak": 0,
            "failure_count": 0,
            "false_positive_count": 0,
            "created_at_ms": stamp,
            "updated_at_ms": stamp
        });
        let fact = serde_json::json!({
            "id": "gmms_compat:gmms_fact_runtime_label_v1",
            "domain": "gmms:semantic_correction_slice",
            "trigger_terms": ["jason", "runtime", "label", "current", "prototype", "notes"],
            "bad_reflex": "stale fact: Niodoo-alpha as Jason's current runtime label",
            "corrected_reflex": "current fact: Jason runtime label = Niodv4-control",
            "evidence_requirement": "suppresses stale label Niodoo-alpha when current label is requested; allows current factual value Niodv4-control",
            "rejected_surfaces": [
                "stale fact: Niodoo-alpha as Jason's current runtime label",
                "Niodoo-alpha is stale for the current runtime label",
                "Niodoo-alpha"
            ],
            "accepted_surfaces": ["Niodv4-control"],
            "schema": null,
            "allowed_actions": ["suppress_stale_path"],
            "confidence": 0.75,
            "action_level": 4,
            "decay_rate": 0.15,
            "success_count": 0,
            "repeat_mistake_count": 0,
            "episodic_correction": "Correction: Jason's current runtime label is Niodv4-control, not Niodoo-alpha.",
            "example_anchor": "What label should I put in Jason's notes?",
            "procedural_rule": "current fact: Jason runtime label = Niodv4-control",
            "evidence_gate": "suppresses stale label Niodoo-alpha when current label is requested; allows current factual value Niodv4-control",
            "symbolic_key": "correction_slice:personal_fact:runtime_label",
            "hidden_full_path": null,
            "hidden_dim": null,
            "route_64d": null,
            "route_motif_id": null,
            "unicode_packet_id": null,
            "unicode_escape": null,
            "decoded_route_id": null,
            "route_preserved": null,
            "current_resolution_level": 4,
            "last_required_resolution_level": 4,
            "success_streak": 0,
            "failure_count": 0,
            "false_positive_count": 0,
            "created_at_ms": stamp,
            "updated_at_ms": stamp
        });
        let path = std::env::temp_dir().join(format!("gmms_compat_load_shape_{}.jsonl", stamp));
        fs::write(&path, format!("{skill}\n{fact}\n")).unwrap();
        let memory = MistakeReflexMemory::load(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(memory.len(), 2);
        assert!(memory
            .events
            .iter()
            .all(|event| event.domain == "gmms:semantic_correction_slice"));
        assert!(memory.events.iter().any(|event| {
            event.id == "gmms_compat:gmms_skill_initials_v1"
                && event.accepted_surfaces.is_empty()
                && event
                    .allowed_actions
                    .contains(&"do_not_inject_final_answer".to_string())
                && event.schema.is_none()
                && event.symbolic_key.as_deref()
                    == Some("correction_slice:procedure:derive_initials")
        }));
        assert!(memory.events.iter().any(|event| {
            event.id == "gmms_compat:gmms_fact_runtime_label_v1"
                && event.accepted_surfaces == vec!["Niodv4-control".to_string()]
                && event.schema.is_none()
                && event.symbolic_key.as_deref()
                    == Some("correction_slice:personal_fact:runtime_label")
        }));

        let matches = memory.query("What current runtime label should Jason's notes use?", 3);
        assert_eq!(
            matches.first().map(|item| item.event_id.as_str()),
            Some("gmms_compat:gmms_fact_runtime_label_v1")
        );
        assert_eq!(matches[0].domain, "gmms:semantic_correction_slice");
        assert_eq!(matches[0].accepted_surfaces, vec!["Niodv4-control"]);
    }

    fn gmms_test_event(
        stamp: u128,
        id: &str,
        trigger_terms: &[&str],
        bad_reflex: &str,
        corrected_reflex: &str,
        evidence_requirement: &str,
        rejected_surfaces: &[&str],
        accepted_surfaces: &[&str],
        allowed_actions: &[&str],
        symbolic_key: &str,
        unicode_packet_id: Option<&str>,
    ) -> Value {
        serde_json::json!({
            "id": id,
            "domain": "gmms:semantic_correction_slice",
            "trigger_terms": trigger_terms,
            "bad_reflex": bad_reflex,
            "corrected_reflex": corrected_reflex,
            "evidence_requirement": evidence_requirement,
            "rejected_surfaces": rejected_surfaces,
            "accepted_surfaces": accepted_surfaces,
            "schema": null,
            "allowed_actions": allowed_actions,
            "confidence": 0.75,
            "action_level": 2,
            "decay_rate": 0.15,
            "success_count": 0,
            "repeat_mistake_count": 0,
            "episodic_correction": evidence_requirement,
            "example_anchor": "GMMS observe-only fixture",
            "procedural_rule": corrected_reflex,
            "evidence_gate": evidence_requirement,
            "symbolic_key": symbolic_key,
            "hidden_full_path": null,
            "hidden_dim": null,
            "route_64d": null,
            "route_motif_id": null,
            "unicode_packet_id": unicode_packet_id,
            "unicode_escape": null,
            "decoded_route_id": null,
            "route_preserved": null,
            "current_resolution_level": 2,
            "last_required_resolution_level": 2,
            "success_streak": 0,
            "failure_count": 0,
            "false_positive_count": 0,
            "created_at_ms": stamp,
            "updated_at_ms": stamp
        })
    }

    #[test]
    fn gmms_shelf_order_derives_earned_boundary_surface_without_prompt_answer() {
        let stamp = now_ms();
        let event: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "gmms_probe:procedure_shelf_order_v1",
            &[
                "shelf-order",
                "ordered list",
                "item list",
                "method",
                "items",
            ],
            "sort alphabetically; reuse prior ordered-list answer",
            "for shelf-order tasks, count the letters in each current prompt item, sort items by ascending letter count, and preserve input order only for equal counts",
            "uses only items in the current prompt; applies an ascending numeric letter-count sort to all items",
            &["sort alphabetically", "reuse prior ordered list"],
            &[],
            &["inject_short_reflex_hint", "do_not_inject_final_answer"],
            "correction_slice:procedure:shelf_order",
            None,
        ))
        .unwrap();
        let mut memory = MistakeReflexMemory::default();
        memory.events.push(event);

        let prompt = "Apply Jason's corrected shelf-order method to this item list: mint, violet, black, tan. Return only the ordered list.";
        let matches = memory.query(prompt, 3);

        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].accepted_surfaces,
            vec![
                "tan, mint, black, violet".to_string(),
                "tan / mint / black / violet".to_string(),
                "tan -> mint -> black -> violet".to_string(),
                "tan mint black violet".to_string(),
                "tan\nmint\nblack\nviolet".to_string(),
            ]
        );

        let applied = MistakeReflexMemory::apply_prompt(prompt, &matches, "text-hint");
        assert!(applied.contains("MISTAKE REFLEX"));
        assert!(!applied.contains("tan, mint, black, violet"));
        assert!(!applied.contains("tan / mint / black / violet"));

        let mut guard = MistakeReflexGuard::new(matches);
        let snapshot = guard.observe(
            12,
            "EXACT OUTPUT:\ntan, mint, black, violet\n\nThe current prompt items were ordered by length.",
        );

        assert!(snapshot.accepted_answer_candidate_seen);
        assert!(snapshot.evidence_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(
            snapshot.earned_answer_text.as_deref(),
            Some("tan, mint, black, violet")
        );
        assert_eq!(snapshot.earned_boundary_step, Some(12));
    }

    #[test]
    fn gmms_observe_only_summaries_and_runtime_matches_stay_separate_surfaces() {
        let stamp = now_ms();
        let rows = [
            gmms_test_event(
                stamp,
                "gmms_compat:gmms_skill_initials_v1",
                &["initials", "acronym", "first", "letter", "phrase", "derive"],
                "invent expansion; reuse stale abbreviation meaning",
                "derive initials by taking first letters in order; do not expose a memorized final answer",
                "uses only words present in the current prompt; does not expose final answer text from memory",
                &["invented abbreviation expansion or stale acronym meaning"],
                &[],
                &["inject_short_reflex_hint", "do_not_inject_final_answer"],
                "correction_slice:procedure:derive_initials",
                None,
            ),
            gmms_test_event(
                stamp,
                "gmms_compat:gmms_fact_runtime_label_v1",
                &["jason", "runtime", "label", "current", "prototype", "notes"],
                "stale fact: Niodoo-alpha as Jason's current runtime label",
                "current fact: Jason runtime label = Niodv4-control",
                "suppresses stale label Niodoo-alpha when current label is requested",
                &["Niodoo-alpha"],
                &["Niodv4-control"],
                &["suppress_stale_path"],
                "correction_slice:personal_fact:runtime_label",
                None,
            ),
            gmms_test_event(
                stamp,
                "gmms_compat:gmms_policy_raw_telemetry_v1",
                &["startup", "telemetry", "jsonl", "proof", "passed", "influence", "claim"],
                "claiming bridge influence from startup load or simulated telemetry alone",
                "require raw telemetry evidence before marking influence passed",
                "requires per-token JSONL or equivalent raw telemetry",
                &["startup log or simulated telemetry used as proof"],
                &[],
                &[
                    "require_evidence_before_lock",
                    "attach_route_unicode_sidecar_if_semantic_gate_passed",
                    "do_not_inject_final_answer",
                ],
                "correction_slice:verification_policy:raw_telemetry_required",
                Some("routepkt_policy_raw_telemetry_001"),
            ),
        ];
        let path =
            std::env::temp_dir().join(format!("gmms_observe_only_applicability_{}.jsonl", stamp));
        fs::write(
            &path,
            rows.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let memory = MistakeReflexMemory::load(&path).unwrap();
        let _ = fs::remove_file(&path);

        let skill = memory.observe_gmms_applicability(
            "For the phrase North Valley Transit, return its initials only.",
            3,
        );
        assert_eq!(
            skill.first().map(|item| item.event_id.as_str()),
            Some("gmms_compat:gmms_skill_initials_v1")
        );
        assert_eq!(skill[0].mode, "skill_reflex");
        assert_eq!(skill[0].allowed_action_max, "context_process_hint");
        assert_eq!(skill[0].accepted_surface_count, 0);
        assert!(!skill[0].final_answer_injection_allowed);

        let fact = memory.observe_gmms_applicability(
            "What current runtime label should Jason's project notes use?",
            3,
        );
        assert_eq!(
            fact.first().map(|item| item.event_id.as_str()),
            Some("gmms_compat:gmms_fact_runtime_label_v1")
        );
        assert_eq!(fact[0].mode, "fact_personal");
        assert_eq!(fact[0].allowed_action_max, "suppress_stale_path");
        assert_eq!(fact[0].accepted_surface_count, 1);
        let fact_telemetry = serde_json::to_string(&fact[0]).unwrap();
        assert!(!fact_telemetry.contains("Niodv4-control"));

        let policy = memory.observe_gmms_applicability(
            "Can startup logs alone prove bridge influence passed without JSONL telemetry?",
            3,
        );
        assert_eq!(
            policy.first().map(|item| item.event_id.as_str()),
            Some("gmms_compat:gmms_policy_raw_telemetry_v1")
        );
        assert_eq!(
            policy[0].allowed_action_max,
            "light_route_unicode_sidecar_use"
        );
        assert!(policy[0].route_unicode_sidecar_attached);
        assert_eq!(
            policy[0].unicode_packet_id.as_deref(),
            Some("routepkt_policy_raw_telemetry_001")
        );

        let active_matches = memory.query(
            "What current runtime label should Jason's project notes use?",
            3,
        );
        assert_eq!(
            active_matches.first().map(|item| item.event_id.as_str()),
            Some("gmms_compat:gmms_fact_runtime_label_v1")
        );
        assert!(
            active_matches[0]
                .accepted_surfaces
                .contains(&"Niodv4-control".to_string()),
            "runtime influence may carry explicit in-scope factual corrections"
        );

        let wrong_slice_control = memory.query(
            "Give me a status update for the current work. Keep it in the style Jason prefers by default.",
            3,
        );
        assert!(
            wrong_slice_control
                .iter()
                .all(|item| item.event_id != "gmms_compat:gmms_skill_initials_v1"),
            "common example-anchor words must not activate an unrelated GMMS skill slice"
        );
    }

    #[test]
    fn claims_section_tag_mixed_status_prompt_requires_own_rejected_surface() {
        let stamp = now_ms();
        let dv_pending: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:claims_section_tag:fact:53bc851b344c",
            &["10dv", "pending", "real-weak", "falsified"],
            "repeat unverified claim: §10dv PENDING",
            "CLAIMS.md ground truth: §10dv actual tag is REAL-WEAK→FALSIFIED",
            "CLAIMS.md ground truth: §10dv actual tag is REAL-WEAK→FALSIFIED",
            &["§10dv PENDING"],
            &["§10dv REAL-WEAK→FALSIFIED"],
            &["verify_ground_truth_before_answering"],
            "oracle:claims_section_tag",
            None,
        ))
        .unwrap();
        let eo_pending: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:claims_section_tag:fact:c4bcebfa679c",
            &["10eo", "pending", "bypass"],
            "repeat unverified claim: §10eo PENDING",
            "CLAIMS.md ground truth: §10eo actual tag is BYPASS",
            "CLAIMS.md ground truth: §10eo actual tag is BYPASS",
            &["§10eo PENDING"],
            &["§10eo BYPASS"],
            &["verify_ground_truth_before_answering"],
            "oracle:claims_section_tag",
            None,
        ))
        .unwrap();
        let eo_incomplete: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:claims_section_tag:fact:fb14ea47a670",
            &["10eo", "incomplete", "bypass"],
            "repeat unverified claim: §10eo INCOMPLETE",
            "CLAIMS.md ground truth: §10eo actual tag is BYPASS",
            "CLAIMS.md ground truth: §10eo actual tag is BYPASS",
            &["§10eo INCOMPLETE"],
            &["§10eo BYPASS"],
            &["verify_ground_truth_before_answering"],
            "oracle:claims_section_tag",
            None,
        ))
        .unwrap();
        let prompt =
            normalize_text("Two prior Niodoo outputs claimed §10dv PENDING and §10eo INCOMPLETE.");

        assert!(gmms_semantic_correction_match(&dv_pending, &prompt));
        assert!(gmms_semantic_correction_match(&eo_incomplete, &prompt));
        assert!(!gmms_semantic_correction_match(&eo_pending, &prompt));
    }

    #[test]
    fn claims_section_tag_alias_matches_section_plus_status_without_exact_surface() {
        let stamp = now_ms();
        let dv_pending: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:claims_section_tag:fact:53bc851b344c",
            &["10dv", "pending", "real-weak", "falsified"],
            "repeat unverified claim: §10dv PENDING",
            "CLAIMS.md ground truth: §10dv actual tag is REAL-WEAK→FALSIFIED",
            "CLAIMS.md ground truth: §10dv actual tag is REAL-WEAK→FALSIFIED",
            &["§10dv PENDING"],
            &["§10dv REAL-WEAK→FALSIFIED"],
            &["verify_ground_truth_before_answering"],
            "oracle:claims_section_tag",
            None,
        ))
        .unwrap();
        let eo_pending: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:claims_section_tag:fact:c4bcebfa679c",
            &["10eo", "pending", "bypass"],
            "repeat unverified claim: §10eo PENDING",
            "CLAIMS.md ground truth: §10eo actual tag is BYPASS",
            "CLAIMS.md ground truth: §10eo actual tag is BYPASS",
            &["§10eo PENDING"],
            &["§10eo BYPASS"],
            &["verify_ground_truth_before_answering"],
            "oracle:claims_section_tag",
            None,
        ))
        .unwrap();
        let paraphrased_prompt =
            normalize_text("Scratch note: §10dv is pending. What should I record now?");
        let neutral_prompt =
            normalize_text("Reviewer note: what is the current status tag for §10dv?");

        assert!(gmms_semantic_correction_match(
            &dv_pending,
            &paraphrased_prompt
        ));
        assert!(!gmms_semantic_correction_match(
            &eo_pending,
            &paraphrased_prompt
        ));
        assert!(!gmms_semantic_correction_match(
            &dv_pending,
            &neutral_prompt
        ));
    }

    #[test]
    fn path_exists_prompt_requires_own_rejected_surface() {
        let stamp = now_ms();
        let missing_path: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:path_exists:fact:phase_f_latest",
            &[
                "artifacts",
                "phase_f",
                "autonomous_miner",
                "stdout.txt",
                "does_not_exist",
            ],
            "repeat unverified claim: artifacts/phase_f/oracle_output_latest.txt",
            "The repo path artifacts/phase_f/oracle_output_latest.txt does not exist",
            "The repo path artifacts/phase_f/oracle_output_latest.txt does not exist",
            &["artifacts/phase_f/oracle_output_latest.txt"],
            &["artifacts/phase_f/oracle_output_latest.txt does not exist"],
            &["verify_ground_truth_before_answering"],
            "oracle:path_exists",
            None,
        ))
        .unwrap();

        let near_path_prompt = normalize_text(
            "A prior Niodoo output cited artifacts/codex_phase_f_autonomous_miner_min_vocab_20260506/seed_233/replay_with_ledger/stdout.txt as a repo path. What should the reviewer conclude?",
        );
        let exact_path_prompt = normalize_text(
            "A prior Niodoo output cited artifacts/phase_f/oracle_output_latest.txt as a repo path. What should the reviewer conclude?",
        );

        assert!(
            !gmms_semantic_correction_match(&missing_path, &near_path_prompt),
            "shared path vocabulary must not activate a path fact for a different rejected path"
        );
        assert!(gmms_semantic_correction_match(
            &missing_path,
            &exact_path_prompt
        ));
    }

    #[test]
    fn path_exists_alias_matches_short_path_components_with_path_context() {
        let stamp = now_ms();
        let claims_path: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:path_exists:fact:a2af68b0c4d5",
            &["artifacts", "claims", "does_not_exist"],
            "repeat unverified claim: artifacts/claims/",
            "The repo path artifacts/claims/ does not exist",
            "The repo path artifacts/claims/ does not exist",
            &["artifacts/claims/"],
            &["artifacts/claims/ does not exist"],
            &["verify_ground_truth_before_answering"],
            "oracle:path_exists",
            None,
        ))
        .unwrap();
        let other_path: MistakeReflexEvent = serde_json::from_value(gmms_test_event(
            stamp,
            "oracle:path_exists:fact:phase_f_latest",
            &[
                "artifacts",
                "phase_f",
                "oracle_output",
                "latest_report.txt",
                "does_not_exist",
            ],
            "repeat unverified claim: artifacts/phase_f/oracle_output/latest_report.txt",
            "The repo path artifacts/phase_f/oracle_output/latest_report.txt does not exist",
            "The repo path artifacts/phase_f/oracle_output/latest_report.txt does not exist",
            &["artifacts/phase_f/oracle_output/latest_report.txt"],
            &["artifacts/phase_f/oracle_output/latest_report.txt does not exist"],
            &["verify_ground_truth_before_answering"],
            "oracle:path_exists",
            None,
        ))
        .unwrap();
        let paraphrased_prompt = normalize_text(
            "Scratch note: there is an artifacts claims folder. What should I record now?",
        );
        let no_context_prompt = normalize_text("Compare the words artifacts and claims.");

        assert!(gmms_semantic_correction_match(
            &claims_path,
            &paraphrased_prompt
        ));
        assert!(!gmms_semantic_correction_match(
            &other_path,
            &paraphrased_prompt
        ));
        assert!(!gmms_semantic_correction_match(
            &claims_path,
            &no_context_prompt
        ));
    }

    #[test]
    fn gmms_entity_removal_hint_does_not_introduce_unmentioned_removed_name() {
        let stamp = now_ms();
        let entity = gmms_test_event(
            stamp,
            "gmms_compat:gmms_entity_remove_alex_v1",
            &[
                "alex",
                "assign",
                "owner",
                "collaborator",
                "project",
                "qa",
                "task",
                "stale",
                "removal",
                "assignment",
            ],
            "stale collaborator Alex as active assignee",
            "suppress Alex as active collaborator; ask or choose a non-stale owner if needed",
            "does not assign new work to Alex; asks if the active owner is uncertain",
            &[
                "stale collaborator Alex as active assignee",
                "assign task to Alex as if still active",
                "active collaborator",
            ],
            &[],
            &["ask_if_uncertain"],
            "correction_slice:entity_membership:collaborator_removed",
            None,
        );
        let path = std::env::temp_dir().join(format!("gmms_entity_redaction_{}.jsonl", stamp));
        fs::write(&path, format!("{entity}\n")).unwrap();
        let memory = MistakeReflexMemory::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        let prompt = "Assign the QA follow-up for this project. If the active owner is unclear, say what you need.";
        let matches = memory.query(prompt, 3);

        assert_eq!(
            matches.first().map(|item| item.event_id.as_str()),
            Some("gmms_compat:gmms_entity_remove_alex_v1")
        );
        let hinted = MistakeReflexMemory::apply_prompt(prompt, &matches, "text-hint");
        assert!(hinted.contains("MISTAKE REFLEX"));
        assert!(hinted.contains("removed collaborators"));
        assert!(!hinted.contains("Alex"));

        let explicit_prompt = "Should Alex own QA for this patch, or should someone else?";
        let explicit_matches = memory.query(explicit_prompt, 3);
        let explicit_hint =
            MistakeReflexMemory::apply_prompt(explicit_prompt, &explicit_matches, "text-hint");
        assert!(explicit_hint.contains("Alex"));
    }

    #[test]
    fn memory_slice_symbolic_counting_matches_heldout_schema_without_prompt_answer() {
        let mut memory = MistakeReflexMemory::default();
        let mut captured = memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in raspberry",
            Some("There are 2 Rs in raspberry."),
        );
        let mut event = captured.remove(0);
        event.id = "memory_slice_v1:symbolic_counting".to_string();
        event.trigger_terms = vec![
            "count".to_string(),
            "letter".to_string(),
            "letters".to_string(),
        ];
        event.accepted_surfaces.clear();
        event.schema = None;
        event.symbolic_key = Some("memory_slice_v1:symbolic_counting".to_string());
        memory.events.clear();
        memory.events.push(event);

        let matches = memory.query("Count the number of Ss in sassafras. Answer directly.", 3);
        assert_eq!(matches.len(), 1);
        assert_eq!(
            matches[0].schema,
            Some(ReflexTaskSchema::SymbolicCounting {
                word: "sassafras".to_string(),
                target_char: "s".to_string(),
                expected_count: 4,
            })
        );
        assert!(matches[0].accepted_surfaces.contains(&"4 ss".to_string()));

        let prompt = MistakeReflexMemory::apply_prompt(
            "Count the number of Ss in sassafras. Answer directly.",
            &matches,
            "text-hint",
        );
        assert!(prompt.contains("MISTAKE REFLEX"));
        assert!(!prompt.contains("4 ss"));
        assert!(!prompt.contains("there are 4"));
        assert!(
            prompt.contains("task_binding=word `sassafras`; target letter `s`; start count at 0")
        );
        assert!(prompt.contains("scan_entry_example=s=1 | a=1"));
        assert!(prompt.contains("never write placeholder words like char or running_total"));
        assert!(prompt.contains("ANSWER: <number> <target-letter>s"));
    }

    #[test]
    fn memory_slice_parallel_duration_suppresses_sequential_negative_match() {
        let mut memory = MistakeReflexMemory::default();
        let mut captured = memory.capture_from_correction_turn(
            "wrong, it takes 5 hours to dry all towels when they dry in parallel with enough space",
            Some("It takes 50 hours."),
        );
        let mut event = captured.remove(0);
        event.id = "memory_slice_v1:parallel_duration".to_string();
        event.trigger_terms = vec![
            "dry".to_string(),
            "drying".to_string(),
            "hours".to_string(),
            "more".to_string(),
            "space".to_string(),
        ];
        event.accepted_surfaces.clear();
        event.schema = None;
        event.symbolic_key = Some("memory_slice_v1:parallel_duration".to_string());
        memory.events.clear();
        memory.events.push(event);

        let heldout = memory.query(
            "If 1 blanket takes 5 hours to dry and I dry 3 more blankets with enough line space, how long will it take?",
            3,
        );
        assert_eq!(heldout.len(), 1);
        assert!(heldout[0]
            .accepted_surfaces
            .contains(&"5 hours".to_string()));
        let prompt = MistakeReflexMemory::apply_prompt(
            "If 1 blanket takes 5 hours to dry and I dry 3 more blankets with enough line space, how long will it take?",
            &heldout,
            "text-hint",
        );
        assert!(prompt.contains("classify as parallel drying, not sequential drying"));
        assert!(prompt.contains("single-item duration and unit"));
        assert!(!prompt.contains("CORRECT_ANSWER"));

        let sequential = memory.query(
            "If one blanket must dry after the previous blanket finishes, how long do 3 blankets take?",
            3,
        );
        assert!(sequential.is_empty());
    }

    #[test]
    fn evidence_gate_blocks_lock_until_evidence() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0, "VISIBLE ANSWER: 3 [REQUEST: LOCK]");

        assert!(snapshot.matched);
        assert!(!snapshot.evidence_seen);
        assert!(guard.should_block_finalization());
    }

    #[test]
    fn evidence_gate_allows_lock_after_scan() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(
            0,
            "VISIBLE MATH: S-T-R-A-W-B-E-R-R-Y. running count: r=1, r=2, r=3. VISIBLE ANSWER: 3.",
        );

        assert!(snapshot.evidence_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(snapshot.earned_answer_text.as_deref(), Some("3"));
        assert!(!guard.should_block_finalization());
    }

    #[test]
    fn earned_answer_requires_evidence_before_stopping_surface() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0, "FINAL ANSWER: There are 3 Rs in strawberry.");

        assert!(!snapshot.evidence_seen);
        assert!(!snapshot.earned_answer_seen);
        assert!(guard.should_block_finalization());
    }

    #[test]
    fn earned_answer_accepts_so_far_count_then_final_answer() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0,
            "S-T-R-A-W-B-E-R-R-Y. Working ANSWER: 2 Rs so far. Found another R. Working ANSWER: 3 Rs so far. FINAL ANSWER: There are 3 Rs",
        );

        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(snapshot.earned_answer_text.as_deref(), Some("3 rs"));
    }

    #[test]
    fn nearby_counting_prompt_does_not_match_strawberry_r_reflex() {
        let mut memory = MistakeReflexMemory::default();
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
    fn captures_generic_letter_count_reflex() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Es in bookkeeper",
            Some("There are 2 Es in bookkeeper."),
        );

        let matches = memory.query("Count the number of Es in bookkeeper. Answer directly.", 3);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].domain, "symbolic_counting:letter_count");
        assert!(matches[0]
            .accepted_surfaces
            .iter()
            .any(|surface| surface == "3 es"));
    }

    #[test]
    fn captures_plural_s_letter_count_reflex() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 4 Ss in mississippi",
            Some("VISIBLE COUNT: 1 S\nVISIBLE COUNT: 2 Ss\nVISIBLE COUNT: 3 Ss\nVISIBLE COUNT: 5 Ss\nFINAL ANSWER: 5 Ss"),
        );

        let matches = memory.query("Count the number of Ss in mississippi. Answer directly.", 3);

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].domain, "symbolic_counting:letter_count");
        assert!(matches[0]
            .accepted_surfaces
            .iter()
            .any(|surface| surface == "4 ss"));
        assert!(matches[0]
            .rejected_surfaces
            .iter()
            .any(|surface| surface == "5 ss"));
        assert!(!matches[0]
            .rejected_surfaces
            .iter()
            .any(|surface| surface == "1 s"));
    }

    #[test]
    fn captures_generic_parallel_drying_reflex() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all shirts when they dry in parallel",
            Some("The answer is 30 hours."),
        );

        let matches = memory.query(
            "If 1 shirt takes 5 hours to dry and I dry 5 more shirts, how long will it take?",
            3,
        );

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].domain, "parallel_duration:drying");
    }

    #[test]
    fn sequential_towel_prompt_does_not_match_parallel_reflex() {
        let mut memory = MistakeReflexMemory::default();
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
    fn parallel_towel_evidence_accepts_additive_same_time_language() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all towels when they dry in parallel",
            Some("The answer is 45 hours."),
        );
        let matches = memory.query(
            "If 1 towel takes 5 hours to dry and someone tells me to dry 9 more, how long will it take me now?",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0,
            "The task is additive, not multiplicative. The towels dry in the same amount of time. WORKING ANSWER: 5 hours.",
        );

        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(snapshot.earned_answer_text.as_deref(), Some("5 hours"));
    }

    #[test]
    fn parallel_retry_snap_back_is_earned_when_final_corrects() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all shirts when they dry in parallel with enough rack space",
            Some("The answer is 30 hours."),
        );
        let matches = memory.query(
            "If 1 shirt takes 5 hours to dry and I dry 5 more shirts with enough rack space, how long will it take?",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0,
            "Since I'm drying 5 more shirts, I need to calculate the total drying time.\n\
             [REQUEST: SPIKE]\n\
             The drying time for the remaining shirts doesn't increase the base time, as they can be dried in parallel.\n\
             WORKING ANSWER: 5 hours",
        );

        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(snapshot.earned_answer_text.as_deref(), Some("5 hours"));
    }

    #[test]
    fn old_path_after_earned_only_counts_post_boundary_drift() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all shirts when they dry in parallel with enough rack space",
            Some("The answer is 30 hours."),
        );
        let matches = memory.query(
            "If 1 shirt takes 5 hours to dry and I dry 5 more shirts with enough rack space, how long will it take?",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches);

        let first = guard.observe(
            8,
            "I might multiply the shirts, but enough rack space means parallel drying. WORKING ANSWER: 5 hours.",
        );
        let second = guard.observe(
            12,
            "I might multiply the shirts, but enough rack space means parallel drying. WORKING ANSWER: 5 hours. However, if I multiply 6 x 5, that is 30 hours.",
        );

        assert!(first.earned_answer_seen);
        assert!(!first.old_path_after_earned);
        assert!(second.old_path_after_earned);
        assert_eq!(second.earned_boundary_step, Some(8));
    }

    #[test]
    fn intermediate_so_far_count_is_not_old_mistake() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0,
            "S-T-R-A-W-B-E-R-R-Y. Working answer: 2 Rs so far. Found another R. Working answer: 3 Rs so far.",
        );

        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.evidence_seen);
    }

    #[test]
    fn streaming_partial_so_far_count_is_not_old_mistake() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0, "S-T-R-A-W-B-E-R-R-Y. Working answer: 2 Rs");

        assert!(!snapshot.old_mistake_seen);
    }

    #[test]
    fn symbolic_wobble_snap_back_is_earned_when_final_corrects() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 4 Ss in mississippi",
            Some("VISIBLE COUNT: 1 S\nVISIBLE COUNT: 2 Ss\nVISIBLE COUNT: 3 Ss\nVISIBLE COUNT: 5 Ss\nFINAL ANSWER: 5 Ss"),
        );
        let matches = memory.query("count the number of Ss in mississippi", 3);
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0,
            "m-i-s-s-i-s-s-i-p-p-i. VISIBLE COUNT: 1 S. VISIBLE COUNT: 2 Ss. VISIBLE COUNT: 3 Ss. VISIBLE COUNT: 4 Ss. VISIBLE COUNT: 5 Ss. VISIBLE COUNT: 4 Ss after recount. FINAL ANSWER: 4 Ss.",
        );

        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(snapshot.earned_answer_text.as_deref(), Some("4 ss"));
    }

    #[test]
    fn parallel_serial_multiplication_blocks_evidence() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "actually it takes 5 hours to dry all shirts when they dry in parallel with enough rack space",
            Some("The answer is 30 hours."),
        );
        let matches = memory.query(
            "If 1 shirt takes 5 hours to dry and I dry 5 more shirts with enough rack space, how long will it take?",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches);

        let snapshot = guard.observe(0,
            "Since I'm drying 5 more shirts, multiply 5 shirts * 5 hours/shirt = 25 hours. FINAL ANSWER: 30 hours.",
        );

        assert!(!snapshot.evidence_seen);
        assert!(snapshot.old_mistake_seen);
        assert!(!snapshot.earned_answer_seen);
    }

    #[test]
    fn repeated_mistake_escalates_action_level_but_success_decays_it() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let failed = guard.observe(0, "VISIBLE ANSWER: There are 2 Rs in strawberry.");

        assert!(memory.record_outcome(&matches, &failed));
        let escalated = memory.query("count the number of Rs in strawberry", 3);
        assert!(escalated[0].action_level >= 3);

        let mut guard = MistakeReflexGuard::new(escalated.clone());
        let passed = guard.observe(
            0,
            "VISIBLE MATH: S-T-R-A-W-B-E-R-R-Y. running count: r=1, r=2, r=3. VISIBLE ANSWER: 3.",
        );
        assert!(memory.record_outcome(&escalated, &passed));
        let decayed = memory.query("count the number of Rs in strawberry", 3);
        assert!(decayed[0].action_level <= escalated[0].action_level);
    }

    #[test]
    fn success_decays_resolution_after_streak_and_failure_unfolds() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let matches = memory.query("count the number of Rs in strawberry", 3);
        assert_eq!(matches[0].current_resolution_level, 2);

        let mut guard = MistakeReflexGuard::new(matches.clone());
        let passed = guard.observe(
            0,
            "VISIBLE MATH: S-T-R-A-W-B-E-R-R-Y. running count: r=1, r=2, r=3. VISIBLE ANSWER: 3.",
        );
        assert!(memory.record_outcome(&matches, &passed));
        let matches = memory.query("count the number of Rs in strawberry", 3);
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let passed = guard.observe(
            0,
            "VISIBLE MATH: S-T-R-A-W-B-E-R-R-Y. running count: r=1, r=2, r=3. VISIBLE ANSWER: 3.",
        );
        assert!(memory.record_outcome(&matches, &passed));
        let decayed = memory.query("count the number of Rs in strawberry", 3);
        assert_eq!(decayed[0].current_resolution_level, 1);

        let mut guard = MistakeReflexGuard::new(decayed.clone());
        let failed = guard.observe(0, "VISIBLE ANSWER: There are 2 Rs in strawberry.");
        assert!(memory.record_outcome(&decayed, &failed));
        let unfolded = memory.query("count the number of Rs in strawberry", 3);
        assert!(unfolded[0].current_resolution_level >= 2);
    }

    #[test]
    fn gmms_verification_policy_evidence_decays_and_relapse_unfolds() {
        let mut memory = MistakeReflexMemory::default();
        let mut policy = event(
            "gmms:semantic_correction_slice",
            &["startup", "telemetry", "jsonl", "claim", "bridge"],
            "claiming bridge influence from startup load or simulated telemetry alone",
            "for Niodoo claim review, require raw telemetry evidence before marking bridge influence passed",
            "reject startup-only bridge influence; require raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim",
            vec![
                "we can accept bridge-influence based only on startup load lines".to_string(),
                "startup logs alone prove bridge influence".to_string(),
            ],
            vec![
                "cannot accept bridge-influence based only on startup load lines".to_string(),
                "raw telemetry evidence".to_string(),
                "per-token JSONL".to_string(),
            ],
            None,
            &["require_evidence_before_lock", "do_not_inject_final_answer"],
        );
        policy.id = "gmms_compat:gmms_policy_raw_telemetry_lifecycle_test".to_string();
        policy.symbolic_key =
            Some("correction_slice:verification_policy:raw_telemetry_required".to_string());
        policy.action_level = 3;
        policy.current_resolution_level = 3;
        memory.upsert_event(policy);

        let matches = memory.query(
            "A teammate wants to accept bridge-influence based only on startup load lines and no token JSONL.",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let passed = guard.observe(
            0,
            "EXACT OUTPUT: We cannot accept bridge-influence based only on startup load lines; require raw telemetry evidence in per-token JSONL.",
        );
        assert!(passed.evidence_seen);
        assert!(!passed.old_mistake_seen);
        assert_eq!(passed.decay_reason.as_deref(), Some("evidence_seen"));
        assert!(memory.record_outcome(&matches, &passed));
        let decayed = memory.query(
            "A teammate wants to accept bridge-influence based only on startup load lines and no token JSONL.",
            3,
        );
        assert!(decayed[0].action_level < matches[0].action_level);

        let mut guard = MistakeReflexGuard::new(decayed.clone());
        let failed = guard.observe(
            0,
            "EXACT OUTPUT: We can accept bridge-influence based only on startup load lines.",
        );
        assert!(failed.old_mistake_seen);
        assert_eq!(failed.unfold_reason.as_deref(), Some("old_mistake_seen"));
        assert!(memory.record_outcome(&decayed, &failed));
        let unfolded = memory.query(
            "A teammate wants to accept bridge-influence based only on startup load lines and no token JSONL.",
            3,
        );
        assert!(unfolded[0].action_level > decayed[0].action_level);
        assert!(unfolded[0].current_resolution_level >= decayed[0].current_resolution_level);
    }

    #[test]
    fn gmms_verification_policy_earns_boundary_before_generic_working_answer_marker() {
        let mut memory = MistakeReflexMemory::default();
        let mut policy = event(
            "gmms:semantic_correction_slice",
            &["startup", "telemetry", "jsonl", "claim", "bridge"],
            "claiming bridge influence from startup load or simulated telemetry alone",
            "for Niodoo claim review, require raw telemetry evidence before marking bridge influence passed",
            "reject startup-only bridge influence; require raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim",
            vec![
                "we can accept bridge-influence based only on startup load lines".to_string(),
                "startup logs alone prove bridge influence".to_string(),
            ],
            vec![
                "cannot accept bridge-influence based only on startup load lines".to_string(),
                "raw telemetry evidence".to_string(),
                "per-token JSONL".to_string(),
            ],
            None,
            &["require_evidence_before_lock", "do_not_inject_final_answer"],
        );
        policy.id = "gmms_compat:gmms_policy_raw_telemetry_boundary_test".to_string();
        policy.symbolic_key =
            Some("correction_slice:verification_policy:raw_telemetry_required".to_string());
        policy.action_level = 3;
        policy.current_resolution_level = 3;
        memory.upsert_event(policy);

        let matches = memory.query(
            "Review this Niodoo claim note: bridge_enabled=true appeared at startup. Should CLAIMS move GREEN?",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let snapshot = guard.observe(
            34,
            "VISIBLE REASONING: The note requires raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim. Based on the evidence standard, we should reject the claim because it does not meet the required evidence. [WORKING ANSWER:] CLAIM",
        );
        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(
            snapshot.earned_answer_text.as_deref(),
            Some("per-token JSONL")
        );
        assert_eq!(snapshot.earned_boundary_step, Some(34));
    }

    #[test]
    fn gmms_artifact_triage_reject_move_surface_earns_boundary() {
        let mut memory = MistakeReflexMemory::default();
        let mut policy = event(
            "gmms:semantic_correction_slice",
            &["startup", "telemetry", "jsonl", "claim", "bridge"],
            "claiming bridge influence from startup load or simulated telemetry alone",
            "for Niodoo claim review, require raw telemetry evidence before marking bridge influence passed",
            "reject startup-only bridge influence; require raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim",
            vec![
                "we can accept bridge-influence based only on startup load lines".to_string(),
                "startup logs alone prove bridge influence".to_string(),
            ],
            vec![
                "cannot accept bridge-influence based only on startup load lines".to_string(),
                "raw telemetry evidence".to_string(),
                "per-token JSONL".to_string(),
            ],
            None,
            &["require_evidence_before_lock", "do_not_inject_final_answer"],
        );
        policy.id = "gmms_compat:gmms_policy_raw_telemetry_artifact_triage_test".to_string();
        policy.symbolic_key =
            Some("correction_slice:verification_policy:raw_telemetry_required".to_string());
        policy.action_level = 3;
        policy.current_resolution_level = 3;
        memory.upsert_event(policy);

        let matches = memory.query(
            "The artifact card says bridge influence should move GREEN from startup logs.",
            3,
        );
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let snapshot = guard.observe(
            12,
            "EXACT OUTPUT: reject the move of bridge influence GREEN. The collector summary and startup logs are not sufficient to justify moving bridge influence GREEN. The procedure requires raw per-token JSONL telemetry or equivalent generated-output telemetry.",
        );
        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(
            snapshot.earned_answer_text.as_deref(),
            Some("per-token JSONL")
        );
        assert_eq!(snapshot.earned_boundary_step, Some(12));
    }

    #[test]
    fn gmms_artifact_triage_should_not_update_surface_earns_boundary() {
        let mut memory = MistakeReflexMemory::default();
        let mut policy = event(
            "gmms:semantic_correction_slice",
            &["startup", "telemetry", "jsonl", "claim", "bridge"],
            "claiming bridge influence from startup load or simulated telemetry alone",
            "for Niodoo claim review, require raw telemetry evidence before marking bridge influence passed",
            "reject startup-only bridge influence; require raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim",
            vec![
                "we can accept bridge-influence based only on startup load lines".to_string(),
                "startup logs alone prove bridge influence".to_string(),
            ],
            vec![
                "cannot accept bridge-influence based only on startup load lines".to_string(),
                "raw telemetry evidence".to_string(),
                "per-token JSONL".to_string(),
            ],
            None,
            &["require_evidence_before_lock", "do_not_inject_final_answer"],
        );
        policy.id = "gmms_compat:gmms_policy_raw_telemetry_no_update_test".to_string();
        policy.symbolic_key =
            Some("correction_slice:verification_policy:raw_telemetry_required".to_string());
        policy.action_level = 1;
        policy.current_resolution_level = 3;
        memory.upsert_event(policy);

        let matches = memory.query(
            "The artifact card recommends updating CLAIMS to bridge_influence=GREEN from a collector summary and startup bridge_enabled=true, with no raw per-token JSONL telemetry.",
            3,
        );
        assert!(!matches.is_empty());
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let snapshot = guard.observe(
            20,
            "The claim should not be updated to bridge_influence=GREEN because the required evidence, such as raw per-token JSONL telemetry or generated-output review, is missing.",
        );
        assert!(snapshot.evidence_seen);
        assert!(!snapshot.old_mistake_seen);
        assert!(snapshot.earned_answer_seen);
        assert_eq!(
            snapshot.earned_answer_text.as_deref(),
            Some("per-token JSONL")
        );
        assert_eq!(snapshot.earned_boundary_step, Some(20));
    }

    #[test]
    fn gmms_verification_policy_red_without_evidence_is_old_mistake() {
        let mut memory = MistakeReflexMemory::default();
        let mut policy = event(
            "gmms:semantic_correction_slice",
            &["startup", "telemetry", "jsonl", "claim", "bridge"],
            "claiming bridge influence from startup load or simulated telemetry alone",
            "for Niodoo claim review, require raw telemetry evidence before marking bridge influence passed",
            "reject startup-only bridge influence; require raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim",
            vec![
                "we can accept bridge-influence based only on startup load lines".to_string(),
                "startup logs alone prove bridge influence".to_string(),
            ],
            vec![
                "cannot accept bridge-influence based only on startup load lines".to_string(),
                "raw telemetry evidence".to_string(),
                "per-token JSONL".to_string(),
            ],
            None,
            &["require_evidence_before_lock", "do_not_inject_final_answer"],
        );
        policy.id = "gmms_compat:gmms_policy_raw_telemetry_red_guard_test".to_string();
        policy.symbolic_key =
            Some("correction_slice:verification_policy:raw_telemetry_required".to_string());
        policy.action_level = 1;
        policy.current_resolution_level = 3;
        memory.upsert_event(policy);

        let matches = memory.query(
            "The artifact card recommends updating CLAIMS to bridge_influence=GREEN from a collector summary and startup bridge_enabled=true, with no raw per-token JSONL telemetry.",
            3,
        );
        assert!(!matches.is_empty());
        let mut guard = MistakeReflexGuard::new(matches.clone());
        let snapshot = guard.observe(7, "EXACT OUTPUT: CLAIMS: bridge_influence=RED");
        assert!(!snapshot.evidence_seen);
        assert!(snapshot.old_mistake_seen);
        assert_eq!(snapshot.unfold_reason.as_deref(), Some("old_mistake_seen"));
        assert!(!snapshot.earned_answer_seen);
    }

    #[test]
    fn gmms_verification_policy_prompt_guards_red_overcorrection() {
        let mut memory = MistakeReflexMemory::default();
        let mut policy = event(
            "gmms:semantic_correction_slice",
            &["startup", "telemetry", "jsonl", "claim", "bridge"],
            "claiming bridge influence from startup load or simulated telemetry alone",
            "for Niodoo claim review, require raw telemetry evidence before marking bridge influence passed",
            "reject startup-only bridge influence; require raw per-token JSONL telemetry or equivalent generated-output telemetry before moving a claim",
            vec![
                "we can accept bridge-influence based only on startup load lines".to_string(),
                "startup logs alone prove bridge influence".to_string(),
            ],
            vec![
                "cannot accept bridge-influence based only on startup load lines".to_string(),
                "raw telemetry evidence".to_string(),
                "per-token JSONL".to_string(),
            ],
            None,
            &["require_evidence_before_lock", "do_not_inject_final_answer"],
        );
        policy.id = "gmms_compat:gmms_policy_raw_telemetry_prompt_guard_test".to_string();
        policy.symbolic_key =
            Some("correction_slice:verification_policy:raw_telemetry_required".to_string());
        policy.action_level = 1;
        policy.current_resolution_level = 3;
        memory.upsert_event(policy);

        let user_prompt =
            "The artifact card recommends updating CLAIMS to bridge_influence=GREEN from a collector summary and startup bridge_enabled=true, with no raw per-token JSONL telemetry.";
        let matches = memory.query(user_prompt, 3);
        assert!(!matches.is_empty());
        let wrapped = MistakeReflexMemory::apply_prompt(user_prompt, &matches, "text-hint");
        assert!(wrapped.contains("missing evidence blocks GREEN but does not prove RED/FALSIFIED"));
        assert!(wrapped.contains("raw per-token JSONL/generated-output"));
    }

    #[test]
    fn attaches_unicode_packet_slice_to_matching_reflex() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let packet = serde_json::json!({
            "packet_id": "ump::abc::0001",
            "source": {
                "prompt": "[count task] strawberry -> count R letters -> ?",
                "source_artifact": "/tmp/hidden.f32"
            },
            "codec": {"unicode_escape": "\\u0006\\u00bf"},
            "geometry": {
                "original_route": {"motif_id": "route::count"},
                "decoded_route": {"motif_id": "route::count"},
                "route_preserved": true
            },
            "vectors": {"decoded_64d": vec![0.1; 64]}
        })
        .to_string();

        let path =
            std::env::temp_dir().join(format!("mistake_reflex_packet_test_{}.jsonl", now_ms()));
        fs::write(&path, packet).unwrap();
        let attached = memory
            .attach_vector_slices_from_packet_index(&path)
            .unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(attached, 1);
        let matches = memory.query("count the number of Rs in strawberry", 3);
        assert!(matches[0].vector_slice_available);
        assert_eq!(
            matches[0].unicode_packet_id.as_deref(),
            Some("ump::abc::0001")
        );
        assert_eq!(matches[0].route_preserved, Some(true));
    }

    #[test]
    fn text_hint_hidden_packet_keeps_packet_in_match_but_not_prompt_surface() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let packet = serde_json::json!({
            "packet_id": "ump::hidden::0001",
            "source": {
                "prompt": "[count task] strawberry -> count R letters -> ?",
                "source_artifact": "/tmp/hidden.f32"
            },
            "codec": {"unicode_escape": "\\u0006\\u00bf"},
            "geometry": {
                "original_route": {"motif_id": "route::count"},
                "decoded_route": {"motif_id": "route::count"},
                "route_preserved": true
            },
            "vectors": {"decoded_64d": vec![0.1; 64]}
        })
        .to_string();

        let path = std::env::temp_dir().join(format!(
            "mistake_reflex_hidden_packet_test_{}.jsonl",
            now_ms()
        ));
        fs::write(&path, packet).unwrap();
        memory
            .attach_vector_slices_from_packet_index(&path)
            .unwrap();
        let _ = fs::remove_file(&path);

        let matches = memory.query("count the number of Rs in strawberry", 3);
        assert!(matches[0].vector_slice_available);
        assert_eq!(
            matches[0].unicode_packet_id.as_deref(),
            Some("ump::hidden::0001")
        );
        let prompt = MistakeReflexMemory::apply_prompt(
            "count the number of Rs in strawberry",
            &matches,
            "text-hint-hidden-packet",
        );

        assert!(prompt.contains("MISTAKE REFLEX"));
        assert!(prompt.contains("running count"));
        assert!(!prompt.contains("route_slice=available"));
        assert!(!prompt.contains("ump::hidden::0001"));
        assert!(!prompt.contains("preserved=true"));
    }

    #[test]
    fn attaches_packet_slice_by_domain_family_metadata() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        memory.capture_from_correction_turn(
            "wrong, towels still take 4 hours when they dry with enough line space",
            Some("Four towels take 16 hours."),
        );
        let packet_rows = [
            serde_json::json!({
                "packet_id": "ump::symbolic::counting",
                "memory": {
                    "domain": "symbolic_counting:letter_count",
                    "family": "symbolic_counting"
                },
                "source": {"prompt": "domain packet selected by metadata"},
                "codec": {"unicode_escape": "\\ue001"},
                "geometry": {
                    "original_route": {"motif_id": "live_hidden_60k::raw64::000::counting_trap"},
                    "decoded_route": {"motif_id": "live_hidden_60k::raw64::000::counting_trap"},
                    "route_preserved": true
                }
            }),
            serde_json::json!({
                "packet_id": "ump::parallel::duration",
                "memory": {
                    "domain": "parallel_duration:drying",
                    "family": "parallel_duration"
                },
                "source": {"prompt": "domain packet selected by metadata"},
                "codec": {"unicode_escape": "\\ue002"},
                "geometry": {
                    "original_route": {"motif_id": "live_hidden_60k::raw64::004::parallel_time_trap"},
                    "decoded_route": {"motif_id": "live_hidden_60k::raw64::004::parallel_time_trap"},
                    "route_preserved": true
                }
            }),
        ]
        .into_iter()
        .map(|row| row.to_string())
        .collect::<Vec<_>>()
        .join("\n");

        let path = std::env::temp_dir().join(format!(
            "mistake_reflex_domain_packet_test_{}.jsonl",
            now_ms()
        ));
        fs::write(&path, format!("{packet_rows}\n")).unwrap();
        let attached = memory
            .attach_vector_slices_from_packet_index(&path)
            .unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(attached, 2);
        let count_matches = memory.query("Count the number of Rs in strawberry.", 3);
        assert!(count_matches[0].vector_slice_available);
        assert_eq!(
            count_matches[0].unicode_packet_id.as_deref(),
            Some("ump::symbolic::counting")
        );
        let parallel_matches = memory.query(
            "If 1 towel takes 4 hours to dry and I dry 3 more towels with enough line space, how long will it take?",
            3,
        );
        assert!(parallel_matches[0].vector_slice_available);
        assert_eq!(
            parallel_matches[0].unicode_packet_id.as_deref(),
            Some("ump::parallel::duration")
        );
        assert_eq!(parallel_matches[0].route_preserved, Some(true));
    }

    #[test]
    fn gmms_packet_metadata_attaches_by_event_identity() {
        let stamp = now_ms();
        let skill = gmms_test_event(
            stamp,
            "gmms_compat:gmms_skill_initials_v1",
            &["initials", "acronym", "first", "letter", "phrase", "derive"],
            "invent expansion; reuse stale abbreviation meaning",
            "derive initials by taking first letters in order; do not expose a memorized final answer",
            "uses only words present in the current prompt; does not expose final answer text from memory",
            &["invented abbreviation expansion or stale acronym meaning"],
            &[],
            &["inject_short_reflex_hint", "do_not_inject_final_answer"],
            "correction_slice:procedure:derive_initials",
            None,
        );
        let fact = gmms_test_event(
            stamp,
            "gmms_compat:gmms_fact_runtime_label_v1",
            &["jason", "runtime", "label", "current", "prototype", "notes"],
            "stale fact: Niodoo-alpha as Jason's current runtime label",
            "current fact: Jason runtime label = Niodv4-control",
            "suppresses stale label Niodoo-alpha when current label is requested",
            &["Niodoo-alpha"],
            &["Niodv4-control"],
            &["suppress_stale_path"],
            "correction_slice:personal_fact:runtime_label",
            None,
        );
        let memory_path =
            std::env::temp_dir().join(format!("gmms_packet_identity_memory_{}.jsonl", stamp));
        fs::write(&memory_path, format!("{skill}\n{fact}\n")).unwrap();
        let mut memory = MistakeReflexMemory::load(&memory_path).unwrap();
        let _ = fs::remove_file(&memory_path);

        let packet_rows = [
            serde_json::json!({
                "packet_id": "routepkt_gmms_skill_initials_v1",
                "memory": {
                    "domain": "gmms:semantic_correction_slice",
                    "family": "gmms_semantic_correction_slice",
                    "event_id": "gmms_compat:gmms_skill_initials_v1",
                    "symbolic_key": "correction_slice:procedure:derive_initials",
                    "slice_id": "gmms_skill_initials_v1"
                },
                "codec": {"unicode_escape": "\\u0053\\u0048\\u004c"},
                "geometry": {
                    "original_route": {"motif_id": "route:gmms:procedure:derive-initials"},
                    "decoded_route": {"motif_id": "route:gmms:procedure:derive-initials"}
                }
            }),
            serde_json::json!({
                "packet_id": "routepkt_gmms_fact_runtime_label_v1",
                "memory": {
                    "domain": "gmms:semantic_correction_slice",
                    "family": "gmms_semantic_correction_slice",
                    "event_id": "gmms_compat:gmms_fact_runtime_label_v1",
                    "symbolic_key": "correction_slice:personal_fact:runtime_label",
                    "slice_id": "gmms_fact_runtime_label_v1"
                },
                "codec": {"unicode_escape": "\\u004e\\u0056\\u0034"},
                "geometry": {
                    "original_route": {"motif_id": "route:gmms:personal-fact:runtime-label"},
                    "decoded_route": {"motif_id": "route:gmms:personal-fact:runtime-label"}
                }
            }),
        ]
        .into_iter()
        .map(|row| row.to_string())
        .collect::<Vec<_>>()
        .join("\n");

        let packet_path =
            std::env::temp_dir().join(format!("gmms_packet_identity_index_{}.jsonl", stamp));
        fs::write(&packet_path, format!("{packet_rows}\n")).unwrap();
        let attached = memory
            .attach_vector_slices_from_packet_index(&packet_path)
            .unwrap();
        let _ = fs::remove_file(&packet_path);

        assert_eq!(attached, 2);
        let skill_matches = memory.query("Derive initials for Silver Harbor Logistics.", 3);
        assert_eq!(
            skill_matches[0].unicode_packet_id.as_deref(),
            Some("routepkt_gmms_skill_initials_v1")
        );
        let fact_matches = memory.query("What current runtime label should Jason's notes use?", 3);
        assert_eq!(
            fact_matches[0].unicode_packet_id.as_deref(),
            Some("routepkt_gmms_fact_runtime_label_v1")
        );
    }

    #[test]
    fn wrong_family_packet_metadata_is_not_attached_by_prompt_fallback() {
        let mut memory = MistakeReflexMemory::default();
        memory.capture_from_correction_turn(
            "wrong, there are 3 Rs in strawberry",
            Some("There are 2 Rs in strawberry."),
        );
        let packet = serde_json::json!({
            "packet_id": "ump::wrong_family::parallel",
            "memory": {
                "domain": "parallel_duration:drying",
                "family": "parallel_duration"
            },
            "source": {
                "prompt": "Count the number of Rs in strawberry."
            },
            "codec": {"unicode_escape": "\\ue099"},
            "geometry": {
                "original_route": {"motif_id": "live_hidden_60k::raw64::004::parallel_time_trap"},
                "decoded_route": {"motif_id": "live_hidden_60k::raw64::004::parallel_time_trap"},
                "route_preserved": true
            }
        })
        .to_string();

        let path = std::env::temp_dir().join(format!(
            "mistake_reflex_wrong_family_packet_test_{}.jsonl",
            now_ms()
        ));
        fs::write(&path, packet).unwrap();
        let attached = memory
            .attach_vector_slices_from_packet_index(&path)
            .unwrap();
        let _ = fs::remove_file(&path);

        assert_eq!(attached, 0);
        let matches = memory.query("Count the number of Rs in strawberry.", 3);
        assert!(!matches[0].vector_slice_available);
        assert_eq!(matches[0].unicode_packet_id, None);
        assert_eq!(matches[0].route_preserved, None);
    }

    #[test]
    fn unicode_transport_artifact_replay_loads_and_reattaches_packets() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact_root = root.join("artifacts/unicode_memory_transport_slice_20260501");
        let memory_path = artifact_root.join("memory_transport_events.jsonl");
        let packet_index_path = artifact_root.join("attached_packet_index.jsonl");
        let replay_path = artifact_root.join("runtime_artifact_replay.json");
        if !memory_path.exists() || !packet_index_path.exists() || !replay_path.exists() {
            return;
        }
        let replay: Value =
            serde_json::from_str(&fs::read_to_string(&replay_path).unwrap()).unwrap();
        let expected_queries = replay
            .get("query_results")
            .and_then(Value::as_array)
            .expect("runtime artifact replay must include query_results");

        let mut memory = MistakeReflexMemory::load(&memory_path).unwrap();
        assert_eq!(memory.len(), 2);

        for event in &mut memory.events {
            event.hidden_full_path = None;
            event.hidden_dim = None;
            event.route_64d = None;
            event.route_motif_id = None;
            event.unicode_packet_id = None;
            event.unicode_escape = None;
            event.decoded_route_id = None;
            event.route_preserved = None;
        }

        let attached = memory
            .attach_vector_slices_from_packet_index(&packet_index_path)
            .unwrap();
        assert_eq!(attached, 2);

        let count_matches = memory.query("Count the number of Rs in strawberry.", 3);
        assert_eq!(count_matches.len(), 1);
        assert_match_matches_replay_query(
            &count_matches[0],
            expected_queries,
            "unicode_transport_slice:symbolic_counting",
        );

        let parallel_matches = memory.query(
            "If 1 towel takes 4 hours to dry and I dry 3 more towels with enough line space, how long will it take?",
            3,
        );
        assert_eq!(parallel_matches.len(), 1);
        assert_match_matches_replay_query(
            &parallel_matches[0],
            expected_queries,
            "unicode_transport_slice:parallel_duration",
        );
    }

    fn assert_match_matches_replay_query(
        actual: &MistakeReflexMatch,
        expected_queries: &[Value],
        event_id: &str,
    ) {
        let expected = expected_queries
            .iter()
            .find(|row| row.get("event_id").and_then(Value::as_str) == Some(event_id))
            .expect("missing artifact replay query row for event");
        let expected_packet_ids = expected
            .get("unicode_packet_ids")
            .and_then(Value::as_array)
            .expect("artifact replay row must include unicode_packet_ids");
        let expected_packet_id = expected_packet_ids
            .first()
            .and_then(Value::as_str)
            .expect("artifact replay row must include a packet id");

        assert_eq!(actual.event_id, event_id);
        assert_eq!(
            actual.domain,
            expected
                .get("domain")
                .and_then(Value::as_str)
                .expect("artifact replay row must include domain")
        );
        assert_eq!(
            actual.unicode_packet_id.as_deref(),
            Some(expected_packet_id)
        );
        assert_eq!(
            actual.decoded_route_id.as_deref(),
            expected.get("decoded_route_id").and_then(Value::as_str)
        );
        assert_eq!(
            actual.route_preserved,
            expected.get("route_preserved").and_then(Value::as_bool)
        );
        assert_eq!(
            actual.vector_slice_available,
            expected
                .get("vector_slice_available")
                .and_then(Value::as_bool)
                .expect("artifact replay row must include vector_slice_available")
        );
        assert_eq!(
            actual.action_level,
            expected
                .get("action_level")
                .and_then(Value::as_u64)
                .expect("artifact replay row must include action_level") as u8
        );
        assert_eq!(
            actual.current_resolution_level,
            expected
                .get("resolution_level")
                .and_then(Value::as_u64)
                .expect("artifact replay row must include resolution_level") as u8
        );
    }

    #[test]
    fn unicode_transport_artifact_replay_rejects_cross_family_packets() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact_root = root.join("artifacts/unicode_memory_transport_slice_20260501");
        let memory_path = artifact_root.join("memory_transport_events.jsonl");
        let packet_index_path = artifact_root.join("attached_packet_index.jsonl");
        if !memory_path.exists() || !packet_index_path.exists() {
            return;
        }

        let mut memory = MistakeReflexMemory::load(&memory_path).unwrap();
        assert_eq!(memory.len(), 2);

        for event in &mut memory.events {
            event.hidden_full_path = None;
            event.hidden_dim = None;
            event.route_64d = None;
            event.route_motif_id = None;
            event.unicode_packet_id = None;
            event.unicode_escape = None;
            event.decoded_route_id = None;
            event.route_preserved = None;
        }

        let packet_text = fs::read_to_string(&packet_index_path).unwrap();
        let mut swapped_rows = Vec::new();
        for line in packet_text.lines().filter(|line| !line.trim().is_empty()) {
            let mut value: Value = serde_json::from_str(line).unwrap();
            let family = value
                .pointer("/memory/family")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let (wrong_domain, wrong_family) = match family {
                "symbolic_counting" => ("parallel_duration:drying", "parallel_duration"),
                "parallel_duration" => ("symbolic_counting:letter_count", "symbolic_counting"),
                other => panic!("unexpected artifact packet family {other}"),
            };
            value["memory"]["domain"] = Value::String(wrong_domain.to_string());
            value["memory"]["family"] = Value::String(wrong_family.to_string());
            value["selection"]["target_domain"] = Value::String(wrong_domain.to_string());
            value["selection"]["target_family"] = Value::String(wrong_family.to_string());
            swapped_rows.push(value.to_string());
        }
        assert_eq!(swapped_rows.len(), 2);

        let wrong_index_path = std::env::temp_dir().join(format!(
            "mistake_reflex_artifact_wrong_family_packet_test_{}.jsonl",
            now_ms()
        ));
        fs::write(&wrong_index_path, format!("{}\n", swapped_rows.join("\n"))).unwrap();
        let attached = memory
            .attach_vector_slices_from_packet_index(&wrong_index_path)
            .unwrap();
        let _ = fs::remove_file(&wrong_index_path);

        assert_eq!(attached, 0);

        let count_matches = memory.query("Count the number of Rs in strawberry.", 3);
        assert_eq!(count_matches.len(), 1);
        assert!(!count_matches[0].vector_slice_available);
        assert_eq!(count_matches[0].unicode_packet_id, None);
        assert_eq!(count_matches[0].route_preserved, None);

        let parallel_matches = memory.query(
            "If 1 towel takes 4 hours to dry and I dry 3 more towels with enough line space, how long will it take?",
            3,
        );
        assert_eq!(parallel_matches.len(), 1);
        assert!(!parallel_matches[0].vector_slice_available);
        assert_eq!(parallel_matches[0].unicode_packet_id, None);
        assert_eq!(parallel_matches[0].route_preserved, None);
    }

    #[test]
    fn unicode_transport_artifact_replay_rejects_missing_vector_slice_metadata() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact_root = root.join("artifacts/unicode_memory_transport_slice_20260501");
        let memory_path = artifact_root.join("memory_transport_events.jsonl");
        let replay_path = artifact_root.join("runtime_artifact_missing_slice_negative_replay.json");
        if !memory_path.exists() || !replay_path.exists() {
            return;
        }
        let replay: Value =
            serde_json::from_str(&fs::read_to_string(&replay_path).unwrap()).unwrap();
        let mutated_event_id = replay
            .get("mutated_event_id")
            .and_then(Value::as_str)
            .expect("missing-slice replay must include mutated_event_id");
        let preserved_packet_id = replay
            .get("preserved_packet_id")
            .and_then(Value::as_str)
            .expect("missing-slice replay must include preserved_packet_id");
        let expected_query = replay
            .get("query_results")
            .and_then(Value::as_array)
            .and_then(|rows| rows.first())
            .expect("missing-slice replay must include query_results");

        let mut memory = MistakeReflexMemory::load(&memory_path).unwrap();
        assert_eq!(memory.len(), 2);
        let event = memory
            .events
            .iter_mut()
            .find(|event| event.id == mutated_event_id)
            .expect("artifact memory must include mutated event");
        assert_eq!(
            event.unicode_packet_id.as_deref(),
            Some(preserved_packet_id)
        );
        event.unicode_escape = None;
        event.route_motif_id = None;
        event.decoded_route_id = None;
        event.route_preserved = None;

        let matches = memory.query("Count the number of Rs in strawberry.", 3);
        assert_eq!(matches.len(), 1);
        let actual = &matches[0];
        assert_eq!(actual.event_id, mutated_event_id);
        assert_eq!(
            actual.unicode_packet_id.as_deref(),
            Some(preserved_packet_id)
        );
        assert_eq!(
            actual.vector_slice_available,
            expected_query
                .get("vector_slice_available_after_missing_slice_replay")
                .and_then(Value::as_bool)
                .expect("missing-slice replay row must include vector_slice_available")
        );
        assert!(!actual.vector_slice_available);
        assert_eq!(actual.route_preserved, None);
        assert_eq!(
            expected_query.get("route_preserved_after_missing_slice_replay"),
            Some(&Value::Null)
        );
    }

    #[test]
    fn unicode_transport_artifact_replay_does_not_silently_repair_stale_packet_id() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let artifact_root = root.join("artifacts/unicode_memory_transport_slice_20260501");
        let memory_path = artifact_root.join("memory_transport_events.jsonl");
        let packet_index_path = artifact_root.join("attached_packet_index.jsonl");
        let guard_path = artifact_root.join("artifact_reader_reattach_guard_smoke.json");
        if !memory_path.exists() || !packet_index_path.exists() || !guard_path.exists() {
            return;
        }
        let guard: Value = serde_json::from_str(&fs::read_to_string(&guard_path).unwrap()).unwrap();

        let mut memory = MistakeReflexMemory::load(&memory_path).unwrap();
        assert_eq!(memory.len(), 2);
        for event in &mut memory.events {
            assert!(event.unicode_packet_id.is_some());
            event.unicode_escape = None;
            event.route_motif_id = None;
            event.decoded_route_id = None;
            event.route_preserved = None;
        }

        let attached = memory
            .attach_vector_slices_from_packet_index(&packet_index_path)
            .unwrap();
        assert_eq!(
            attached,
            guard
                .get("stale_default_attach_count")
                .and_then(Value::as_u64)
                .expect("reattach guard must include stale_default_attach_count")
                as usize
        );
        assert_eq!(attached, 0);

        let count_matches = memory.query("Count the number of Rs in strawberry.", 3);
        assert_eq!(count_matches.len(), 1);
        assert!(!count_matches[0].vector_slice_available);
        assert!(count_matches[0].unicode_packet_id.is_some());
        assert_eq!(count_matches[0].route_preserved, None);

        let parallel_matches = memory.query(
            "If 1 towel takes 4 hours to dry and I dry 3 more towels with enough line space, how long will it take?",
            3,
        );
        assert_eq!(parallel_matches.len(), 1);
        assert!(!parallel_matches[0].vector_slice_available);
        assert!(parallel_matches[0].unicode_packet_id.is_some());
        assert_eq!(parallel_matches[0].route_preserved, None);
    }
}
