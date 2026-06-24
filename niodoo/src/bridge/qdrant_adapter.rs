//! # Qdrant-Backed Memory Store (feature-gated `qdrant_backend`)
//!
//! Sidecar layer that mirrors the existing in-memory `CorrectionPacketStore`
//! and `MistakeReflexMemory` to a Qdrant collection. Solves the launch-only
//! load constraint flagged by CLAIMS §10eb step 5: with this layer enabled,
//! new corrections become live within one decode turn.
//!
//! ## Architecture
//!
//! - **Dual-write at every persistence site.** The existing `save()` /
//!   `write_to_jsonl()` calls stay untouched. After each one, the runtime
//!   calls into this adapter to upsert the same record to Qdrant. JSONL
//!   remains the durable audit trail; Qdrant becomes the queryable live
//!   index.
//! - **Per-turn refresh.** At the top of each decode turn (when
//!   `turn_id % query_refresh_every_turns == 0`), the runtime calls
//!   `refresh_correction_packets_into` / `refresh_mistake_reflex_into` to
//!   REPLACE the in-memory store contents with the current Qdrant scroll.
//!   The per-token `forward_with_decay` path at `principia.rs:3213` and the
//!   per-turn `query` path at `simulation.rs:1675` keep operating on the
//!   in-memory store — they don't know Qdrant exists.
//! - **Graceful degrade.** Every Qdrant call is wrapped in `try_*` — on
//!   error, the runtime logs a warning and continues. JSONL is the source
//!   of truth when Qdrant is unreachable.
//!
//! ## Invariants preserved
//!
//! - Zero changes to `decide_packet_authority` gate logic.
//! - Zero changes to `CorrectionPacket::forward_with_pull` math.
//! - Zero changes to `MistakeReflexMemory::query` scoring.
//! - Default build (without `qdrant_backend`) is byte-identical to today.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use qdrant_client::qdrant::{
    point_id::PointIdOptions, vectors_config::Config as VectorsConfigEnum, with_payload_selector,
    CreateCollection, Distance, NamedVectors, PointId, PointStruct, ScrollPoints, UpsertPoints,
    Value, Vector, VectorParams, VectorParamsMap, Vectors, VectorsConfig, WithPayloadSelector,
};
use qdrant_client::{Payload, Qdrant};
use uuid::Uuid;

use crate::bridge::correction_packets::{CorrectionPacket, CorrectionPacketJson};
use crate::bridge::CorrectionPacketStore;
use crate::runtime::mistake_reflex::{MistakeReflexEvent, MistakeReflexMemory};

/// Deterministic uuid-v5 namespace for niodoo Qdrant point IDs. Hashing a
/// packet_id or event_id under this namespace gives the same UUID across
/// processes, so upserts stay idempotent.
const NIODOO_QDRANT_UUID_NS: Uuid = Uuid::from_bytes([
    0x6e, 0x69, 0x6f, 0x64, 0x6f, 0x6f, 0x5f, 0x71, 0x64, 0x72, 0x61, 0x6e, 0x74, 0x5f, 0x76, 0x35,
]);

/// Configuration carried from CLI flags into the adapter.
#[derive(Clone, Debug)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub correction_packets_collection: String,
    pub mistake_reflex_collection: String,
    pub query_refresh_every_turns: usize,
    pub per_turn_cache_size: usize,
}

impl QdrantConfig {
    /// Reasonable default. The url + collection names mirror the CLI
    /// defaults in `cli.rs`.
    pub fn default_local() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            api_key: None,
            correction_packets_collection: "niodoo_correction_packets".to_string(),
            mistake_reflex_collection: "niodoo_mistake_reflex_events".to_string(),
            query_refresh_every_turns: 1,
            per_turn_cache_size: 256,
        }
    }
}

/// Shared client handle. Cloning is cheap (Arc).
#[derive(Clone)]
pub struct QdrantBackend {
    inner: Arc<QdrantBackendInner>,
}

struct QdrantBackendInner {
    client: Qdrant,
    cfg: QdrantConfig,
    runtime: tokio::runtime::Handle,
}

impl QdrantBackend {
    /// Build a backend handle. Requires being called from within a tokio
    /// runtime (which `run_simulation` already is, since it's `async`).
    /// Picks up the ambient tokio handle via `tokio::runtime::Handle::current`.
    pub fn try_new(cfg: QdrantConfig) -> Result<Self> {
        let mut builder = Qdrant::from_url(&cfg.url);
        if let Some(ref key) = cfg.api_key {
            if !key.is_empty() {
                builder = builder.api_key(key.clone());
            }
        }
        let client = builder
            .build()
            .with_context(|| format!("failed to build Qdrant client for {}", cfg.url))?;
        let runtime = tokio::runtime::Handle::try_current()
            .context("QdrantBackend::try_new must be called from within a tokio runtime")?;
        Ok(Self {
            inner: Arc::new(QdrantBackendInner {
                client,
                cfg,
                runtime,
            }),
        })
    }

    pub fn cfg(&self) -> &QdrantConfig {
        &self.inner.cfg
    }

    /// Ensure both collections exist with the expected schema. Idempotent —
    /// no-op if collections already exist with the right shape.
    pub fn ensure_collections(&self) -> Result<()> {
        self.ensure_correction_packets_collection()?;
        self.ensure_mistake_reflex_collection()?;
        Ok(())
    }

    fn ensure_correction_packets_collection(&self) -> Result<()> {
        let name = self.inner.cfg.correction_packets_collection.clone();
        let exists = self
            .block_on(async { self.inner.client.collection_exists(name.clone()).await })?;
        if exists {
            return Ok(());
        }
        // Named vectors: target_z_64d (size 64) + route_64d (size 64).
        // Both COSINE. Optional/sparse-ness is handled by upserting empty
        // vectors when a packet has no route_64d.
        let mut named_vectors: HashMap<String, VectorParams> = HashMap::new();
        named_vectors.insert(
            "target_z_64d".to_string(),
            VectorParams {
                size: 64,
                distance: Distance::Cosine as i32,
                ..Default::default()
            },
        );
        named_vectors.insert(
            "route_64d".to_string(),
            VectorParams {
                size: 64,
                distance: Distance::Cosine as i32,
                ..Default::default()
            },
        );
        let req = CreateCollection {
            collection_name: name,
            vectors_config: Some(VectorsConfig {
                config: Some(VectorsConfigEnum::ParamsMap(VectorParamsMap {
                    map: named_vectors,
                })),
            }),
            ..Default::default()
        };
        self.block_on(async { self.inner.client.create_collection(req).await })?;
        Ok(())
    }

    fn ensure_mistake_reflex_collection(&self) -> Result<()> {
        let name = self.inner.cfg.mistake_reflex_collection.clone();
        let exists = self
            .block_on(async { self.inner.client.collection_exists(name.clone()).await })?;
        if exists {
            return Ok(());
        }
        // Single named vector "route_64d" (size 64). Events without a route
        // probe get an all-zero vector; payload filtering still works.
        let mut named_vectors: HashMap<String, VectorParams> = HashMap::new();
        named_vectors.insert(
            "route_64d".to_string(),
            VectorParams {
                size: 64,
                distance: Distance::Cosine as i32,
                ..Default::default()
            },
        );
        let req = CreateCollection {
            collection_name: name,
            vectors_config: Some(VectorsConfig {
                config: Some(VectorsConfigEnum::ParamsMap(VectorParamsMap {
                    map: named_vectors,
                })),
            }),
            ..Default::default()
        };
        self.block_on(async { self.inner.client.create_collection(req).await })?;
        Ok(())
    }

    // ── correction_packets ──────────────────────────────────────────────

    /// Bulk-upsert every packet in the given store into Qdrant. Used at
    /// startup to seed from a freshly loaded JSONL.
    pub fn try_seed_correction_packets(&self, store: &CorrectionPacketStore) -> Result<usize> {
        let points: Vec<PointStruct> = store
            .iter_all_packets()
            .map(point_from_correction_packet)
            .collect();
        if points.is_empty() {
            return Ok(0);
        }
        let count = points.len();
        let req = UpsertPoints {
            collection_name: self.inner.cfg.correction_packets_collection.clone(),
            wait: Some(true),
            points,
            ..Default::default()
        };
        self.block_on(async { self.inner.client.upsert_points(req).await })?;
        Ok(count)
    }

    /// Upsert one packet. Used by mutating sites (correction mints, LOCK
    /// invalidation flips) so the next turn's refresh sees the change.
    pub fn try_upsert_correction_packet(&self, packet: &CorrectionPacket) -> Result<()> {
        let point = point_from_correction_packet(packet);
        let req = UpsertPoints {
            collection_name: self.inner.cfg.correction_packets_collection.clone(),
            wait: Some(false),
            points: vec![point],
            ..Default::default()
        };
        self.block_on(async { self.inner.client.upsert_points(req).await })?;
        Ok(())
    }

    /// Scroll every point in the correction-packets collection and replace
    /// the contents of the in-memory store with the result. Called at
    /// turn boundaries (per `query_refresh_every_turns`).
    pub fn refresh_correction_packets_into(
        &self,
        store: &mut CorrectionPacketStore,
    ) -> Result<usize> {
        let mut fresh = CorrectionPacketStore::new();
        let mut next_offset: Option<PointId> = None;
        let limit_per_page: u32 = 256;
        loop {
            let req = ScrollPoints {
                collection_name: self.inner.cfg.correction_packets_collection.clone(),
                limit: Some(limit_per_page),
                with_payload: Some(WithPayloadSelector {
                    selector_options: Some(with_payload_selector::SelectorOptions::Enable(true)),
                }),
                with_vectors: Some(true.into()),
                offset: next_offset.clone(),
                ..Default::default()
            };
            let resp = self.block_on(async { self.inner.client.scroll(req).await })?;
            for point in &resp.result {
                if let Some(packet) = correction_packet_from_retrieved(point) {
                    fresh.insert(packet);
                }
            }
            next_offset = resp.next_page_offset;
            if next_offset.is_none() || resp.result.is_empty() {
                break;
            }
        }
        let count = fresh.total();
        *store = fresh;
        Ok(count)
    }

    // ── mistake_reflex ──────────────────────────────────────────────────

    /// Bulk-upsert every event in the given memory into Qdrant. Used at
    /// startup to seed from a freshly loaded JSONL.
    pub fn try_seed_mistake_reflex(&self, memory: &MistakeReflexMemory) -> Result<usize> {
        let points: Vec<PointStruct> = memory
            .events()
            .iter()
            .map(point_from_mistake_reflex_event)
            .collect();
        if points.is_empty() {
            return Ok(0);
        }
        let count = points.len();
        let req = UpsertPoints {
            collection_name: self.inner.cfg.mistake_reflex_collection.clone(),
            wait: Some(true),
            points,
            ..Default::default()
        };
        self.block_on(async { self.inner.client.upsert_points(req).await })?;
        Ok(count)
    }

    /// Upsert one mistake-reflex event. Used by capture sites so the
    /// next turn's refresh sees the new event.
    pub fn try_upsert_mistake_reflex_event(&self, event: &MistakeReflexEvent) -> Result<()> {
        let point = point_from_mistake_reflex_event(event);
        let req = UpsertPoints {
            collection_name: self.inner.cfg.mistake_reflex_collection.clone(),
            wait: Some(false),
            points: vec![point],
            ..Default::default()
        };
        self.block_on(async { self.inner.client.upsert_points(req).await })?;
        Ok(())
    }

    /// Scroll every point in the mistake-reflex collection and replace the
    /// in-memory events with the result. Per-turn refresh.
    pub fn refresh_mistake_reflex_into(&self, memory: &mut MistakeReflexMemory) -> Result<usize> {
        let mut fresh: Vec<MistakeReflexEvent> = Vec::new();
        let mut next_offset: Option<PointId> = None;
        let limit_per_page: u32 = 256;
        loop {
            let req = ScrollPoints {
                collection_name: self.inner.cfg.mistake_reflex_collection.clone(),
                limit: Some(limit_per_page),
                with_payload: Some(WithPayloadSelector {
                    selector_options: Some(with_payload_selector::SelectorOptions::Enable(true)),
                }),
                with_vectors: Some(true.into()),
                offset: next_offset.clone(),
                ..Default::default()
            };
            let resp = self.block_on(async { self.inner.client.scroll(req).await })?;
            for point in &resp.result {
                if let Some(event) = mistake_reflex_event_from_retrieved(point) {
                    fresh.push(event);
                }
            }
            next_offset = resp.next_page_offset;
            if next_offset.is_none() || resp.result.is_empty() {
                break;
            }
        }
        let count = fresh.len();
        memory.replace_events(fresh);
        Ok(count)
    }

    // ── internal helpers ────────────────────────────────────────────────

    /// Block on a future using the captured tokio handle. We do this
    /// because the adapter's public API is sync (mirroring the existing
    /// `MistakeReflexMemory::save`/`load` shape), but the qdrant-client is
    /// async. Calling from inside the simulation loop is fine — the loop
    /// is already inside the tokio runtime.
    fn block_on<F, T>(&self, fut: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        // We're already on the runtime — use Handle::block_on. Allows the
        // current task to yield while the future runs.
        let handle = self.inner.runtime.clone();
        tokio::task::block_in_place(move || handle.block_on(fut))
    }
}

// ── conversion helpers ──────────────────────────────────────────────────

fn point_id_for(slug: &str) -> PointId {
    let uuid = Uuid::new_v5(&NIODOO_QDRANT_UUID_NS, slug.as_bytes());
    PointId {
        point_id_options: Some(PointIdOptions::Uuid(uuid.to_string())),
    }
}

fn point_from_correction_packet(packet: &CorrectionPacket) -> PointStruct {
    let snapshot = packet.to_json_snapshot();
    let payload_value: serde_json::Value =
        serde_json::to_value(&snapshot).unwrap_or(serde_json::Value::Null);
    let payload: Payload = match payload_value {
        serde_json::Value::Object(map) => {
            let entries: HashMap<String, Value> =
                map.into_iter().map(|(k, v)| (k, json_to_value(v))).collect();
            entries.into()
        }
        _ => Payload::new(),
    };

    let target_vec: Vec<f32> = packet.target_z_64d.to_vec();
    let route_vec: Vec<f32> = vec![0.0; 64]; // we don't carry a separate route probe per packet
                                              // in CorrectionPacket today; reserve the named vector for future use.

    let mut named: HashMap<String, Vector> = HashMap::new();
    named.insert("target_z_64d".to_string(), target_vec.into());
    named.insert("route_64d".to_string(), route_vec.into());

    PointStruct {
        id: Some(point_id_for(&packet.packet_id)),
        vectors: Some(Vectors {
            vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vectors(
                NamedVectors { vectors: named },
            )),
        }),
        payload: payload.into(),
    }
}

fn correction_packet_from_retrieved(
    point: &qdrant_client::qdrant::RetrievedPoint,
) -> Option<CorrectionPacket> {
    let payload_map = &point.payload;
    let serde_map: serde_json::Map<String, serde_json::Value> = payload_map
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();
    let snapshot: CorrectionPacketJson =
        serde_json::from_value(serde_json::Value::Object(serde_map)).ok()?;
    CorrectionPacket::from_json_snapshot(snapshot).ok()
}

fn point_from_mistake_reflex_event(event: &MistakeReflexEvent) -> PointStruct {
    let payload_value: serde_json::Value =
        serde_json::to_value(event).unwrap_or(serde_json::Value::Null);
    let payload: Payload = match payload_value {
        serde_json::Value::Object(map) => {
            let entries: HashMap<String, Value> =
                map.into_iter().map(|(k, v)| (k, json_to_value(v))).collect();
            entries.into()
        }
        _ => Payload::new(),
    };

    // route_64d may be missing or wrong-sized. Normalize to 64.
    let route_vec: Vec<f32> = match &event.route_64d {
        Some(v) if v.len() == 64 => v.clone(),
        _ => vec![0.0; 64],
    };

    let mut named: HashMap<String, Vector> = HashMap::new();
    named.insert("route_64d".to_string(), route_vec.into());

    PointStruct {
        id: Some(point_id_for(&event.id)),
        vectors: Some(Vectors {
            vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vectors(
                NamedVectors { vectors: named },
            )),
        }),
        payload: payload.into(),
    }
}

fn mistake_reflex_event_from_retrieved(
    point: &qdrant_client::qdrant::RetrievedPoint,
) -> Option<MistakeReflexEvent> {
    let payload_map = &point.payload;
    let serde_map: serde_json::Map<String, serde_json::Value> = payload_map
        .iter()
        .map(|(k, v)| (k.clone(), value_to_json(v)))
        .collect();
    serde_json::from_value(serde_json::Value::Object(serde_map)).ok()
}

// ── serde_json::Value ↔ qdrant Value bridge ────────────────────────────

fn json_to_value(v: serde_json::Value) -> Value {
    use qdrant_client::qdrant::value::Kind;
    match v {
        serde_json::Value::Null => Value {
            kind: Some(Kind::NullValue(0)),
        },
        serde_json::Value::Bool(b) => Value {
            kind: Some(Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value {
                    kind: Some(Kind::IntegerValue(i)),
                }
            } else if let Some(f) = n.as_f64() {
                Value {
                    kind: Some(Kind::DoubleValue(f)),
                }
            } else {
                Value {
                    kind: Some(Kind::StringValue(n.to_string())),
                }
            }
        }
        serde_json::Value::String(s) => Value {
            kind: Some(Kind::StringValue(s)),
        },
        serde_json::Value::Array(arr) => {
            let values: Vec<Value> = arr.into_iter().map(json_to_value).collect();
            Value {
                kind: Some(Kind::ListValue(qdrant_client::qdrant::ListValue { values })),
            }
        }
        serde_json::Value::Object(map) => {
            let fields: HashMap<String, Value> =
                map.into_iter().map(|(k, v)| (k, json_to_value(v))).collect();
            Value {
                kind: Some(Kind::StructValue(qdrant_client::qdrant::Struct {
                    fields,
                })),
            }
        }
    }
}

fn value_to_json(v: &Value) -> serde_json::Value {
    use qdrant_client::qdrant::value::Kind;
    match &v.kind {
        Some(Kind::NullValue(_)) | None => serde_json::Value::Null,
        Some(Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(Kind::IntegerValue(i)) => serde_json::Value::Number((*i).into()),
        Some(Kind::DoubleValue(f)) => serde_json::Number::from_f64(*f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(value_to_json).collect())
        }
        Some(Kind::StructValue(strct)) => {
            let map: serde_json::Map<String, serde_json::Value> = strct
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_point_id_for_same_slug() {
        let a = point_id_for("packet_abc");
        let b = point_id_for("packet_abc");
        assert_eq!(a, b);
    }

    #[test]
    fn different_slug_different_id() {
        let a = point_id_for("packet_abc");
        let b = point_id_for("packet_xyz");
        assert_ne!(a, b);
    }

    #[test]
    fn json_value_roundtrip_string() {
        let input = serde_json::Value::String("hello".to_string());
        let v = json_to_value(input.clone());
        let out = value_to_json(&v);
        assert_eq!(out, input);
    }

    #[test]
    fn json_value_roundtrip_object_with_mixed_types() {
        let input = serde_json::json!({
            "name": "packet_42",
            "vq_code": 17,
            "pull": 0.25,
            "active": true,
            "tags": ["a", "b"],
            "absent": serde_json::Value::Null,
        });
        let v = json_to_value(input.clone());
        let out = value_to_json(&v);
        assert_eq!(out, input);
    }
}
