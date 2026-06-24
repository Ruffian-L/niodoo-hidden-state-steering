//! Lightweight direct Qdrant REST client for the [REQUEST: REMEMBER] vault tether.
//!
//! Uses a small synchronous HTTP client from the decode loop / end-of-turn.
//! Designed for the "quiet queue + next turn [INTERNAL MEMORY RECALL]" safer design.
//!
//! The 64D probe (target_z / last_probe_bucket_mean_64 from the exact step the model emitted REMEMBER)
//! is sent as the query vector. The collection is expected to have points with 64D vectors (named or main)
//! for the chat history / self-created memories. This aligns retrieval with the same geometry used for
//! correction packets and route memory.
//!
//! For the historical 770MB vault (prior Grok/Claude/Gemini chats), the ingestion scripts are updated
//! to also store a 64D projection of each chunk (using nomic Matryoshka truncate to 64 or simple
//! deterministic projection). This makes the tether actually useful instead of random.
//!
//! The runtime does the query (so it has the live probe) and the save (so niodoo can persist its own
//! creations/memories keyed by the probe at creation time). The driver (py read_aloud etc.) decides
//! the *presentation* timing ("at start of next turn" as visible side note) for safety / no mid-thought hijack.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Simple client for vault search + self-memory upsert on REMEMBER.
#[derive(Clone)]
pub struct VaultClient {
    base_url: String,
    collection: String,
    endpoint: Option<HttpEndpoint>,
    /// If true, we will try searching with a named vector "memory_64d" / "route_64d" first.
    /// Falls back to unnamed vector with the 64D probe (for collections set up that way).
    prefer_named_64d: bool,
}

#[derive(Clone)]
struct HttpEndpoint {
    host: String,
    port: u16,
    path_prefix: String,
}

struct HttpJsonResponse {
    status: u16,
    body: Vec<u8>,
}

impl VaultClient {
    pub fn new(base_url: &str, collection: &str) -> Self {
        let normalized_base_url = base_url.trim_end_matches('/').to_string();
        Self {
            endpoint: parse_http_endpoint(&normalized_base_url),
            base_url: normalized_base_url,
            collection: collection.to_string(),
            prefer_named_64d: true,
        }
    }

    /// Search the vault using the 64D probe (the target_z at the moment of [REQUEST: REMEMBER]).
    /// Returns up to `limit` text payloads (the "flashback" content).
    /// Quiet / best-effort: on any error returns empty vec (caller logs a soft note).
    pub fn search_by_probe_64(&self, probe: &[f32; 64], limit: usize) -> Vec<String> {
        if self.base_url.is_empty() || self.collection.is_empty() || self.endpoint.is_none() {
            return vec![];
        }

        // Try named 64D vector first (the clean alignment with packets / route geometry).
        if self.prefer_named_64d {
            if let Ok(hits) = self.try_search_named_64(probe, limit) {
                if !hits.is_empty() {
                    return hits;
                }
            }
        }

        // Fallback: unnamed vector (the 64D probe directly). Works if the collection
        // was created with size=64 vectors for memories.
        self.try_search_unnamed_64(probe, limit).unwrap_or_default()
    }

    fn try_search_named_64(&self, probe: &[f32; 64], limit: usize) -> Result<Vec<String>> {
        let path = format!("/collections/{}/points/search", self.collection);

        // Try common names used in the packet/route system.
        for name in ["memory_64d", "route_64d", "probe_64d", "target_z_64d"].iter() {
            let body = json!({
                "vector": { "name": name, "vector": probe.to_vec() },
                "limit": limit,
                "with_payload": true,
                "params": { "hnsw_ef": 128 }
            });

            let resp = self
                .request_json("POST", &path, &body)
                .with_context(|| format!("vault search POST to {}{}", self.base_url, path))?;

            if !http_status_success(resp.status) {
                continue;
            }

            let json: Value =
                serde_json::from_slice(&resp.body).with_context(|| "parse vault search json")?;
            if let Some(hits) = self.extract_text_hits(&json) {
                if !hits.is_empty() {
                    return Ok(hits);
                }
            }
        }
        Ok(vec![])
    }

    fn try_search_unnamed_64(&self, probe: &[f32; 64], limit: usize) -> Result<Vec<String>> {
        let path = format!("/collections/{}/points/search", self.collection);

        let body = json!({
            "vector": probe.to_vec(),
            "limit": limit,
            "with_payload": true,
            "params": { "hnsw_ef": 128 }
        });

        let resp = self.request_json("POST", &path, &body).with_context(|| {
            format!(
                "vault search POST (unnamed 64) to {}{}",
                self.base_url, path
            )
        })?;

        if !http_status_success(resp.status) {
            return Ok(vec![]);
        }

        let json: Value =
            serde_json::from_slice(&resp.body).with_context(|| "parse vault search json")?;
        Ok(self.extract_text_hits(&json).unwrap_or_default())
    }

    fn extract_text_hits(&self, json: &Value) -> Option<Vec<String>> {
        let result = json.get("result")?;
        let arr = result.as_array()?;
        let mut out = vec![];
        for hit in arr {
            if let Some(payload) = hit.get("payload") {
                if let Some(text) = payload.get("text").and_then(|v| v.as_str()) {
                    if !text.trim().is_empty() {
                        out.push(text.trim().to_string());
                    }
                } else if let Some(txt) = payload.get("content").and_then(|v| v.as_str()) {
                    if !txt.trim().is_empty() {
                        out.push(txt.trim().to_string());
                    }
                }
            }
        }
        Some(out)
    }

    /// Save a memory created by niodoo itself (the "read through his own creation and save memories" point).
    /// Uses the 64D probe at the time of the REMEMBER as the vector key.
    /// The text is the visible reasoning / payload around the tag (or the flashback + insight).
    /// This populates the vault with self-curated entries in the exact geometry space used for future probes.
    pub fn save_self_memory(&self, probe: &[f32; 64], text: &str) -> Result<()> {
        if self.base_url.is_empty()
            || self.collection.is_empty()
            || self.endpoint.is_none()
            || text.trim().is_empty()
        {
            return Ok(());
        }

        // Stable numeric id: Qdrant accepts unsigned integers or UUIDs as point ids.
        let point_id = {
            let mut h: u64 = 0xcbf29ce484222325;
            for &b in text.as_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };

        let vector_64 = probe.to_vec();

        // Deterministic 4096 from the same text (so self-memories have a full-size vector too
        // and the upsert succeeds against a collection that declares an unnamed 4096 default).
        // This is only for schema compat; retrieval for the tether uses the 64D named.
        let vector_4096: Vec<f32> = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            text.hash(&mut h);
            let seed = h.finish();
            (0..4096)
                .map(|i| {
                    let v = (seed
                        .wrapping_add((i as u64).wrapping_mul(6364136223846793005))
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        >> 33) as u32;
                    (v as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        };

        let payload = json!({
            "text": text,
            "content": text,
            "source": "niodoo_self_creation",
            "type": "remember_vault_entry",
            "created_at": chrono::Utc::now().to_rfc3339()
        });

        // Provide both the unnamed 4096 (for collection schema) and the named memory_64d (for the tether).
        // This matches how the ingest sets up points and ensures save_self_memory succeeds.
        let body = json!({
            "points": [{
                "id": point_id,
                "vector": {
                    "": vector_4096,
                    "memory_64d": vector_64
                },
                "payload": payload
            }]
        });

        let path = format!("/collections/{}/points?wait=true", self.collection);
        let resp = self
            .request_json("PUT", &path, &body)
            .with_context(|| format!("vault upsert PUT to {}{}", self.base_url, path))?;

        if !http_status_success(resp.status) {
            // Best effort; don't fail the main generation.
            eprintln!("[VAULT] save_self_memory non-success: {}", resp.status);
        }

        Ok(())
    }

    fn request_json(&self, method: &str, path: &str, body: &Value) -> Result<HttpJsonResponse> {
        let endpoint = self
            .endpoint
            .as_ref()
            .ok_or_else(|| anyhow!("vault base url must be http://host[:port][/prefix]"))?;
        endpoint.request_json(method, path, body)
    }
}

impl HttpEndpoint {
    fn request_json(&self, method: &str, path: &str, body: &Value) -> Result<HttpJsonResponse> {
        let payload = serde_json::to_vec(body).with_context(|| "serialize vault request json")?;
        let full_path = format!("{}{}", self.path_prefix, path);
        let host_header = if self.port == 80 {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        };
        let mut stream = TcpStream::connect((self.host.as_str(), self.port))
            .with_context(|| format!("connect to vault {}:{}", self.host, self.port))?;
        stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(10))).ok();

        write!(
            stream,
            "{method} {full_path} HTTP/1.1\r\nHost: {host_header}\r\nContent-Type: application/json\r\nAccept: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n",
            payload.len()
        )
        .with_context(|| "write vault request headers")?;
        stream
            .write_all(&payload)
            .with_context(|| "write vault request body")?;

        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .with_context(|| "read vault response")?;
        parse_http_response(&response)
    }
}

fn parse_http_endpoint(base_url: &str) -> Option<HttpEndpoint> {
    let rest = base_url
        .trim()
        .trim_end_matches('/')
        .strip_prefix("http://")?;
    let (host_port, path_prefix) = rest.split_once('/').unwrap_or((rest, ""));
    if host_port.is_empty() {
        return None;
    }

    let (host, port) = if let Some((host, port_text)) = host_port.rsplit_once(':') {
        let port = port_text.parse::<u16>().ok()?;
        (host.to_string(), port)
    } else {
        (host_port.to_string(), 80)
    };

    if host.is_empty() {
        return None;
    }

    let path_prefix = if path_prefix.is_empty() {
        String::new()
    } else {
        format!("/{}", path_prefix.trim_end_matches('/'))
    };

    Some(HttpEndpoint {
        host,
        port,
        path_prefix,
    })
}

fn parse_http_response(response: &[u8]) -> Result<HttpJsonResponse> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| anyhow!("vault response missing HTTP header terminator"))?;
    let header_bytes = &response[..header_end];
    let body_bytes = &response[header_end + 4..];
    let headers = std::str::from_utf8(header_bytes).with_context(|| "parse vault headers utf8")?;
    let status_line = headers
        .lines()
        .next()
        .ok_or_else(|| anyhow!("vault response missing status line"))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("vault response missing status code"))?
        .parse::<u16>()
        .with_context(|| format!("parse vault status line {status_line:?}"))?;

    let body = if headers.lines().any(header_is_chunked) {
        decode_chunked_body(body_bytes)?
    } else {
        body_bytes.to_vec()
    };

    Ok(HttpJsonResponse { status, body })
}

fn header_is_chunked(line: &str) -> bool {
    let Some((name, value)) = line.split_once(':') else {
        return false;
    };
    name.trim().eq_ignore_ascii_case("transfer-encoding")
        && value
            .split(',')
            .any(|part| part.trim().eq_ignore_ascii_case("chunked"))
}

fn decode_chunked_body(mut bytes: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = bytes
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| anyhow!("chunked vault response missing chunk size"))?;
        let size_line =
            std::str::from_utf8(&bytes[..line_end]).with_context(|| "parse chunk size utf8")?;
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .with_context(|| format!("parse chunk size {size_hex:?}"))?;
        bytes = &bytes[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if bytes.len() < size + 2 {
            bail!("chunked vault response ended before chunk body");
        }
        out.extend_from_slice(&bytes[..size]);
        bytes = &bytes[size + 2..];
    }
}

fn http_status_success(status: u16) -> bool {
    (200..300).contains(&status)
}
