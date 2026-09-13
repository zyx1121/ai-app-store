//! The GGUF catalogue: search on Hugging Face, fit against the device, download.
//!
//! Only what llama.cpp runs is listed, decision D7: root level single file GGUF
//! plus the `mmproj` sibling a vision model needs. Split shards and files in
//! subfolders are skipped because the platform never assembles them.

use std::io;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::apps::ModelKind;
use crate::error::{Error, Result};
use crate::paths;

/// A Model is identified by a Hugging Face repo plus a quant name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub repo: String,
    pub quant: String,
}

impl ModelRef {
    pub fn new(repo: impl Into<String>, quant: impl Into<String>) -> Self {
        Self {
            repo: repo.into(),
            quant: quant.into(),
        }
    }

    /// Filesystem safe identifier, also used as the instance registry key.
    pub fn slug(&self) -> String {
        format!("{}-{}", self.repo.replace('/', "_"), self.quant)
    }
}

/// One downloadable file in a repo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelFile {
    pub filename: String,
    /// Quant name parsed from the filename, unsloth `UD-` prefix kept.
    pub quant: String,
    pub size_bytes: u64,
    /// The mmproj file that belongs with this quant, for vision models.
    pub mmproj: Option<String>,
}

/// One search hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelSummary {
    pub repo: String,
    pub downloads: u64,
    pub likes: u64,
    /// Quant names found in the repo, parsed by the same rules as [`files`].
    ///
    /// Search asks for `full=false`, so this is only filled when Hugging Face
    /// carries the file list on the hit. [`files`] is the authority.
    pub quants: Vec<String>,
    /// The HF pipeline tag, `text-generation` or `image-text-to-text` here.
    pub pipeline_tag: Option<String>,
    /// True when the repo needs an accepted licence before it serves files.
    pub gated: bool,
}

/// A page of search results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPage {
    pub items: Vec<ModelSummary>,
    /// Opaque cursor for the next page, `None` at the end.
    pub next_cursor: Option<String>,
}

/// How a model sits against the device memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Fit {
    /// Weights plus KV cache fit inside the budget.
    Ready,
    /// Weights fit, the headroom for KV cache is thin.
    Maybe,
    /// Weights alone do not fit.
    Incompatible,
}

/// Progress of one download, reported to the caller as bytes land.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

/// Callback shape accepted by [`download`].
pub type ProgressFn<'a> = &'a (dyn Fn(Progress) + Send + Sync);

/// One downloaded model as it sits on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelRecord {
    repo: String,
    quant: String,
    filename: String,
    mmproj: Option<String>,
}

/// The weights cost their file size plus a tenth, for the runtime around them.
const WEIGHT_NUMERATOR: u128 = 11;
const WEIGHT_DENOMINATOR: u128 = 10;
/// Bytes one byte of KV cache is stored in: f16 keys and f16 values.
const KV_BYTES_PER_ELEMENT: u128 = 2;
/// Keys and values, the two halves of the cache.
const KV_TENSORS: u128 = 2;
/// Rough size of a billion parameters as GGUF, for the estimate that has no
/// header to read: the quants this platform serves land near 4.8 bits a weight.
const BYTES_PER_BILLION_PARAMS: u128 = 600 * 1024 * 1024;
/// The fallback rate, PLAN.md section 5: 18 MiB of KV cache per thousand tokens
/// per billion parameters.
///
/// Measured against the model this platform is built around rather than
/// guessed. Qwen3-8B keeps `2 x 36 layers x 8 kv heads x 128 head_dim x 2 bytes`
/// = 147,456 bytes a token, which is 140.6 MiB per thousand tokens, 17.6 MiB
/// once divided by its 8 billion parameters. The rate this file shipped with was
/// 0.5 MiB, 35 times under the truth, so a model was badged Ready at a context
/// it could never be given.
const FALLBACK_KV_BYTES_PER_1K_PER_1B: u128 = 18 * 1024 * 1024;

/// Context sizes and slot counts the planner will hand out, most generous first.
///
/// The planner walks this until the estimate fits, so the ladder is the whole
/// policy: a roomy device gets the top rung and a tight one is stepped down
/// rather than being told the model does not fit at all.
const PLAN_LADDER: &[(u32, u32)] = &[
    (32768, 4),
    (32768, 2),
    (16384, 2),
    (16384, 1),
    (8192, 1),
    (4096, 1),
    (2048, 1),
];

/// Share of the budget an instance is planned to stay inside, PLAN.md section 5.
const CEILING_NUMERATOR: u64 = 9;
const CEILING_DENOMINATOR: u64 = 10;

/// The rung a model has to reach before the store calls it Ready.
///
/// Below this the planner is still willing to start the model, but a 2048 token
/// context is a demo and not a usable app, so the badge says Maybe instead.
const FIT_MIN_CTX: u32 = 4096;
/// One slot, the floor of [`PLAN_LADDER`].
const FIT_MIN_PARALLEL: u32 = 1;

/// Base of the Hugging Face API.
const HF_API: &str = "https://huggingface.co/api";
/// Base of the Hugging Face file endpoint.
const HF_HOST: &str = "https://huggingface.co";
/// Hits per search page.
const SEARCH_LIMIT: u32 = 30;
/// Floor between two [`Progress`] callbacks.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(200);
/// Sidecar naming one downloaded model, so [`installed`] does not parse slugs.
const RECORD_FILE: &str = "model.json";
/// Token for gated repos, read from the environment when present.
const TOKEN_ENV: &str = "AIAS_HF_TOKEN";
/// Hops a download follows before it gives up.
const MAX_REDIRECTS: u32 = 5;

/// Search Hugging Face for GGUF repos this runtime can serve.
///
/// Unfiltered by pipeline tag; [`search_kind`] narrows to what a manifest can
/// declare.
pub async fn search(query: &str, cursor: Option<&str>) -> Result<SearchPage> {
    search_kind(query, None, cursor).await
}

/// Search Hugging Face for GGUF repos, optionally narrowed to one kind.
///
/// `kind` maps onto the HF `pipeline_tag` filter: [`ModelKind::Llm`] is
/// `text-generation`, [`ModelKind::Vlm`] is `image-text-to-text`. Those two tags
/// are the only ones llama.cpp serves, so a caller with no preference still
/// only sees GGUF repos.
pub async fn search_kind(
    query: &str,
    kind: Option<ModelKind>,
    cursor: Option<&str>,
) -> Result<SearchPage> {
    let limit = SEARCH_LIMIT.to_string();
    let mut params: Vec<(&str, &str)> = vec![
        ("filter", "gguf"),
        ("search", query),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", &limit),
        ("full", "false"),
    ];
    if let Some(kind) = kind {
        params.push(("pipeline_tag", pipeline_tag(kind)));
    }
    if let Some(cursor) = cursor {
        params.push(("cursor", cursor));
    }

    let response = http()
        .get(format!("{HF_API}/models"))
        .query(&params)
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(net)?;
    let response = check_status(response, "hugging face search").await?;

    let next_cursor = response
        .headers()
        .get(reqwest::header::LINK)
        .and_then(|value| value.to_str().ok())
        .and_then(next_cursor_from_link);

    let raw: Vec<RawModel> = response.json().await.map_err(net)?;
    Ok(SearchPage {
        items: raw.into_iter().map(ModelSummary::from).collect(),
        next_cursor,
    })
}

/// List the GGUF files of one repo, with their mmproj siblings.
///
/// Rules, all inherited from the previous PoC:
///
/// 1. Only root level files ending in `.gguf` count; a subfolder is skipped.
/// 2. A split shard, `-00001-of-00003.gguf`, is skipped.
/// 3. `mmproj*.gguf` is not a quant. The best one is attached to every quant.
pub async fn files(repo: &str) -> Result<Vec<ModelFile>> {
    let response = http()
        .get(format!("{HF_API}/models/{repo}"))
        .query(&[("blobs", "true")])
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(net)?;
    let response = check_status(response, repo).await?;
    let raw: RawRepo = response.json().await.map_err(net)?;
    Ok(parse_files(&raw))
}

/// Download the weights, and the mmproj file when the model is a vision model.
///
/// Streams into `<data_dir>/models/<slug>/`, resuming a previous `.part` when
/// one is there, and writes a `<filename>.sha256` sidecar. Hugging Face publishes
/// the SHA-256 of an LFS object in `x-linked-etag`; when it does, the bytes are
/// checked against it and a mismatch fails the download. Returns the path of the
/// main GGUF.
pub async fn download(
    model: &ModelRef,
    data_dir: &Path,
    progress: ProgressFn<'_>,
) -> Result<PathBuf> {
    let entries = files(&model.repo).await?;
    let entry = entries
        .iter()
        .find(|file| file.quant.eq_ignore_ascii_case(&model.quant))
        .ok_or_else(|| {
            Error::NotFound(format!("quant `{}` in repo `{}`", model.quant, model.repo))
        })?;

    let dir = model_dir(model, data_dir);
    tokio::fs::create_dir_all(&dir).await?;

    let path = download_file(&model.repo, &entry.filename, &dir, progress).await?;
    if let Some(mmproj) = &entry.mmproj {
        download_file(&model.repo, mmproj, &dir, progress).await?;
    }

    let record = ModelRecord {
        repo: model.repo.clone(),
        quant: entry.quant.clone(),
        filename: entry.filename.clone(),
        mmproj: entry.mmproj.clone(),
    };
    tokio::fs::write(dir.join(RECORD_FILE), serde_json::to_vec_pretty(&record)?).await?;

    Ok(path)
}

/// Every model already on disk: what it is, where its weights are, how big.
pub fn installed(data_dir: &Path) -> Vec<(ModelRef, PathBuf, u64)> {
    let root = paths::models_dir(data_dir);
    let Ok(entries) = std::fs::read_dir(&root) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path();
        if !dir.is_dir() {
            continue;
        }
        let Ok(bytes) = std::fs::read(dir.join(RECORD_FILE)) else {
            continue;
        };
        let Ok(record) = serde_json::from_slice::<ModelRecord>(&bytes) else {
            continue;
        };
        // The sidecar is written by this platform, but it sits in a directory
        // the user can edit, so its name is checked like any other.
        if !is_safe_filename(&record.filename) {
            continue;
        }
        let path = dir.join(&record.filename);
        let Ok(meta) = std::fs::metadata(&path) else {
            continue;
        };
        found.push((ModelRef::new(record.repo, record.quant), path, meta.len()));
    }
    found.sort_by(|left, right| left.0.cmp(&right.0));
    found
}

/// The mmproj downloaded next to a model, when it is a vision model.
///
/// `llama-server` needs it as `--mmproj`, and the Models page shows it.
pub fn installed_mmproj(model: &ModelRef, data_dir: &Path) -> Option<PathBuf> {
    let dir = model_dir(model, data_dir);
    let bytes = std::fs::read(dir.join(RECORD_FILE)).ok()?;
    let record: ModelRecord = serde_json::from_slice(&bytes).ok()?;
    let mmproj = record.mmproj?;
    if !is_safe_filename(&mmproj) {
        return None;
    }
    let path = dir.join(mmproj);
    path.is_file().then_some(path)
}

/// Delete one downloaded model, weights, mmproj and sidecars together.
pub fn remove(model: &ModelRef, data_dir: &Path) -> Result<()> {
    let dir = model_dir(model, data_dir);
    if !dir.is_dir() {
        return Err(Error::NotFound(format!(
            "model `{}` is not downloaded",
            model.slug()
        )));
    }
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

/// Where a downloaded model lives.
pub fn model_dir(model: &ModelRef, data_dir: &Path) -> PathBuf {
    paths::models_dir(data_dir).join(model.slug())
}

/// What one token of KV cache costs, read out of the GGUF header.
///
/// Grouped query attention is why this cannot be guessed from the file size:
/// two models of the same weight can differ by an order of magnitude in cache
/// per token, which is the whole reason the context has to be planned against
/// real numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KvLayout {
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

/// Memory one instance needs, weights and KV cache together, in MB.
///
/// The weights cost their file size plus a tenth. The cache costs
/// `ctx x n_parallel x 2 x n_layers x n_kv_heads x head_dim x 2 bytes`: every
/// slot holds its own context, and keys and values are both stored in f16.
///
/// Without a [`KvLayout`] the cache falls back to
/// [`FALLBACK_KV_BYTES_PER_1K_PER_1B`] per thousand tokens per billion
/// parameters, with the parameter count inferred from the file size. That is the pre download case, where there is no header to read;
/// it is a coarse number and [`kv_layout`] is used wherever the file is
/// actually on disk.
pub fn estimate_mb(size_bytes: u64, ctx: u32, n_parallel: u32, layout: Option<KvLayout>) -> u64 {
    let weights = u128::from(size_bytes) * WEIGHT_NUMERATOR / WEIGHT_DENOMINATOR;
    let tokens = u128::from(ctx) * u128::from(n_parallel.max(1));

    let cache = match layout {
        Some(layout) => {
            tokens
                * KV_TENSORS
                * u128::from(layout.n_layers)
                * u128::from(layout.n_kv_heads)
                * u128::from(layout.head_dim)
                * KV_BYTES_PER_ELEMENT
        }
        None => {
            tokens * u128::from(size_bytes) * FALLBACK_KV_BYTES_PER_1K_PER_1B
                / (1000 * BYTES_PER_BILLION_PARAMS)
        }
    };

    u64::try_from((weights + cache).div_ceil(1024 * 1024)).unwrap_or(u64::MAX)
}

/// The most generous context and slot count that fits 90% of `budget_mb`.
///
/// The planner used to read the context off a ladder of leftover megabytes and
/// never asked what that context would cost, so a model that fitted on its own
/// was started with a cache several times its own size. Now every rung is
/// priced with [`estimate_mb`] and the first one that fits wins. When nothing
/// fits, the floor is returned and the caller refuses the instance with a
/// number to show.
pub fn plan_context(size_bytes: u64, budget_mb: u64, layout: Option<KvLayout>) -> (u32, u32) {
    let ceiling = ceiling_mb(budget_mb);
    for (ctx, n_parallel) in PLAN_LADDER {
        if estimate_mb(size_bytes, *ctx, *n_parallel, layout) <= ceiling {
            return (*ctx, *n_parallel);
        }
    }
    *PLAN_LADDER.last().expect("the ladder has a floor")
}

/// 90% of a budget, the ceiling every planning decision uses.
pub fn ceiling_mb(budget_mb: u64) -> u64 {
    budget_mb * CEILING_NUMERATOR / CEILING_DENOMINATOR
}

/// Badge a model against a memory budget, both in the units of PLAN.md section 5.
///
/// `budget_mb` is [`crate::hardware::DeviceProfile::effective_memory_mb`]. The
/// badge asks the same question admission will, through the same planner
/// [`crate::instances::params_for`] uses: [`plan_context`] steps the context and
/// the slot count down until the whole estimate fits, and the badge reads the
/// rung it stopped on.
///
/// `Ready` means that rung is still a usable one, at least
/// [`FIT_MIN_CTX`] tokens over [`FIT_MIN_PARALLEL`] slot, inside the 90%
/// ceiling. `Maybe` means it fits at all, which is the floor rung inside the
/// whole budget. Nothing is downloaded yet, so there is no header to read and
/// the coarse cache estimate is used.
pub fn fit(size_bytes: u64, budget_mb: u64) -> Fit {
    let (ctx, n_parallel) = plan_context(size_bytes, budget_mb, None);
    let estimate = estimate_mb(size_bytes, ctx, n_parallel, None);
    if ctx >= FIT_MIN_CTX && n_parallel >= FIT_MIN_PARALLEL && estimate <= ceiling_mb(budget_mb) {
        Fit::Ready
    } else if estimate <= budget_mb {
        Fit::Maybe
    } else {
        Fit::Incompatible
    }
}

/// The KV cache layout of a GGUF file, or `None` when it cannot be read.
///
/// GGUF starts with a typed key value block, and llama.cpp puts the numbers
/// that decide the cache in it: block count, attention head counts and the
/// embedding length. Anything unexpected gives `None` and the caller falls back
/// to the coarse estimate, because a wrong layout is worse than no layout.
pub fn kv_layout(path: &Path) -> Option<KvLayout> {
    // Buffered because walking past the tokenizer vocabulary is a few hundred
    // thousand very small reads.
    let mut reader = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let metadata = read_gguf_metadata(&mut reader)?;

    let architecture = match metadata.get("general.architecture") {
        Some(GgufValue::Text(name)) => name.clone(),
        _ => return None,
    };
    let number = |key: &str| match metadata.get(&format!("{architecture}.{key}")) {
        Some(GgufValue::Number(value)) => u32::try_from(*value).ok(),
        // A per layer head count is published as an array; the largest is the
        // one the cache has to be sized for.
        Some(GgufValue::Numbers(values)) => values
            .iter()
            .copied()
            .max()
            .and_then(|value| u32::try_from(value).ok()),
        _ => None,
    };

    let n_layers = number("block_count")?;
    let n_heads = number("attention.head_count")?;
    let n_kv_heads = number("attention.head_count_kv").unwrap_or(n_heads);
    // `key_length` is authoritative when it is published; otherwise the heads
    // divide the embedding evenly, which is how llama.cpp derives it too.
    let head_dim = match number("attention.key_length") {
        Some(length) => length,
        None => number("embedding_length")?.checked_div(n_heads)?,
    };

    if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 {
        return None;
    }
    Some(KvLayout {
        n_layers,
        n_kv_heads,
        head_dim,
    })
}

/// The only GGUF metadata shapes this platform has a use for.
enum GgufValue {
    Number(u64),
    Numbers(Vec<u64>),
    Text(String),
    Ignored,
}

/// GGUF magic, little endian `"GGUF"`.
const GGUF_MAGIC: [u8; 4] = *b"GGUF";
/// Guard against a corrupt count turning into a huge allocation.
const GGUF_MAX_KV: u64 = 4096;
/// Longest key or string value read out of a header.
const GGUF_MAX_STRING: u64 = 1024 * 1024;
/// Longest array kept rather than skipped. Per layer head counts are the only
/// arrays worth reading; the tokenizer vocabulary is hundreds of thousands of
/// strings and is stepped over.
const GGUF_MAX_ARRAY: u64 = 4096;

fn read_gguf_metadata<R: Read + Seek>(
    reader: &mut R,
) -> Option<std::collections::HashMap<String, GgufValue>> {
    let magic = read_exact::<R, 4>(reader)?;
    if magic != GGUF_MAGIC {
        return None;
    }
    let version = read_u32(reader)?;
    if !(2..=3).contains(&version) {
        return None;
    }
    let _tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;
    if kv_count > GGUF_MAX_KV {
        return None;
    }

    let mut found = std::collections::HashMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(reader)?;
        let kind = read_u32(reader)?;
        let value = read_gguf_value(reader, kind)?;
        found.insert(key, value);
    }
    Some(found)
}

/// Bytes one fixed width GGUF value occupies, `None` for strings and arrays.
fn gguf_width(kind: u32) -> Option<i64> {
    Some(match kind {
        0 | 1 | 7 => 1,
        2 | 3 => 2,
        4..=6 => 4,
        10..=12 => 8,
        _ => return None,
    })
}

fn read_gguf_value<R: Read + Seek>(reader: &mut R, kind: u32) -> Option<GgufValue> {
    Some(match kind {
        0 | 1 | 7 => GgufValue::Number(u64::from(read_exact::<R, 1>(reader)?[0])),
        2 | 3 => GgufValue::Number(u64::from(u16::from_le_bytes(read_exact::<R, 2>(reader)?))),
        4 | 5 => GgufValue::Number(u64::from(read_u32(reader)?)),
        6 | 12 => {
            skip(reader, gguf_width(kind)?)?;
            GgufValue::Ignored
        }
        8 => GgufValue::Text(read_gguf_string(reader)?),
        9 => read_gguf_array(reader)?,
        10 | 11 => GgufValue::Number(read_u64(reader)?),
        _ => return None,
    })
}

/// Read an array of numbers, or step over one of anything else.
///
/// The tokenizer vocabulary is an array of a few hundred thousand strings and
/// sits in front of nothing this platform needs, but the header is a flat
/// stream, so it has to be walked past exactly rather than ignored. Getting
/// this wrong is how the whole header stops parsing and the planner silently
/// falls back to the coarse estimate.
fn read_gguf_array<R: Read + Seek>(reader: &mut R) -> Option<GgufValue> {
    let element = read_u32(reader)?;
    let length = read_u64(reader)?;

    match gguf_width(element) {
        Some(width) => {
            if length > GGUF_MAX_ARRAY || !matches!(element, 0..=5 | 7 | 10 | 11) {
                skip(reader, width.checked_mul(i64::try_from(length).ok()?)?)?;
                return Some(GgufValue::Ignored);
            }
            let mut numbers = Vec::with_capacity(usize::try_from(length).ok()?);
            for _ in 0..length {
                match read_gguf_value(reader, element)? {
                    GgufValue::Number(value) => numbers.push(value),
                    _ => return None,
                }
            }
            Some(GgufValue::Numbers(numbers))
        }
        // A string array, or an array of arrays: walk it element by element.
        None => {
            for _ in 0..length {
                read_gguf_value(reader, element)?;
            }
            Some(GgufValue::Ignored)
        }
    }
}

fn read_gguf_string<R: Read + Seek>(reader: &mut R) -> Option<String> {
    let length = read_u64(reader)?;
    if length > GGUF_MAX_STRING {
        return None;
    }
    let mut bytes = vec![0u8; usize::try_from(length).ok()?];
    reader.read_exact(&mut bytes).ok()?;
    // A vocabulary entry is not always valid UTF-8 and is stepped over anyway,
    // so a lossy read keeps the walk going instead of failing the header.
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

fn skip<R: Seek>(reader: &mut R, bytes: i64) -> Option<()> {
    reader.seek(SeekFrom::Current(bytes)).ok()?;
    Some(())
}

fn read_exact<R: Read, const N: usize>(reader: &mut R) -> Option<[u8; N]> {
    let mut bytes = [0u8; N];
    reader.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

fn read_u32<R: Read>(reader: &mut R) -> Option<u32> {
    Some(u32::from_le_bytes(read_exact::<R, 4>(reader)?))
}

fn read_u64<R: Read>(reader: &mut R) -> Option<u64> {
    Some(u64::from_le_bytes(read_exact::<R, 8>(reader)?))
}

/// The HF `pipeline_tag` a manifest kind maps onto.
fn pipeline_tag(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Llm => "text-generation",
        ModelKind::Vlm => "image-text-to-text",
    }
}

// --- Hugging Face payloads ------------------------------------------------

/// One repo as the search endpoint returns it.
#[derive(Debug, Deserialize)]
struct RawModel {
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    pipeline_tag: Option<String>,
    /// `false` when open, the gating mode as a string when not.
    #[serde(default)]
    gated: serde_json::Value,
    #[serde(default)]
    siblings: Vec<RawSibling>,
}

/// One repo as the model endpoint returns it with `blobs=true`.
#[derive(Debug, Deserialize)]
struct RawRepo {
    #[serde(default)]
    siblings: Vec<RawSibling>,
}

/// One file inside a repo.
#[derive(Debug, Deserialize)]
struct RawSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    lfs: Option<RawLfs>,
}

/// LFS pointer metadata, present for every file over 10 MB.
#[derive(Debug, Deserialize)]
struct RawLfs {
    #[serde(default)]
    size: Option<u64>,
}

impl RawSibling {
    fn size_bytes(&self) -> u64 {
        self.lfs
            .as_ref()
            .and_then(|lfs| lfs.size)
            .or(self.size)
            .unwrap_or(0)
    }
}

impl From<RawModel> for ModelSummary {
    fn from(raw: RawModel) -> Self {
        let quants = parse_files(&RawRepo {
            siblings: raw.siblings,
        })
        .into_iter()
        .map(|file| file.quant)
        .collect();
        Self {
            repo: raw.id,
            downloads: raw.downloads,
            likes: raw.likes,
            quants,
            pipeline_tag: raw.pipeline_tag,
            gated: !matches!(
                raw.gated,
                serde_json::Value::Bool(false) | serde_json::Value::Null
            ),
        }
    }
}

// --- Parsing --------------------------------------------------------------

/// The quant name at the end of a GGUF filename, `UD-` prefix kept.
fn quant_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)[-_.]((?:UD-)?(?:I?Q\d+[A-Z0-9_]*|TQ\d+[A-Z0-9_]*|MXFP4[A-Z0-9_]*|BF16|FP16|FP32|F16|F32))\.gguf$",
        )
        .expect("quant regex is valid")
    })
}

/// One piece of a split GGUF, which the platform never downloads.
fn shard_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)-\d+-of-\d+\.gguf$").expect("shard regex is valid"))
}

/// Parse the quant out of a GGUF filename, `None` when it carries no quant.
fn quant_from_filename(filename: &str) -> Option<String> {
    quant_regex()
        .captures(filename)
        .map(|caps| caps[1].to_uppercase())
}

/// True when a repo filename is one path component and nothing more.
///
/// Everything under `models/<slug>/` is named by Hugging Face, and a name is
/// joined onto that directory to make a path. `Path::join` on Windows treats
/// both separators and a drive letter as structure, so `..\..\x.gguf` and
/// `C:\x.gguf` would leave the directory; the platform only ever wants a leaf,
/// so anything that is not one is refused rather than sanitized.
fn is_safe_filename(name: &str) -> bool {
    !name.is_empty()
        && !name.starts_with('.')
        && !name.contains(['/', '\\', ':'])
        && !name.chars().any(char::is_control)
        && Path::new(name).components().count() == 1
}

/// [`is_safe_filename`] as a hard failure, for the places that cannot skip.
fn safe_filename(name: &str) -> Result<&str> {
    if is_safe_filename(name) {
        return Ok(name);
    }
    Err(Error::InvalidManifest(format!(
        "`{name}` is not a file name: a repo file must be a single path component"
    )))
}

/// True for the projector file a vision model needs next to its weights.
fn is_mmproj(filename: &str) -> bool {
    filename.to_ascii_lowercase().starts_with("mmproj")
}

/// How much a projector is wanted: f16 first, then bf16, then anything else.
fn mmproj_rank(name: &str) -> u8 {
    let name = name.to_ascii_lowercase();
    if name.contains("bf16") {
        1
    } else if name.contains("f16") {
        2
    } else {
        0
    }
}

/// Pick one mmproj out of a repo: f16 first, then the largest.
fn pick_mmproj(candidates: &[(String, u64)]) -> Option<String> {
    candidates
        .iter()
        .max_by_key(|(name, size)| (mmproj_rank(name), *size))
        .map(|(name, _)| name.clone())
}

/// Turn a repo payload into the quants the platform can serve.
fn parse_files(repo: &RawRepo) -> Vec<ModelFile> {
    let mut mmprojs: Vec<(String, u64)> = Vec::new();
    let mut quants: Vec<ModelFile> = Vec::new();

    for sibling in &repo.siblings {
        let name = sibling.rfilename.as_str();
        // Subfolders hold shards and unrelated formats, rule 1. A name that is
        // not a plain leaf is not a root level file either, and it is the one
        // that would escape the model directory, so the same check drops both.
        if !is_safe_filename(name) || !name.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        if is_mmproj(name) {
            mmprojs.push((name.to_string(), sibling.size_bytes()));
            continue;
        }
        // A split model is not one file, rule 2.
        if shard_regex().is_match(name) {
            continue;
        }
        let Some(quant) = quant_from_filename(name) else {
            continue;
        };
        quants.push(ModelFile {
            filename: name.to_string(),
            quant,
            size_bytes: sibling.size_bytes(),
            mmproj: None,
        });
    }

    let mmproj = pick_mmproj(&mmprojs);
    for file in &mut quants {
        file.mmproj = mmproj.clone();
    }
    quants.sort_by(|left, right| left.quant.cmp(&right.quant));
    quants
}

/// The raw `cursor` value carried by the `rel="next"` entry of a `Link` header.
fn next_cursor_from_link(link: &str) -> Option<String> {
    for part in link.split(',') {
        let mut pieces = part.split(';');
        let url = pieces.next()?.trim();
        let is_next = pieces.any(|piece| piece.trim().replace('"', "") == "rel=next");
        if !is_next {
            continue;
        }
        let url = url.trim_start_matches('<').trim_end_matches('>');
        let url = reqwest::Url::parse(url).ok()?;
        return url
            .query_pairs()
            .find(|(key, _)| key == "cursor")
            .map(|(_, value)| value.into_owned());
    }
    None
}

// --- Transfer -------------------------------------------------------------

/// One client for the whole process, so connections are reused across files.
fn http() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("aias/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .build()
            .expect("http client builds")
    })
}

/// Any network failure. `runtime.rs` carries the same helper.
fn net(err: impl std::fmt::Display) -> Error {
    Error::Http(err.to_string())
}

/// Turn a non success status into an error, keeping 404 distinguishable.
async fn check_status(response: reqwest::Response, what: &str) -> Result<reqwest::Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(Error::NotFound(what.to_string()));
    }
    // Hugging Face answers 401 for a repo the caller may not see, whether it is
    // private, gated or absent, so the three cannot be told apart here.
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
        return Err(Error::NotFound(format!(
            "{what}: private, gated or absent on Hugging Face; set {TOKEN_ENV} for a gated repo"
        )));
    }
    let body = response.text().await.unwrap_or_default();
    let body: String = body.chars().take(200).collect();
    Err(Error::Http(format!("http {status} for {what}: {body}")))
}

/// A second client for file transfers, following redirects by hand.
///
/// The SHA-256 of an LFS object only exists on the redirect `huggingface.co`
/// answers with; the CDN behind it replies with an `etag` of its own that is a
/// different value. Following redirects automatically would hide the header
/// that matters, so the hops are walked here.
fn downloader() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("aias/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("http client builds")
    })
}

/// Attach the token for gated repos when the environment carries one.
///
/// Only `huggingface.co` is trusted with it. The CDN URL it redirects to is
/// already signed, and a bearer token must not follow a redirect off origin.
fn authorize(request: reqwest::RequestBuilder, url: &reqwest::Url) -> reqwest::RequestBuilder {
    if url.host_str() != Some("huggingface.co") {
        return request;
    }
    match std::env::var(TOKEN_ENV) {
        Ok(token) if !token.trim().is_empty() => request.bearer_auth(token.trim()),
        _ => request,
    }
}

/// Read a 64 character hex digest out of one header.
fn hex_header(headers: &reqwest::header::HeaderMap, name: &str) -> Option<String> {
    let value = headers.get(name)?.to_str().ok()?;
    let value = value.trim_start_matches("W/").trim_matches('"');
    if value.len() == 64 && value.chars().all(|c| c.is_ascii_hexdigit()) {
        return Some(value.to_lowercase());
    }
    None
}

/// The SHA-256 Hugging Face publishes for a file, when it publishes one.
///
/// `x-linked-etag` is the hash of the LFS object. A plain `etag` is only the
/// hash when `huggingface.co` served the bytes itself, which happens for small
/// files that are not in LFS.
fn published_sha256(headers: &reqwest::header::HeaderMap, from_origin: bool) -> Option<String> {
    hex_header(headers, "x-linked-etag")
        .or_else(|| from_origin.then(|| hex_header(headers, "etag")).flatten())
}

/// Follow the redirect chain by hand, keeping the digest the origin published.
async fn open_stream(url: &str, resume_from: u64) -> Result<(reqwest::Response, Option<String>)> {
    let mut next =
        reqwest::Url::parse(url).map_err(|err| Error::Http(format!("bad url `{url}`: {err}")))?;
    let mut sha = None;

    for _ in 0..MAX_REDIRECTS {
        let from_origin = next.host_str() == Some("huggingface.co");
        let mut request = authorize(downloader().get(next.clone()), &next);
        if resume_from > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={resume_from}-"));
        }
        let response = request.send().await.map_err(net)?;

        if sha.is_none() {
            sha = published_sha256(response.headers(), from_origin);
        }
        if !response.status().is_redirection() {
            return Ok((response, sha));
        }

        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| Error::Http(format!("redirect without a location for {next}")))?;
        next = next
            .join(location)
            .map_err(|err| Error::Http(format!("bad redirect target: {err}")))?;
    }

    Err(Error::Http(format!("too many redirects for {url}")))
}

/// Stream one file into `dir`, resuming and verifying. Returns its path.
async fn download_file(
    repo: &str,
    filename: &str,
    dir: &Path,
    progress: ProgressFn<'_>,
) -> Result<PathBuf> {
    // The name comes from the Hugging Face API, so it is checked before it is
    // ever joined onto a path, here as well as in `parse_files`.
    let filename = safe_filename(filename)?;
    let target = dir.join(filename);
    if target.is_file() {
        return Ok(target);
    }
    let part = dir.join(format!("{filename}.part"));

    let mut resume_from = match tokio::fs::metadata(&part).await {
        Ok(meta) => meta.len(),
        Err(_) => 0,
    };

    let url = format!("{HF_HOST}/{repo}/resolve/main/{filename}");
    let (response, expected) = open_stream(&url, resume_from).await?;

    // A complete `.part` makes the server reject the range; that is a finished
    // download, not a failure.
    if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE && resume_from > 0 {
        let digest = hash_file(&part).await?;
        return finish(&part, &target, filename, dir, &digest, expected.as_deref()).await;
    }
    // The server ignored the range, so the bytes on disk are worthless.
    if resume_from > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        resume_from = 0;
    }
    let response = check_status(response, &url).await?;

    let total = response.content_length().map(|len| len + resume_from);

    let mut hasher = Sha256::new();
    let mut file = if resume_from > 0 {
        hash_into(&part, &mut hasher).await?;
        tokio::fs::OpenOptions::new()
            .append(true)
            .open(&part)
            .await?
    } else {
        tokio::fs::File::create(&part).await?
    };

    let mut downloaded = resume_from;
    let mut last_report = Instant::now();
    progress(Progress {
        downloaded_bytes: downloaded,
        total_bytes: total,
    });

    let mut response = response;
    while let Some(chunk) = response.chunk().await.map_err(net)? {
        file.write_all(&chunk).await?;
        hasher.update(&chunk);
        downloaded += chunk.len() as u64;
        if last_report.elapsed() >= PROGRESS_INTERVAL {
            last_report = Instant::now();
            progress(Progress {
                downloaded_bytes: downloaded,
                total_bytes: total,
            });
        }
    }
    file.flush().await?;
    file.sync_all().await?;
    drop(file);
    progress(Progress {
        downloaded_bytes: downloaded,
        total_bytes: total,
    });

    let digest = hex::encode(hasher.finalize());
    finish(&part, &target, filename, dir, &digest, expected.as_deref()).await
}

/// Verify, rename and record the sidecar for a finished transfer.
async fn finish(
    part: &Path,
    target: &Path,
    filename: &str,
    dir: &Path,
    digest: &str,
    expected: Option<&str>,
) -> Result<PathBuf> {
    if let Some(expected) = expected
        && expected != digest
    {
        // The bytes on disk are wrong, and appending to them would stay wrong.
        let _ = tokio::fs::remove_file(part).await;
        return Err(Error::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("sha256 mismatch for {filename}: expected {expected}, got {digest}"),
        )));
    }
    tokio::fs::rename(part, target).await?;
    tokio::fs::write(
        dir.join(format!("{filename}.sha256")),
        format!("{digest}  {filename}\n"),
    )
    .await?;
    Ok(target.to_path_buf())
}

/// SHA-256 of a whole file.
async fn hash_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    hash_into(path, &mut hasher).await?;
    Ok(hex::encode(hasher.finalize()))
}

/// Feed a file into a running hasher, so a resumed download still verifies.
async fn hash_into(path: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;

    /// A repo carrying every shape the rules in [`files`] have to handle.
    const REPO_FIXTURE: &str = r#"{
      "id": "unsloth/Qwen3.5-9B-GGUF",
      "siblings": [
        { "rfilename": "README.md", "size": 1024 },
        { "rfilename": "Qwen3.5-9B-Q8_0.gguf", "lfs": { "size": 9000000000 } },
        { "rfilename": "Qwen3.5-9B-UD-IQ2_XXS.gguf", "lfs": { "size": 3000000000 } },
        { "rfilename": "Qwen3.5-9B-ud-q4_k_xl.gguf", "lfs": { "size": 5000000000 } },
        { "rfilename": "Qwen3.5-9B-BF16-00001-of-00003.gguf", "lfs": { "size": 6000000000 } },
        { "rfilename": "Qwen3.5-9B-BF16-00002-of-00003.gguf", "lfs": { "size": 6000000000 } },
        { "rfilename": "Q4_K_M/Qwen3.5-9B-Q4_K_M.gguf", "lfs": { "size": 5500000000 } },
        { "rfilename": "mmproj-F32.gguf", "lfs": { "size": 1400000000 } },
        { "rfilename": "mmproj-BF16.gguf", "lfs": { "size": 700000001 } },
        { "rfilename": "mmproj-F16.gguf", "lfs": { "size": 700000000 } },
        { "rfilename": "ggml-model.gguf", "lfs": { "size": 10 } },
        { "rfilename": "tokenizer.json", "size": 2048 }
      ]
    }"#;

    fn fixture() -> Vec<ModelFile> {
        let repo: RawRepo = serde_json::from_str(REPO_FIXTURE).expect("fixture parses");
        parse_files(&repo)
    }

    #[test]
    fn a_repo_file_name_has_to_be_one_path_component() {
        assert!(is_safe_filename("Qwen3-8B-Q4_K_M.gguf"));
        assert!(is_safe_filename("mmproj-F16.gguf"));

        // The Windows separator is the one the old check missed: this name has
        // no `/`, so it passed the subfolder rule and then walked up two levels.
        assert!(!is_safe_filename("..\\..\\x.gguf"));
        assert!(!is_safe_filename("../x.gguf"));
        assert!(!is_safe_filename("..\\x.gguf"));
        assert!(!is_safe_filename("C:\\Windows\\x.gguf"));
        assert!(!is_safe_filename("C:x.gguf"));
        assert!(!is_safe_filename("sub/dir.gguf"));
        assert!(!is_safe_filename(".."));
        assert!(!is_safe_filename("."));
        assert!(!is_safe_filename(".hidden.gguf"));
        assert!(!is_safe_filename(""));
        assert!(!is_safe_filename("bad\u{0}name.gguf"));
    }

    #[test]
    fn a_traversing_file_name_never_reaches_a_path() {
        // Through the catalogue: a hostile sibling is dropped, not listed.
        let repo: RawRepo = serde_json::from_str(
            r#"{"id":"evil/repo","siblings":[
                {"rfilename":"..\\..\\Evil-Q8_0.gguf","lfs":{"size":10}},
                {"rfilename":"Model-Q4_K_M.gguf","lfs":{"size":20}}
            ]}"#,
        )
        .expect("fixture parses");
        let names: Vec<String> = parse_files(&repo)
            .into_iter()
            .map(|file| file.filename)
            .collect();
        assert_eq!(names, ["Model-Q4_K_M.gguf"]);

        // And at the download, which is the last place the name becomes a path.
        let err = safe_filename("..\\..\\x.gguf").expect_err("traversal must fail");
        assert!(err.to_string().contains("single path component"), "{err}");
    }

    #[test]
    fn slug_is_filesystem_safe() {
        assert_eq!(
            ModelRef::new("Qwen/Qwen3-14B-GGUF", "Q4_K_M").slug(),
            "Qwen_Qwen3-14B-GGUF-Q4_K_M"
        );
    }

    #[test]
    fn quant_comes_from_the_filename() {
        assert_eq!(
            quant_from_filename("Qwen3-0.6B-Q8_0.gguf").as_deref(),
            Some("Q8_0")
        );
        assert_eq!(
            quant_from_filename("Llama-3.2-3B-Instruct-Q4_K_M.gguf").as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(
            quant_from_filename("gemma-3-4b-it-q4_0.gguf").as_deref(),
            Some("Q4_0")
        );
        assert_eq!(
            quant_from_filename("Qwen3-8B-BF16.gguf").as_deref(),
            Some("BF16")
        );
        assert_eq!(
            quant_from_filename("Qwen3-8B-MXFP4_MOE.gguf").as_deref(),
            Some("MXFP4_MOE")
        );
        assert_eq!(quant_from_filename("tokenizer.json"), None);
    }

    #[test]
    fn unsloth_ud_prefix_survives() {
        assert_eq!(
            quant_from_filename("Qwen3.5-9B-UD-IQ2_XXS.gguf").as_deref(),
            Some("UD-IQ2_XXS")
        );
        assert_eq!(
            quant_from_filename("Qwen3.5-9B-ud-q4_k_xl.gguf").as_deref(),
            Some("UD-Q4_K_XL")
        );
    }

    #[test]
    fn shards_and_subfolders_are_skipped() {
        let files = fixture();
        let quants: Vec<&str> = files.iter().map(|file| file.quant.as_str()).collect();
        assert_eq!(quants, ["Q8_0", "UD-IQ2_XXS", "UD-Q4_K_XL"]);
    }

    #[test]
    fn mmproj_is_attached_not_listed() {
        let files = fixture();
        assert!(files.iter().all(|file| !is_mmproj(&file.filename)));
        assert!(
            files
                .iter()
                .all(|file| file.mmproj.as_deref() == Some("mmproj-F16.gguf"))
        );
    }

    #[test]
    fn size_prefers_the_lfs_value() {
        let files = fixture();
        let q8 = files.iter().find(|file| file.quant == "Q8_0").unwrap();
        assert_eq!(q8.size_bytes, 9_000_000_000);
    }

    #[test]
    fn search_hits_read_gated_and_pipeline_tag() {
        let raw: Vec<RawModel> = serde_json::from_str(
            r#"[
              { "id": "a/b", "downloads": 7, "likes": 2, "pipeline_tag": "text-generation", "gated": false },
              { "id": "c/d", "gated": "manual", "pipeline_tag": "image-text-to-text" }
            ]"#,
        )
        .expect("fixture parses");
        let items: Vec<ModelSummary> = raw.into_iter().map(ModelSummary::from).collect();

        assert_eq!(items[0].repo, "a/b");
        assert_eq!(items[0].downloads, 7);
        assert!(!items[0].gated);
        assert_eq!(items[0].pipeline_tag.as_deref(), Some("text-generation"));
        assert!(items[1].gated);
        assert_eq!(items[1].downloads, 0);
    }

    #[test]
    fn cursor_comes_out_of_the_link_header() {
        let link = "<https://huggingface.co/api/models?filter=gguf&cursor=eyJfaWQiOnsiJGd0IjoiNjY4In19&limit=30>; rel=\"next\"";
        assert_eq!(
            next_cursor_from_link(link).as_deref(),
            Some("eyJfaWQiOnsiJGd0IjoiNjY4In19")
        );
        assert_eq!(next_cursor_from_link("<https://x/y>; rel=\"prev\""), None);
    }

    #[test]
    fn kinds_map_onto_pipeline_tags() {
        assert_eq!(pipeline_tag(ModelKind::Llm), "text-generation");
        assert_eq!(pipeline_tag(ModelKind::Vlm), "image-text-to-text");
    }

    /// Qwen3-8B: 36 blocks, 8 KV heads of 128, which is 144 KB of cache a token.
    const QWEN3_8B: KvLayout = KvLayout {
        n_layers: 36,
        n_kv_heads: 8,
        head_dim: 128,
    };

    #[test]
    fn the_context_is_priced_before_it_is_handed_out() {
        // The case the reviewer gave: a 7B class model at Q4_K_M, about 5 GB on
        // disk, on a 24 GB card. The weights leave 18 GB free, so the old rule
        // read "plenty of headroom" and asked for 32k of context across four
        // slots. That is 131072 tokens at 144 KB each: 18 GB of KV cache on top
        // of the weights, over the budget, and llama-server died loading it.
        let size = 5028 * MB;
        let budget = 24 * 1024;
        let ceiling = ceiling_mb(budget);

        let greedy = estimate_mb(size, 32768, 4, Some(QWEN3_8B));
        assert!(
            greedy > ceiling,
            "32k over four slots costs {greedy} MB, which must not fit {ceiling} MB"
        );

        // What the planner hands out instead: the same context, half the slots.
        let (ctx, n_parallel) = plan_context(size, budget, Some(QWEN3_8B));
        assert_eq!((ctx, n_parallel), (32768, 2));
        let planned = estimate_mb(size, ctx, n_parallel, Some(QWEN3_8B));
        assert!(
            planned <= ceiling,
            "the planned {planned} MB must fit the {ceiling} MB ceiling"
        );

        // A smaller card is stepped further down rather than being refused.
        assert_eq!(plan_context(size, 12 * 1024, Some(QWEN3_8B)), (16384, 2));
        assert_eq!(plan_context(size, 8 * 1024, Some(QWEN3_8B)), (8192, 1));
    }

    #[test]
    fn the_cache_scales_with_the_context_and_the_slots() {
        let size = 5028 * MB;
        let weights = estimate_mb(size, 0, 1, Some(QWEN3_8B));

        // 144 KB a token, so doubling either factor doubles the cache.
        let one = estimate_mb(size, 4096, 1, Some(QWEN3_8B)) - weights;
        let doubled = estimate_mb(size, 8192, 1, Some(QWEN3_8B)) - weights;
        assert_eq!(doubled, one * 2);
        assert_eq!(
            estimate_mb(size, 4096, 2, Some(QWEN3_8B)) - weights,
            one * 2
        );
        assert_eq!(one, 4096 * 2 * 36 * 8 * 128 * 2 / (1024 * 1024));
    }

    #[test]
    fn fit_badges() {
        // 8 GB model on a 32 GB budget: room for the weights and a full context.
        assert_eq!(fit(8 * 1024 * MB, 32768), Fit::Ready);
        // 8 GB model on a 10 GB budget: the weights alone are 8.8 GB, so the
        // planner is down to the 2048 token floor and the badge says so.
        assert_eq!(fit(8 * 1024 * MB, 10240), Fit::Maybe);
        // 8 GB model on a 9.3 GB budget: not even the floor rung fits.
        assert_eq!(fit(8 * 1024 * MB, 9500), Fit::Incompatible);
        // 8 GB model on a 6 GB budget: the weights alone do not fit.
        assert_eq!(fit(8 * 1024 * MB, 6144), Fit::Incompatible);
    }

    #[test]
    fn fit_boundaries() {
        // 1000 MB of weights cost 1100 MB. The floor rung adds 62 MB of cache
        // and the smallest rung the badge calls Ready adds 123 MB.
        let size = 1000 * MB;
        assert_eq!(estimate_mb(size, 2048, 1, None), 1162);
        assert_eq!(estimate_mb(size, 4096, 1, None), 1223);

        // Ready ends where the 4096 token rung stops fitting 90% of the budget.
        assert_eq!(fit(size, 1359), Fit::Ready);
        assert_eq!(fit(size, 1358), Fit::Maybe);
        // Maybe ends where the floor stops fitting in the budget at all.
        assert_eq!(fit(size, 1162), Fit::Maybe);
        assert_eq!(fit(size, 1161), Fit::Incompatible);
    }

    #[test]
    fn the_badge_will_not_promise_a_context_the_planner_would_refuse() {
        // The case the reviewer gave: Qwen3-8B at Q4_K_M, about 5.0 GB on disk
        // and 8 billion parameters, on a machine with an 8 GB budget. The badge
        // used to price it at 0.5 MiB of cache per thousand tokens per billion
        // parameters, which said the top rung of the ladder fitted; admission
        // then read the GGUF header and refused the same model.
        let size = 5028 * MB;
        let budget = 8 * 1024;

        let greedy = estimate_mb(size, 32768, 4, None);
        assert!(
            greedy > ceiling_mb(budget),
            "32k over four slots costs {greedy} MB, which must not fit {} MB",
            ceiling_mb(budget)
        );

        // The badge now reads the rung the planner stops on, which is the rung
        // the instance would actually be started with.
        let planned = plan_context(size, budget, None);
        assert_eq!(planned, (8192, 1));
        assert_eq!(fit(size, budget), Fit::Ready);

        // The coarse estimate is within a tenth of the header it stands in for.
        let with_header = estimate_mb(size, 8192, 1, Some(QWEN3_8B));
        let without = estimate_mb(size, 8192, 1, None);
        assert!(
            without.abs_diff(with_header) * 10 < with_header,
            "the pre download estimate {without} MB must track the header {with_header} MB"
        );
    }

    #[test]
    fn a_gguf_header_gives_up_the_cache_layout() {
        let dir = std::env::temp_dir().join(format!("aias-gguf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.gguf");
        std::fs::write(&path, gguf_header()).unwrap();

        assert_eq!(kv_layout(&path), Some(QWEN3_8B));

        // Anything that is not a header this platform understands is `None`,
        // and the caller falls back rather than planning on a wrong number.
        let broken = dir.join("broken.gguf");
        std::fs::write(&broken, b"not a gguf file at all").unwrap();
        assert_eq!(kv_layout(&broken), None);
        assert_eq!(kv_layout(&dir.join("absent.gguf")), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A GGUF v3 header carrying exactly the keys [`kv_layout`] reads.
    fn gguf_header() -> Vec<u8> {
        fn string(out: &mut Vec<u8>, text: &str) {
            out.extend_from_slice(&(text.len() as u64).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        fn text_kv(out: &mut Vec<u8>, key: &str, value: &str) {
            string(out, key);
            out.extend_from_slice(&8u32.to_le_bytes());
            string(out, value);
        }
        fn u32_kv(out: &mut Vec<u8>, key: &str, value: u32) {
            string(out, key);
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&value.to_le_bytes());
        }

        // The vocabulary a real GGUF carries: hundreds of thousands of strings
        // in front of nothing useful, and the reason the header has to be
        // walked past exactly rather than given up on.
        fn vocabulary(out: &mut Vec<u8>, key: &str, words: u64) {
            string(out, key);
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&8u32.to_le_bytes());
            out.extend_from_slice(&words.to_le_bytes());
            for index in 0..words {
                string(out, &format!("token{index}"));
            }
        }

        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&7u64.to_le_bytes());
        text_kv(&mut out, "general.architecture", "qwen3");
        vocabulary(&mut out, "tokenizer.ggml.tokens", 20_000);
        u32_kv(&mut out, "qwen3.block_count", 36);
        u32_kv(&mut out, "qwen3.attention.head_count", 32);
        u32_kv(&mut out, "qwen3.attention.head_count_kv", 8);
        u32_kv(&mut out, "qwen3.attention.key_length", 128);
        // A per layer head count is published as an array, and the cache has to
        // be sized for the largest.
        string(&mut out, "qwen3.attention.head_count_kv_per_layer");
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        out
    }

    #[test]
    fn installed_is_empty_without_a_data_dir() {
        assert!(installed(Path::new("/nonexistent/aias")).is_empty());
    }

    #[test]
    fn removing_a_model_that_is_not_there_is_not_found() {
        let err = remove(
            &ModelRef::new("Qwen/Qwen3-14B-GGUF", "Q4_K_M"),
            Path::new("/nonexistent/aias"),
        )
        .unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }
}
