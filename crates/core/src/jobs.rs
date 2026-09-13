//! The registry behind every long operation the local API starts.
//!
//! A pull or a build outlives one HTTP request, so the route starts a job and
//! answers with its id; `GET /v1/jobs/<id>` is what reports progress and the
//! result (PLAN.md section 6.1).
//!
//! The registry is in process and holds no disk state, which is decision D10:
//! the running app is the one owner, and a job id only means anything to the
//! process that handed it out. The desktop shell and `aias api serve` each have
//! their own, and neither writes a file the other could read.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use rand::RngExt as _;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::models::Progress;

/// Bytes of a job id, hex encoded. Long enough that an id is not guessable by
/// a second process on the machine that reached the port without the token.
const ID_BYTES: usize = 8;

/// Identifier of one job, hex of [`ID_BYTES`] random bytes.
pub type JobId = String;

/// What a job is doing. One variant per long route in the API contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobKind {
    /// `POST /v1/models/pull`, a Hugging Face download.
    ModelsPull,
    /// `POST /v1/apps/build`, the manifest `build` commands.
    AppsBuild,
}

/// Where a job is in its life.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum JobState {
    /// Registered, not started yet.
    Queued,
    /// Running now.
    Running,
    /// Finished, `result` holds what it produced.
    Done,
    /// Finished, `error` says why it stopped.
    Failed,
}

/// The error shape of a failed job, the same `{code, message}` pair the API
/// answers a failed request with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobError {
    /// Stable tag from [`Error::code`].
    pub code: String,
    pub message: String,
}

impl From<&Error> for JobError {
    fn from(err: &Error) -> Self {
        Self {
            code: err.code().to_string(),
            message: err.to_string(),
        }
    }
}

/// One job as `GET /v1/jobs/<id>` reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
    pub id: JobId,
    pub kind: JobKind,
    pub state: JobState,
    /// Bytes so far, only ever set by a download.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Progress>,
    /// What the job produced, as JSON, once it is `done`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Why it stopped, once it is `failed`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
}

/// How long a finished job stays readable.
const MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
/// How many jobs the registry holds before the oldest finished ones go.
const MAX_JOBS: usize = 200;

/// One job and when it was registered.
struct Entry {
    at: Instant,
    job: Job,
}

/// Every job this process started, by id.
static JOBS: OnceLock<Mutex<HashMap<JobId, Entry>>> = OnceLock::new();

/// A poisoned registry is still a usable registry: one panicked job must not
/// take every later one down with it.
fn jobs() -> MutexGuard<'static, HashMap<JobId, Entry>> {
    JOBS.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Drop what nobody is going to ask for again.
///
/// The registry is in process and had no bound, so a long running session that
/// pulled models and rebuilt apps all day kept every result and every error
/// message of the day in memory. Two limits, applied when a job is registered:
/// anything older than [`MAX_AGE`], and beyond [`MAX_JOBS`] the oldest
/// finished ones. A job that is still queued or running is never evicted by
/// the count, because the id was handed to a caller who is still polling it.
fn evict(jobs: &mut HashMap<JobId, Entry>) {
    let now = Instant::now();
    jobs.retain(|_, entry| now.duration_since(entry.at) < MAX_AGE);

    let Some(excess) = jobs.len().checked_sub(MAX_JOBS) else {
        return;
    };
    let mut finished: Vec<(JobId, Instant)> = jobs
        .iter()
        .filter(|(_, entry)| matches!(entry.job.state, JobState::Done | JobState::Failed))
        .map(|(id, entry)| (id.clone(), entry.at))
        .collect();
    finished.sort_by_key(|(_, at)| *at);
    for (id, _) in finished.into_iter().take(excess) {
        jobs.remove(&id);
    }
}

/// Register a job and mark it running. The caller drives it and finishes it.
///
/// This is the path the desktop shell takes: `models_download` already owns the
/// await and the progress event, so it reports into a job rather than handing
/// the work over to one.
pub fn start(kind: JobKind) -> JobId {
    let mut bytes = [0u8; ID_BYTES];
    rand::rng().fill(&mut bytes);
    let id = hex::encode(bytes);

    let mut jobs = jobs();
    evict(&mut jobs);
    jobs.insert(
        id.clone(),
        Entry {
            at: Instant::now(),
            job: Job {
                id: id.clone(),
                kind,
                state: JobState::Running,
                progress: None,
                result: None,
                error: None,
            },
        },
    );
    drop(jobs);
    id
}

/// Record how far a download has got. Ignored once the job has finished.
pub fn report(id: &str, progress: Progress) {
    if let Some(entry) = jobs().get_mut(id)
        && matches!(entry.job.state, JobState::Queued | JobState::Running)
    {
        entry.job.progress = Some(progress);
    }
}

/// Finish a job with what it produced.
pub fn finish<T: Serialize>(id: &str, value: &T) {
    let result = serde_json::to_value(value).ok();
    if let Some(entry) = jobs().get_mut(id) {
        entry.job.state = JobState::Done;
        entry.job.result = result;
    }
}

/// Finish a job with why it stopped.
pub fn fail(id: &str, err: &Error) {
    let error = JobError::from(err);
    if let Some(entry) = jobs().get_mut(id) {
        entry.job.state = JobState::Failed;
        entry.job.error = Some(error);
    }
}

/// Finish a job from a [`Result`], which is what a spawned job ends with.
pub fn settle<T: Serialize>(id: &str, outcome: &Result<T>) {
    match outcome {
        Ok(value) => finish(id, value),
        Err(err) => fail(id, err),
    }
}

/// Register a job, run the future it hands back on the tokio runtime and return
/// the id at once.
///
/// The future is built from the id rather than passed in ready, because a pull
/// reports its progress into the job it is: the closure is the only place that
/// can hold the id before the job exists.
pub fn spawn_job<T, F, Fut>(kind: JobKind, make: F) -> JobId
where
    T: Serialize + Send + 'static,
    F: FnOnce(JobId) -> Fut,
    Fut: Future<Output = Result<T>> + Send + 'static,
{
    let id = start(kind);
    let fut = make(id.clone());
    let job = id.clone();
    tokio::spawn(async move {
        let outcome = fut.await;
        settle(&job, &outcome);
    });
    id
}

/// One job, or `None` when this process never handed that id out.
pub fn get(id: &str) -> Option<Job> {
    jobs().get(id).map(|entry| entry.job.clone())
}

/// Every job this process started, newest first is not promised: the registry
/// is a map and the API only ever asks for one id.
pub fn list() -> Vec<Job> {
    jobs().values().map(|entry| entry.job.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_registry_keeps_the_newest_jobs_and_drops_the_rest() {
        let mut registry: HashMap<JobId, Entry> = HashMap::new();
        let old = Instant::now() - MAX_AGE - Duration::from_secs(1);

        let entry = |id: &str, at: Instant, state: JobState| Entry {
            at,
            job: Job {
                id: id.to_string(),
                kind: JobKind::AppsBuild,
                state,
                progress: None,
                result: None,
                error: None,
            },
        };

        // A day old job goes, whatever it was doing.
        registry.insert("stale".into(), entry("stale", old, JobState::Done));
        registry.insert(
            "stale-running".into(),
            entry("stale-running", old, JobState::Running),
        );
        // A fresh one stays, and so does a running one over the cap.
        registry.insert(
            "running".into(),
            entry("running", Instant::now(), JobState::Running),
        );
        for index in 0..MAX_JOBS + 10 {
            let id = format!("done-{index:04}");
            registry.insert(
                id.clone(),
                entry(
                    &id,
                    Instant::now() - Duration::from_secs(1000 - index as u64),
                    JobState::Done,
                ),
            );
        }

        evict(&mut registry);

        assert!(!registry.contains_key("stale"));
        assert!(!registry.contains_key("stale-running"));
        assert!(
            registry.contains_key("running"),
            "a job somebody is polling is not evicted by the count"
        );
        assert!(registry.len() <= MAX_JOBS + 1, "{}", registry.len());
        // The oldest finished ones are the ones that went.
        assert!(!registry.contains_key("done-0000"));
        assert!(registry.contains_key(&format!("done-{:04}", MAX_JOBS + 9)));
    }

    #[test]
    fn a_job_carries_its_progress_and_its_result() {
        let id = start(JobKind::ModelsPull);
        assert_eq!(get(&id).unwrap().state, JobState::Running);

        report(
            &id,
            Progress {
                downloaded_bytes: 10,
                total_bytes: Some(100),
            },
        );
        assert_eq!(
            get(&id).unwrap().progress,
            Some(Progress {
                downloaded_bytes: 10,
                total_bytes: Some(100),
            })
        );

        finish(&id, &serde_json::json!({ "path": "/tmp/model.gguf" }));
        let job = get(&id).unwrap();
        assert_eq!(job.state, JobState::Done);
        assert_eq!(job.result.unwrap()["path"], "/tmp/model.gguf");
        // A late callback from a download that already finished is dropped.
        report(
            &id,
            Progress {
                downloaded_bytes: 100,
                total_bytes: Some(100),
            },
        );
        assert_eq!(get(&id).unwrap().progress.unwrap().downloaded_bytes, 10);
    }

    #[test]
    fn a_failed_job_reports_the_error_code() {
        let id = start(JobKind::AppsBuild);
        fail(&id, &Error::NotFound("nothing".into()));
        let job = get(&id).unwrap();
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.error.unwrap().code, "not_found");
    }

    #[test]
    fn an_unknown_id_is_none() {
        assert!(get("0000000000000000").is_none());
    }

    #[tokio::test]
    async fn a_spawned_job_settles_on_its_own() {
        let id = spawn_job(JobKind::AppsBuild, |_id| async { Ok("built") });
        for _ in 0..100 {
            if get(&id).unwrap().state != JobState::Running {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        let job = get(&id).unwrap();
        assert_eq!(job.state, JobState::Done, "{job:?}");
        assert_eq!(job.result.unwrap(), serde_json::json!("built"));
    }
}
