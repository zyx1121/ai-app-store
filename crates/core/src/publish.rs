//! Publishing an app to the store index (PLAN.md decision D5).
//!
//! The index is a git repo of manifests pointing at git repos, so publishing is
//! a pull request against `apps.yaml` and nothing else. This module checks that
//! the repo a user wants to publish is actually fetchable by someone else, it
//! builds the one entry that describes it, and it opens the pull request with
//! `gh` when that is available. Without `gh` it hands back the entry and the URL
//! to paste it into, because an agent on a machine without `gh` must still be
//! able to finish the job.
//!
//! Every value reaches `git` and `gh` as an argv element, never as a shell
//! string (PLAN.md section 3 rule 6).

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::apps::{self, Manifest};
use crate::error::{Error, Result};
use crate::process::Quiet as _;

/// The store index, `owner/repo` as `gh` names it.
pub const INDEX_REPO: &str = "zyx1121/aias-index";
/// The one file in the index repo.
pub const INDEX_FILE: &str = "apps.yaml";
/// Where a user edits the index by hand when `gh` is not there.
pub const INDEX_EDIT_URL: &str = "https://github.com/zyx1121/aias-index/edit/main/apps.yaml";

/// Prefix of the branch a publish pushes to the user's fork.
const BRANCH_PREFIX: &str = "add-";
/// Characters of the published commit kept in the branch name.
const SHORT_SHA: usize = 7;

/// One entry of `apps.yaml`, as it is written by a publish.
///
/// [`crate::apps::IndexEntry`] is the read side and carries the defaults a
/// hand written entry may omit. This is the write side, so it has exactly the
/// five keys a published app needs and nothing that would serialize as null.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishEntry {
    pub name: String,
    /// HTTPS git URL, the only thing the index points at.
    pub repo: String,
    /// Branch the entry pins, the branch the repo is published from.
    #[serde(rename = "ref")]
    pub git_ref: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

/// What a publish produced: a pull request, or the entry to paste by hand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PublishResult {
    pub name: String,
    pub entry: PublishEntry,
    /// The entry as it appears in `apps.yaml`, one list item.
    pub entry_yaml: String,
    /// The pull request, when `gh` opened one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pr_url: Option<String>,
    /// Where the file is edited in a browser.
    pub index_url: String,
    /// What the user or the agent has to do next, in one sentence.
    pub instructions: String,
}

/// What the git checks read off the repo being published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoState {
    /// The origin remote, normalized to an HTTPS GitHub URL.
    pub repo: String,
    /// The checked out branch.
    pub branch: String,
    /// The commit the entry will point at, through the branch.
    pub sha: String,
}

/// Validate, check the repo is pushed, and open the pull request.
///
/// The order matters: a manifest that does not validate is not worth a fork,
/// and a repo nobody else can clone is not worth a pull request.
pub async fn publish(dir: &Path) -> Result<PublishResult> {
    let manifest = apps::validate_dir(dir)?;
    let state = repo_state(dir).await?;
    let entry = entry_for(&manifest, &state);
    let entry_yaml = entry_yaml(&entry)?;

    if !gh_ready().await {
        return Ok(PublishResult {
            name: entry.name.clone(),
            entry,
            entry_yaml,
            pr_url: None,
            index_url: INDEX_EDIT_URL.to_string(),
            instructions: format!(
                "`gh` is not available or not logged in, so no pull request was opened. Open {INDEX_EDIT_URL}, append the entry above to the list, and open a pull request. Install `gh` and run `gh auth login` to have this done for you."
            ),
        });
    }

    let pr_url = open_pr(&entry, &state.sha).await?;
    Ok(PublishResult {
        name: entry.name.clone(),
        entry,
        entry_yaml,
        instructions: format!(
            "The pull request is open at {pr_url}. The app appears in the store on every machine once it is merged."
        ),
        pr_url: Some(pr_url),
        index_url: INDEX_EDIT_URL.to_string(),
    })
}

/// The index entry a manifest and a repo produce.
pub fn entry_for(manifest: &Manifest, state: &RepoState) -> PublishEntry {
    PublishEntry {
        name: manifest.name.clone(),
        repo: state.repo.clone(),
        git_ref: state.branch.clone(),
        description: manifest.description.clone(),
        tags: manifest.tags.clone(),
    }
}

/// Render one entry as the list item it becomes in `apps.yaml`.
///
/// Serialization does the quoting, so a description with a colon in it cannot
/// break the file.
pub fn entry_yaml(entry: &PublishEntry) -> Result<String> {
    let mapping = serde_yaml_ng::to_string(entry)?;
    let mut out = String::new();
    for (index, line) in mapping.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        if index == 0 {
            out.push_str("- ");
        } else {
            out.push_str("  ");
        }
        out.push_str(line);
        out.push('\n');
    }
    Ok(out)
}

/// Put one entry into an `apps.yaml`, replacing the entry of the same name.
///
/// The file is text and not a parse tree on purpose: everyone else's entries
/// keep their comments, their key order and their blank lines. Only the entry
/// being published is written by this code.
pub fn merge_entry(existing: &str, entry: &PublishEntry) -> Result<String> {
    let block = entry_yaml(entry)?;
    let lines: Vec<&str> = existing.lines().collect();
    let indent = list_indent(&lines);
    let marker = format!("{}- ", " ".repeat(indent));
    let indented: Vec<String> = block
        .lines()
        .map(|line| format!("{}{line}", " ".repeat(indent)))
        .collect();

    // Every list item, as a range of lines: from its `- ` to the next one.
    let starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with(&marker))
        .map(|(index, _)| index)
        .collect();

    let mut out: Vec<String> = Vec::new();
    let mut replaced = false;
    let mut skip_until = 0usize;

    for (index, line) in lines.iter().enumerate() {
        if index < skip_until {
            continue;
        }
        if starts.contains(&index) {
            let end = starts
                .iter()
                .copied()
                .find(|start| *start > index)
                .unwrap_or(lines.len());
            let item = &lines[index..end];
            if item_name(item, indent)? == Some(entry.name.clone()) {
                out.extend(indented.iter().cloned());
                replaced = true;
            } else {
                out.extend(item.iter().map(|line| line.to_string()));
            }
            skip_until = end;
            continue;
        }
        out.push(line.to_string());
    }

    if !replaced {
        // A file that ends without a newline, or with a trailing blank line,
        // must not swallow or split the new entry.
        while out.last().is_some_and(|line| line.trim().is_empty()) {
            out.pop();
        }
        out.extend(indented);
    }

    let mut text = out.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    Ok(text)
}

/// Indentation of the list items in an `apps.yaml`.
///
/// A bare list is at column 0 and a list under `apps:` is indented, and both
/// shapes are published today (see `apps::IndexFile`).
fn list_indent(lines: &[&str]) -> usize {
    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with("- ") {
            return line.len() - trimmed.len();
        }
    }
    if lines
        .iter()
        .any(|line| line.trim_start().starts_with("apps:"))
    {
        return 2;
    }
    0
}

/// The `name` of one list item, read by parsing the item on its own.
fn item_name(item: &[&str], indent: usize) -> Result<Option<String>> {
    let text: String = item
        .iter()
        .map(|line| {
            let line = if line.len() >= indent {
                &line[indent..]
            } else {
                line.trim_start()
            };
            format!("{line}\n")
        })
        .collect();
    let parsed: serde_yaml_ng::Value = match serde_yaml_ng::from_str(&text) {
        Ok(parsed) => parsed,
        // A hand written entry that does not parse is left alone rather than
        // failing the publish: it is not the entry being written.
        Err(_) => return Ok(None),
    };
    let name = parsed
        .as_sequence()
        .and_then(|items| items.first())
        .and_then(|item| item.get("name"))
        .and_then(|name| name.as_str())
        .map(str::to_string);
    Ok(name)
}

/// Read the repo the app will be published from, and refuse what nobody else
/// could clone.
///
/// Four failures are worth one message each: a dirty tree publishes code that
/// is not in the repo, a missing `origin` is a repo nobody else has seen, a non
/// GitHub remote is not something the index can point at, and a commit that is
/// only local means the store would index a repo without the code in it. The
/// missing remote is the one an agent hits first on a freshly scaffolded app,
/// so it says what to do rather than repeating git's exit status.
pub async fn repo_state(dir: &Path) -> Result<RepoState> {
    let status = git_output(dir, &["status", "--porcelain"]).await?;
    if !status.trim().is_empty() {
        return Err(Error::Process(format!(
            "the repo has uncommitted changes, commit and push them first:\n{}",
            status.trim()
        )));
    }

    let remote = git_output(dir, &["remote", "get-url", "origin"])
        .await
        .map_err(|_| {
            Error::Process(
                "the repo has no `origin` remote: create a GitHub repo, run `git remote add origin https://github.com/<owner>/<repo>`, push the branch, then publish again".into(),
            )
        })?;
    let repo = https_github(remote.trim())?;

    let branch = git_output(dir, &["rev-parse", "--abbrev-ref", "HEAD"]).await?;
    let branch = branch.trim().to_string();
    if branch == "HEAD" {
        return Err(Error::Process(
            "the repo is in a detached HEAD, check out the branch you want to publish".into(),
        ));
    }
    check_git_ref(&branch)?;

    let sha = git_output(dir, &["rev-parse", "HEAD"])
        .await?
        .trim()
        .to_string();

    git(dir, &["fetch", "--quiet", "origin"]).await?;
    let remote_branch = format!("origin/{branch}");
    let pushed = git_status(
        dir,
        &["merge-base", "--is-ancestor", "HEAD", &remote_branch],
    )
    .await?;
    if !pushed {
        return Err(Error::Process(format!(
            "the commit on `{branch}` is not on `{remote_branch}`, push it first: git push origin {branch}"
        )));
    }

    Ok(RepoState { repo, branch, sha })
}

/// `https://github.com/<owner>/<repo>`, or a message saying why not.
///
/// SSH remotes are rejected rather than rewritten: the index is cloned by
/// machines that have no key for the repo, so an SSH remote is a repo the
/// store cannot use even when the publisher can.
pub fn https_github(remote: &str) -> Result<String> {
    let trimmed = remote.trim().trim_end_matches('/');
    let rest = trimmed.strip_prefix("https://github.com/").ok_or_else(|| {
        Error::Process(format!(
            "`{trimmed}` is not an HTTPS GitHub remote: the index can only point at https://github.com/<owner>/<repo>"
        ))
    })?;
    let rest = rest.strip_suffix(".git").unwrap_or(rest);
    let mut parts = rest.split('/');
    let owner = parts.next().unwrap_or_default();
    let repo = parts.next().unwrap_or_default();
    let extra = parts.next();
    let ok = !owner.is_empty()
        && !repo.is_empty()
        && extra.is_none()
        && [owner, repo].iter().all(|part| {
            part.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        });
    if !ok {
        return Err(Error::Process(format!(
            "`{trimmed}` is not an HTTPS GitHub remote: the index can only point at https://github.com/<owner>/<repo>"
        )));
    }
    Ok(format!("https://github.com/{owner}/{repo}"))
}

/// True when `gh` is on PATH and logged in.
pub async fn gh_ready() -> bool {
    gh_output(None, &["auth", "status"]).await.is_ok()
}

/// True when the logged in account may push a branch to the index itself.
///
/// GitHub reports the viewer's permission on a repo as one of `ADMIN`,
/// `MAINTAIN`, `WRITE`, `TRIAGE`, `READ` or `NONE`; only the first three can
/// push. Anything unreadable is treated as read only, which takes the fork
/// path, which is what everyone outside the index repo does anyway.
async fn can_push_to_index() -> bool {
    let permission = gh_output(
        None,
        &[
            "repo",
            "view",
            INDEX_REPO,
            "--json",
            "viewerPermission",
            "--jq",
            ".viewerPermission",
        ],
    )
    .await;
    matches!(
        permission.as_deref().map(str::trim),
        Ok("ADMIN" | "MAINTAIN" | "WRITE")
    )
}

/// Fork the index, write the entry, push the branch and open the pull request.
///
/// The clone is a scratch directory: the index is a few kilobytes and keeping
/// it would be a second copy of a file that changes under us.
///
/// A publisher who can already push to the index skips the fork: GitHub refuses
/// to let one account own both a repo and a fork of it, so the maintainer of
/// the index would otherwise never be able to publish an app.
///
/// `sha` is the commit the entry points at, and it is in the branch name, so a
/// second publish of the same app is not the non fast forward push the fixed
/// name `add-<name>` used to be.
pub async fn open_pr(entry: &PublishEntry, sha: &str) -> Result<String> {
    let branch = branch_for(&entry.name, sha);
    check_git_ref(&branch)?;

    let owner = gh_output(None, &["api", "user", "--jq", ".login"])
        .await?
        .trim()
        .to_string();
    if owner.is_empty() {
        return Err(Error::Process(
            "`gh api user` did not name the logged in account".into(),
        ));
    }

    // Push to the index itself when that is allowed, and to a fork otherwise.
    let forked = !can_push_to_index().await;
    let remote_repo = if forked {
        // Forking twice is not an error for `gh`, so this is the whole
        // `if needed`. `--remote` is not accepted next to a repository
        // argument, and the default is not to add one, so it is not passed.
        gh_output(None, &["repo", "fork", INDEX_REPO, "--clone=false"]).await?;
        format!("{owner}/aias-index")
    } else {
        INDEX_REPO.to_string()
    };

    let scratch = Scratch::new(&entry.name);
    let dir = scratch.dir();
    gh_output(
        None,
        &[
            "repo",
            "clone",
            &remote_repo,
            &dir.to_string_lossy(),
            "--",
            "--quiet",
        ],
    )
    .await?;

    // The checkout comes first: on a branch that is already on the remote the
    // entry is merged into that branch's `apps.yaml` and not into `main`'s.
    let branch = checkout_branch(dir, &branch).await?;

    let index_path = dir.join(INDEX_FILE);
    let existing = std::fs::read_to_string(&index_path).unwrap_or_default();
    let merged = merge_entry(&existing, entry)?;
    std::fs::write(&index_path, merged)?;

    git(dir, &["add", "--", INDEX_FILE]).await?;
    // A reused branch that already carries this exact entry has nothing to
    // commit, and `git commit` calls that a failure.
    let staged = git_output(dir, &["status", "--porcelain"]).await?;
    if !staged.trim().is_empty() {
        git(
            dir,
            &[
                "commit",
                "--message",
                &format!("Add {} to the store index", entry.name),
            ],
        )
        .await?;
    }
    let branch = push_branch(dir, branch).await?;

    let head = if forked {
        format!("{owner}:{branch}")
    } else {
        branch.clone()
    };
    let body = pr_body(entry)?;
    let created = gh_output(
        Some(dir),
        &[
            "pr",
            "create",
            "--repo",
            INDEX_REPO,
            "--head",
            &head,
            "--title",
            &format!("Add {} to the store index", entry.name),
            "--body",
            &body,
        ],
    )
    .await;

    let pr = match created {
        Ok(pr) => pr,
        // A branch that is reused already has the pull request this publish
        // would open, and that pull request is the answer, not an error.
        Err(err) => match existing_pr(&branch).await {
            Some(url) => return Ok(url),
            None => return Err(err),
        },
    };

    let url = pr
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| line.starts_with("https://"))
        .ok_or_else(|| Error::Process(format!("`gh pr create` printed no URL: {}", pr.trim())))?;
    Ok(url.to_string())
}

/// The branch one publish pushes: the app and the commit it publishes.
///
/// Every publish used to push `add-<name>`, so the second publish of an app
/// rewrote a branch that was still on the fork and git refused the push as a
/// non fast forward. With the commit in the name a new commit is a new branch,
/// and the same commit twice is the same branch, which is the one case where
/// reusing the branch is the right answer.
pub fn branch_for(name: &str, sha: &str) -> String {
    let short: String = sha
        .trim()
        .chars()
        .take(SHORT_SHA)
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    if short.is_empty() {
        return format!("{BRANCH_PREFIX}{name}");
    }
    format!("{BRANCH_PREFIX}{name}-{short}")
}

/// Check out the branch this publish commits on, and say which one that is.
///
/// A branch that is already on the remote is reused from its own tip, so the
/// push that follows is a fast forward. A branch that is there with a history
/// of its own is left alone and this publish takes a name of its own instead,
/// because rewriting someone else's branch would drop whatever is on it.
async fn checkout_branch(dir: &Path, branch: &str) -> Result<String> {
    let listed = git_output(dir, &["ls-remote", "--heads", "origin", branch]).await?;
    if listed.trim().is_empty() {
        git(dir, &["checkout", "-B", branch]).await?;
        return Ok(branch.to_string());
    }

    let remote_ref = format!("refs/remotes/origin/{branch}");
    let fetched = git_status(
        dir,
        &[
            "fetch",
            "--quiet",
            "origin",
            &format!("+{branch}:{remote_ref}"),
        ],
    )
    .await?;
    let shared = fetched && git_status(dir, &["merge-base", "HEAD", &remote_ref]).await?;
    if shared {
        git(dir, &["checkout", "-B", branch, &remote_ref]).await?;
        return Ok(branch.to_string());
    }

    let fresh = stamped(branch);
    check_git_ref(&fresh)?;
    git(dir, &["checkout", "-B", &fresh]).await?;
    Ok(fresh)
}

/// Push the branch, and move the commit to a branch of its own if git refuses.
///
/// The refusal left to handle here is a race: the branch was reusable when it
/// was read and somebody moved it before the push. One retry on a name nothing
/// else can hold is enough, and the entry reaches the index either way.
async fn push_branch(dir: &Path, branch: String) -> Result<String> {
    if git_status(dir, &["push", "--set-upstream", "origin", &branch]).await? {
        return Ok(branch);
    }
    let fresh = stamped(&branch);
    check_git_ref(&fresh)?;
    git(dir, &["checkout", "-B", &fresh]).await?;
    git(dir, &["push", "--set-upstream", "origin", &fresh]).await?;
    Ok(fresh)
}

/// `branch` with the current second appended, a name nothing else holds.
fn stamped(branch: &str) -> String {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or_default();
    format!("{branch}-{stamp}")
}

/// The open pull request whose head is `branch`, when the index has one.
///
/// Publishing the same commit twice reaches `gh pr create` with a branch that
/// already has a pull request, which `gh` calls an error. The pull request is
/// what the publisher asked for, so it is read back instead.
async fn existing_pr(branch: &str) -> Option<String> {
    let listed = gh_output(
        None,
        &[
            "pr", "list", "--repo", INDEX_REPO, "--head", branch, "--state", "open", "--json",
            "url", "--jq", ".[0].url",
        ],
    )
    .await
    .ok()?;
    listed
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("https://"))
        .map(str::to_string)
}

/// The pull request body: the entry, so a reviewer reads it without a diff.
fn pr_body(entry: &PublishEntry) -> Result<String> {
    Ok(format!(
        "Adds `{}` to the store index.\n\n- Name: `{}`\n- Repo: {}\n- Ref: `{}`\n- Description: {}\n- Tags: {}\n\n```yaml\n{}```\n",
        entry.name,
        entry.name,
        entry.repo,
        entry.git_ref,
        clamp(&entry.description, DESCRIPTION_LIMIT),
        if entry.tags.is_empty() {
            "none".to_string()
        } else {
            entry.tags.join(", ")
        },
        entry_yaml(entry)?
    ))
}

/// A unique scratch directory for one publish.
/// Longest description put in a pull request body.
///
/// The manifest is quoted in full below it, so this line is a summary and not
/// the record. A description of any length is valid in a manifest, and a long
/// one turns the body into a wall nobody reviewing the index will read.
const DESCRIPTION_LIMIT: usize = 500;

/// `text` cut to `limit` characters, on a character boundary, with an ellipsis.
fn clamp(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let kept: String = text.chars().take(limit).collect();
    format!("{}...", kept.trim_end())
}

/// A scratch clone that is removed however the caller leaves.
///
/// `open_pr` has a dozen exit paths, one per `gh` and `git` call, and each one
/// used to leave a clone of the index behind in the temp directory. The clone
/// is small but it carries a git remote with the user's fork in it, so it is
/// not something to scatter.
struct Scratch(PathBuf);

impl Scratch {
    /// A fresh empty path. An old directory at the same name is removed first.
    fn new(name: &str) -> Self {
        let dir = scratch_dir(name);
        std::fs::remove_dir_all(&dir).ok();
        Self(dir)
    }

    fn dir(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}

fn scratch_dir(name: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!("aias-publish-{name}-{stamp}"))
}

/// Refuse a branch git would read as an option or as a path.
fn check_git_ref(git_ref: &str) -> Result<()> {
    let mut chars = git_ref.chars();
    let head_ok = matches!(chars.next(), Some(first) if first.is_ascii_alphanumeric());
    let tail_ok = chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '-'));
    if head_ok && tail_ok {
        return Ok(());
    }
    Err(Error::Process(format!(
        "`{git_ref}` is not a usable git branch: it must match ^[A-Za-z0-9][A-Za-z0-9._/-]*$"
    )))
}

/// Run git in a directory and return its stdout.
async fn git_output(dir: &Path, args: &[&str]) -> Result<String> {
    let output = run("git", Some(dir), args).await?;
    if !output.status.success() {
        return Err(Error::Process(format!(
            "git {} failed with {}: {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Run git in a directory, keeping only whether it succeeded.
async fn git(dir: &Path, args: &[&str]) -> Result<()> {
    git_output(dir, args).await.map(|_| ())
}

/// Run a git command whose failure is an answer and not an error.
async fn git_status(dir: &Path, args: &[&str]) -> Result<bool> {
    Ok(run("git", Some(dir), args).await?.status.success())
}

/// Run `gh` and return its stdout, with its stderr in the error.
async fn gh_output(dir: Option<&Path>, args: &[&str]) -> Result<String> {
    let output = run("gh", dir, args).await?;
    if !output.status.success() {
        return Err(Error::Process(format!(
            "gh {} failed with {}: {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// One place where a child process is started, so every call is argv only.
async fn run(program: &str, dir: Option<&Path>, args: &[&str]) -> Result<std::process::Output> {
    let mut command = tokio::process::Command::new(program);
    command.args(args.iter().map(OsStr::new));
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    command
        .quiet()
        .output()
        .await
        .map_err(|err| Error::Process(format!("could not run {program}: {err}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry() -> PublishEntry {
        PublishEntry {
            name: "contract-review".into(),
            repo: "https://github.com/acme/contract-review".into(),
            git_ref: "main".into(),
            description: "Review a contract: flag deviations.".into(),
            tags: vec!["legal".into(), "documents".into()],
        }
    }

    fn manifest() -> Manifest {
        serde_yaml_ng::from_str(
            r#"
name: contract-review
description: "Review a contract: flag deviations."
version: 0.3.1
runtime: node
start: ["node", "server.js"]
tags: [legal, documents]
"#,
        )
        .unwrap()
    }

    #[test]
    fn an_entry_takes_its_name_and_description_from_the_manifest() {
        let state = RepoState {
            repo: "https://github.com/acme/contract-review".into(),
            branch: "main".into(),
            sha: "abc1234".into(),
        };
        assert_eq!(entry_for(&manifest(), &state), entry());
    }

    #[test]
    fn a_scratch_clone_is_removed_however_the_caller_leaves() {
        let dir = {
            let scratch = Scratch::new("leave-nothing");
            std::fs::create_dir_all(scratch.dir()).unwrap();
            std::fs::write(scratch.dir().join("apps.yaml"), "apps: []\n").unwrap();
            let dir = scratch.dir().to_path_buf();
            assert!(dir.is_dir());
            dir
            // Every exit path of `open_pr` ends here, including the dozen that
            // return an error in the middle of the clone.
        };
        assert!(!dir.exists(), "{} was left behind", dir.display());
    }

    #[test]
    fn a_long_description_is_cut_before_it_reaches_the_pull_request() {
        assert_eq!(clamp("short", DESCRIPTION_LIMIT), "short");

        let long = "a".repeat(DESCRIPTION_LIMIT + 100);
        let cut = clamp(&long, DESCRIPTION_LIMIT);
        assert_eq!(cut.chars().count(), DESCRIPTION_LIMIT + 3);
        assert!(cut.ends_with("..."));

        // Cutting happens on characters, so a multi byte one is never split.
        let wide = "\u{4e00}".repeat(10);
        assert_eq!(clamp(&wide, 4), "\u{4e00}\u{4e00}\u{4e00}\u{4e00}...");
    }

    #[test]
    fn an_entry_renders_as_one_list_item() {
        let yaml = entry_yaml(&entry()).unwrap();
        assert!(yaml.starts_with("- name: contract-review\n"));
        for line in yaml.lines().skip(1) {
            assert!(line.starts_with("  "), "not indented: {line}");
        }
        // The index reader has to accept what the publisher wrote.
        let parsed: Vec<apps::IndexEntry> = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed[0].name, "contract-review");
        assert_eq!(parsed[0].git_ref, "main");
        assert_eq!(parsed[0].tags, vec!["legal", "documents"]);
    }

    #[test]
    fn a_description_with_a_colon_survives_the_round_trip() {
        let yaml = entry_yaml(&entry()).unwrap();
        let parsed: Vec<apps::IndexEntry> = serde_yaml_ng::from_str(&yaml).unwrap();
        assert_eq!(parsed[0].description, "Review a contract: flag deviations.");
    }

    #[test]
    fn merging_into_an_empty_file_writes_the_first_entry() {
        let merged = merge_entry("", &entry()).unwrap();
        let parsed: Vec<apps::IndexEntry> = serde_yaml_ng::from_str(&merged).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn merging_appends_and_keeps_the_other_entries_untouched() {
        let existing = "# The store index\n- name: note-taker\n  repo: https://github.com/acme/note-taker\n  ref: main\n  description: Summarize meetings\n";
        let merged = merge_entry(existing, &entry()).unwrap();
        assert!(merged.starts_with("# The store index\n- name: note-taker\n"));
        let parsed: Vec<apps::IndexEntry> = serde_yaml_ng::from_str(&merged).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].name, "contract-review");
    }

    #[test]
    fn merging_replaces_the_entry_with_the_same_name() {
        let existing = "- name: contract-review\n  repo: https://github.com/old/contract-review\n  ref: v1\n  description: Old\n- name: note-taker\n  repo: https://github.com/acme/note-taker\n  ref: main\n  description: Summarize meetings\n";
        let merged = merge_entry(existing, &entry()).unwrap();
        let parsed: Vec<apps::IndexEntry> = serde_yaml_ng::from_str(&merged).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "contract-review");
        assert_eq!(parsed[0].repo, "https://github.com/acme/contract-review");
        assert_eq!(parsed[0].git_ref, "main");
        assert_eq!(parsed[1].name, "note-taker");
    }

    #[test]
    fn merging_keeps_the_wrapped_shape_of_the_file() {
        let existing = "apps:\n  - name: note-taker\n    repo: https://github.com/acme/note-taker\n    ref: main\n    description: Summarize meetings\n";
        let merged = merge_entry(existing, &entry()).unwrap();
        assert!(merged.starts_with("apps:\n  - name: note-taker\n"));
        for line in merged.lines().skip(1) {
            assert!(line.starts_with("  "), "lost the indent: {line}");
        }
        let parsed: apps::IndexEntry = serde_yaml_ng::from_str::<serde_yaml_ng::Value>(&merged)
            .unwrap()
            .get("apps")
            .and_then(|apps| apps.as_sequence())
            .and_then(|apps| apps.last())
            .cloned()
            .map(serde_yaml_ng::from_value)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.name, "contract-review");
    }

    #[test]
    fn an_https_github_remote_is_normalized_and_everything_else_is_refused() {
        assert_eq!(
            https_github("https://github.com/acme/contract-review.git").unwrap(),
            "https://github.com/acme/contract-review"
        );
        assert_eq!(
            https_github("https://github.com/acme/contract-review/").unwrap(),
            "https://github.com/acme/contract-review"
        );
        assert!(https_github("git@github.com:acme/contract-review.git").is_err());
        assert!(https_github("https://gitlab.com/acme/contract-review").is_err());
        assert!(https_github("https://github.com/acme").is_err());
    }

    #[test]
    fn a_branch_git_would_read_as_an_option_is_refused() {
        assert!(check_git_ref("main").is_ok());
        assert!(check_git_ref("add-contract-review").is_ok());
        assert!(check_git_ref("--upload-pack=touch").is_err());
        assert!(check_git_ref("a branch").is_err());
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "aias-publish-{}-{tag}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Who the commits in these tests belong to.
    ///
    /// The identity is in the environment and not in a config file, so a
    /// machine without a global git identity still runs the suite.
    fn git_identity() {
        for (key, value) in [
            ("GIT_AUTHOR_NAME", "aias test"),
            ("GIT_AUTHOR_EMAIL", "test@example.com"),
            ("GIT_COMMITTER_NAME", "aias test"),
            ("GIT_COMMITTER_EMAIL", "test@example.com"),
        ] {
            // Safety: the suite sets the same values from every test, so no
            // reader can see a half written identity.
            unsafe { std::env::set_var(key, value) };
        }
    }

    fn run_git(dir: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .args(args)
            .current_dir(dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("git runs");
        assert!(status.success(), "git {args:?} failed");
    }

    /// An app repo with a valid manifest, committed, with the given origin.
    fn app_repo(dir: &Path, origin: &str) {
        std::fs::write(
            dir.join("aias.yaml"),
            "name: contract-review\ndescription: \"Review a contract: flag deviations.\"\nversion: 0.3.1\nruntime: node\nstart: [\"node\", \"server.js\"]\ntags: [legal, documents]\n",
        )
        .unwrap();
        std::fs::write(dir.join("Dockerfile"), "FROM scratch\nEXPOSE $PORT\n").unwrap();
        run_git(dir, &["init", "--initial-branch=main"]);
        run_git(dir, &["add", "."]);
        run_git(dir, &["commit", "--message", "initial"]);
        run_git(dir, &["remote", "add", "origin", origin]);
    }

    #[tokio::test]
    async fn an_uncommitted_change_stops_the_publish() {
        git_identity();
        let dir = TempDir::new("dirty");
        app_repo(&dir.0, "https://github.com/acme/contract-review.git");
        std::fs::write(dir.0.join("README.md"), "not committed\n").unwrap();

        let err = repo_state(&dir.0).await.unwrap_err();
        assert!(err.to_string().contains("uncommitted changes"), "{err}");
    }

    #[tokio::test]
    async fn a_repo_without_an_origin_is_told_to_add_one() {
        git_identity();
        let dir = TempDir::new("no-origin");
        app_repo(&dir.0, "https://github.com/acme/contract-review.git");
        run_git(&dir.0, &["remote", "remove", "origin"]);

        let err = repo_state(&dir.0).await.unwrap_err();
        assert!(err.to_string().contains("no `origin` remote"), "{err}");
        assert!(err.to_string().contains("git remote add origin"), "{err}");
    }

    #[tokio::test]
    async fn an_ssh_remote_stops_the_publish() {
        git_identity();
        let dir = TempDir::new("ssh");
        app_repo(&dir.0, "git@github.com:acme/contract-review.git");

        let err = repo_state(&dir.0).await.unwrap_err();
        assert!(err.to_string().contains("HTTPS GitHub remote"), "{err}");
    }

    /// The commit an app is published from, and a second one after an edit.
    const SHA: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA_TWO: &str = "fedcba9876543210fedcba9876543210fedcba98";

    /// PATH is process wide, so two shims at once would see each other's.
    #[cfg(unix)]
    static SHIM_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    /// A `gh` shim on PATH, with a real bare repo standing in for the index.
    ///
    /// The shim records every argv it is given and keeps the pull requests it
    /// opened, so a second publish through it meets the branch and the pull
    /// request the first one left behind, which is the case that used to fail.
    ///
    /// Unix only: the shim is a shell script, and the Windows runner reaches
    /// the same code through the pure functions above.
    #[cfg(unix)]
    struct GhShim {
        /// Held for as long as PATH carries the shim.
        _guard: tokio::sync::MutexGuard<'static, ()>,
        /// Kept alive so the bare index below is still on disk.
        root: TempDir,
        remote: PathBuf,
        log: PathBuf,
        previous_path: String,
    }

    #[cfg(unix)]
    impl GhShim {
        /// `permission` is what the shim answers for the viewer's permission
        /// on the index, which is what decides between the fork path and the
        /// push straight to the index.
        async fn new(permission: &str) -> Self {
            use std::os::unix::fs::PermissionsExt as _;

            let guard = SHIM_LOCK.lock().await;
            git_identity();
            let root = TempDir::new("gh");
            let bin = root.0.join("bin");
            std::fs::create_dir_all(&bin).unwrap();
            let log = root.0.join("gh.log");
            let prs = root.0.join("prs");
            std::fs::create_dir_all(&prs).unwrap();

            // The index as GitHub would serve it: a bare repo with one entry.
            let remote = root.0.join("aias-index.git");
            run_git(
                &root.0,
                &[
                    "init",
                    "--bare",
                    "--initial-branch=main",
                    remote.to_str().unwrap(),
                ],
            );
            let seed = root.0.join("seed");
            std::fs::create_dir_all(&seed).unwrap();
            run_git(&seed, &["init", "--initial-branch=main"]);
            std::fs::write(
                seed.join("apps.yaml"),
                "- name: note-taker\n  repo: https://github.com/acme/note-taker\n  ref: main\n  description: Summarize meetings\n",
            )
            .unwrap();
            run_git(&seed, &["add", "."]);
            run_git(&seed, &["commit", "--message", "seed"]);
            run_git(
                &seed,
                &["remote", "add", "origin", remote.to_str().unwrap()],
            );
            run_git(&seed, &["push", "origin", "main"]);

            // The shim answers the calls `open_pr` makes. A pull request is a
            // file named after its head branch, so `pr create` refuses a second
            // one on the same branch exactly as GitHub does and `pr list` finds
            // the one that is open.
            let shim = r#"#!/bin/sh
printf '%s\n' "$*" >> "$AIAS_TEST_GH_LOG"
head=${6#*:}
case "$1:$2" in
  auth:status) exit 0 ;;
  repo:view) echo "$AIAS_TEST_GH_PERMISSION" ; exit 0 ;;
  repo:fork) exit 0 ;;
  repo:clone) git clone --quiet "$AIAS_TEST_INDEX_REMOTE" "$4" ; exit $? ;;
  api:user) echo testuser ; exit 0 ;;
  pr:list)
    if [ -f "$AIAS_TEST_GH_PRS/$head" ] ; then cat "$AIAS_TEST_GH_PRS/$head" ; fi
    exit 0 ;;
  pr:create)
    if [ -f "$AIAS_TEST_GH_PRS/$head" ] ; then
      echo "a pull request for branch $head already exists" >&2
      exit 1
    fi
    number=$(( $(ls "$AIAS_TEST_GH_PRS" | wc -l) + 7 ))
    echo "https://github.com/zyx1121/aias-index/pull/$number" > "$AIAS_TEST_GH_PRS/$head"
    cat "$AIAS_TEST_GH_PRS/$head"
    exit 0 ;;
esac
exit 1
"#;
            let gh = bin.join("gh");
            std::fs::write(&gh, shim).unwrap();
            std::fs::set_permissions(&gh, std::fs::Permissions::from_mode(0o755)).unwrap();

            let previous_path = std::env::var("PATH").unwrap_or_default();
            // Safety: the shim directory only adds `gh`, so every other program
            // on PATH still resolves the same way, and the lock above keeps one
            // shim at a time.
            unsafe {
                std::env::set_var("PATH", format!("{}:{previous_path}", bin.display()));
                std::env::set_var("AIAS_TEST_GH_LOG", &log);
                std::env::set_var("AIAS_TEST_GH_PRS", &prs);
                std::env::set_var("AIAS_TEST_INDEX_REMOTE", &remote);
                std::env::set_var("AIAS_TEST_GH_PERMISSION", permission);
            }

            Self {
                _guard: guard,
                root,
                remote,
                log,
                previous_path,
            }
        }

        /// Publish the entry at `sha` and hand back the pull request URL.
        async fn publish(&self, sha: &str) -> String {
            assert!(gh_ready().await, "the shim answers `gh auth status`");
            open_pr(&entry(), sha).await.unwrap()
        }

        /// Every `gh` argv the shim was given, one call per line.
        fn calls(&self) -> String {
            std::fs::read_to_string(&self.log).unwrap_or_default()
        }

        /// The publish branches on the bare index, sorted.
        fn branches(&self) -> Vec<String> {
            let out = std::process::Command::new("git")
                .args([
                    "--git-dir",
                    self.remote.to_str().unwrap(),
                    "for-each-ref",
                    "--format=%(refname:short)",
                    "refs/heads/",
                ])
                .output()
                .expect("git runs");
            let mut branches: Vec<String> = String::from_utf8_lossy(&out.stdout)
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with(BRANCH_PREFIX))
                .map(str::to_string)
                .collect();
            branches.sort();
            branches
        }

        /// One file as it is on a branch of the bare index, if it is there.
        fn file_on(&self, branch: &str, path: &str) -> Option<String> {
            let out = std::process::Command::new("git")
                .args([
                    "--git-dir",
                    self.remote.to_str().unwrap(),
                    "show",
                    &format!("{branch}:{path}"),
                ])
                .output()
                .expect("git runs");
            out.status
                .success()
                .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
        }

        /// The entries a publish pushed, read off its branch.
        fn entries_on(&self, branch: &str) -> Vec<apps::IndexEntry> {
            let text = self
                .file_on(branch, INDEX_FILE)
                .unwrap_or_else(|| panic!("{branch} has no {INDEX_FILE}"));
            serde_yaml_ng::from_str(&text).unwrap()
        }

        /// Put a branch of somebody else's history on the index under `branch`.
        fn take_branch(&self, branch: &str) {
            let work = self.root.0.join("taken");
            std::fs::create_dir_all(&work).unwrap();
            run_git(&work, &["init", "--initial-branch=main"]);
            std::fs::write(work.join("taken.txt"), "another history\n").unwrap();
            run_git(&work, &["add", "."]);
            run_git(&work, &["commit", "--message", "taken"]);
            run_git(
                &work,
                &["remote", "add", "origin", self.remote.to_str().unwrap()],
            );
            run_git(&work, &["push", "origin", &format!("main:{branch}")]);
        }
    }

    #[cfg(unix)]
    impl Drop for GhShim {
        fn drop(&mut self) {
            // Safety: the lock is still held, so nothing else is reading PATH
            // through this shim while it goes away.
            unsafe { std::env::set_var("PATH", &self.previous_path) };
        }
    }

    #[test]
    fn a_publish_branch_names_the_app_and_the_commit() {
        assert_eq!(
            branch_for("contract-review", SHA),
            "add-contract-review-0123456"
        );
        assert_ne!(
            branch_for("contract-review", SHA),
            branch_for("contract-review", SHA_TWO)
        );
        // A repo state without a commit still produces a usable branch.
        assert_eq!(branch_for("contract-review", ""), "add-contract-review");
        assert!(check_git_ref(&branch_for("contract-review", SHA)).is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_gh_path_forks_pushes_and_opens_a_pull_request() {
        // A publisher who only reads the index gets the fork path. `--remote`
        // is not passed: `gh` refuses it next to a repository argument.
        let shim = GhShim::new("READ").await;
        let url = shim.publish(SHA).await;
        assert_eq!(url, "https://github.com/zyx1121/aias-index/pull/7");

        let calls = shim.calls();
        let branch = branch_for("contract-review", SHA);
        assert!(
            calls.contains("repo fork zyx1121/aias-index --clone=false"),
            "{calls}"
        );
        assert!(!calls.contains("--remote"), "{calls}");
        assert!(calls.contains("api user --jq .login"), "{calls}");
        assert!(calls.contains("repo clone testuser/aias-index"), "{calls}");
        assert!(
            calls.contains(&format!(
                "pr create --repo zyx1121/aias-index --head testuser:{branch}"
            )),
            "{calls}"
        );

        // The branch the publish pushed carries both entries, and it is still
        // there afterwards: the pull request is open on it.
        assert_eq!(shim.branches(), vec![branch.clone()]);
        let parsed = shim.entries_on(&branch);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1].name, "contract-review");
        assert_eq!(parsed[1].repo, "https://github.com/acme/contract-review");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn the_index_maintainer_pushes_to_the_index_itself() {
        // The maintainer of the index cannot fork it, so the branch goes to the
        // index itself and the pull request head is the bare branch name.
        let shim = GhShim::new("ADMIN").await;
        let url = shim.publish(SHA).await;
        assert_eq!(url, "https://github.com/zyx1121/aias-index/pull/7");

        let calls = shim.calls();
        let branch = branch_for("contract-review", SHA);
        assert!(!calls.contains("repo fork"), "{calls}");
        assert!(calls.contains("repo clone zyx1121/aias-index"), "{calls}");
        assert!(
            calls.contains(&format!(
                "pr create --repo zyx1121/aias-index --head {branch}"
            )),
            "{calls}"
        );
        assert_eq!(shim.entries_on(&branch).len(), 2);
    }

    /// The regression: publishing the same app twice used to be a non fast
    /// forward push onto the branch the first publish left on the fork.
    #[cfg(unix)]
    #[tokio::test]
    async fn publishing_the_same_app_twice_succeeds_both_times() {
        let shim = GhShim::new("READ").await;
        let first = shim.publish(SHA).await;

        // The same commit again: the branch and its open pull request are the
        // answer, and nothing is pushed over anybody's head.
        let again = shim.publish(SHA).await;
        assert_eq!(again, first, "the open pull request is what comes back");

        // A new commit is a branch of its own, and a pull request of its own.
        let third = shim.publish(SHA_TWO).await;
        assert_ne!(third, first, "a second commit opens a second pull request");

        let mut expected = vec![
            branch_for("contract-review", SHA),
            branch_for("contract-review", SHA_TWO),
        ];
        expected.sort();
        assert_eq!(shim.branches(), expected);
        for branch in &expected {
            let parsed = shim.entries_on(branch);
            assert_eq!(parsed.len(), 2, "{branch}");
            assert_eq!(parsed[1].name, "contract-review");
        }
        assert!(shim.calls().contains("pr list --repo zyx1121/aias-index"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_branch_another_history_holds_is_left_alone() {
        let shim = GhShim::new("READ").await;
        let taken = branch_for("contract-review", SHA);
        shim.take_branch(&taken);

        let url = shim.publish(SHA).await;
        assert_eq!(url, "https://github.com/zyx1121/aias-index/pull/7");

        // The entry went to a branch of its own, named after the one it could
        // not reuse, and the branch that was there still holds its own commit.
        let branches = shim.branches();
        assert_eq!(branches.len(), 2, "{branches:?}");
        let pushed = branches
            .iter()
            .find(|branch| *branch != &taken)
            .unwrap_or_else(|| panic!("nothing was pushed: {branches:?}"));
        assert!(pushed.starts_with(&format!("{taken}-")), "{pushed}");
        assert_eq!(
            shim.file_on(&taken, "taken.txt").as_deref(),
            Some("another history\n")
        );
        assert!(
            shim.file_on(&taken, INDEX_FILE).is_none(),
            "the branch was rewritten"
        );
        assert_eq!(shim.entries_on(pushed).len(), 2);
    }
}
