//! One error type for the whole core crate.

/// Anything the core can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A scaffolded entry point with no implementation behind it yet.
    #[error("not implemented: {0}")]
    NotImplemented(&'static str),

    /// The manifest parsed but broke one of the rules in PLAN.md section 3.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),

    /// Something the caller asked for does not exist on disk or upstream.
    #[error("not found: {0}")]
    NotFound(String),

    /// The device profile has no backend in the PLAN.md section 2.1 table.
    #[error("unsupported hardware: {0}")]
    UnsupportedHardware(String),

    /// A request reached the local API without the bearer token, or with the
    /// wrong one. Its own variant because it is the one failure the API answers
    /// before a handler runs.
    #[error("unauthorized: {0}")]
    Unauthorized(String),

    /// A child process the platform owns failed.
    #[error("process failed: {0}")]
    Process(String),

    /// A request over the network failed, or answered with a status the caller
    /// cannot use. Kept apart from [`Error::Io`] so the frontend can tell a
    /// missing file from an unreachable host.
    #[error("http: {0}")]
    Http(String),

    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result alias used by every public function in this crate.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Stable machine readable tag, used by the CLI and by the Tauri layer.
    pub fn code(&self) -> &'static str {
        match self {
            Error::NotImplemented(_) => "not_implemented",
            Error::InvalidManifest(_) => "invalid_manifest",
            Error::NotFound(_) => "not_found",
            Error::UnsupportedHardware(_) => "unsupported_hardware",
            Error::Unauthorized(_) => "unauthorized",
            Error::Process(_) => "process",
            Error::Http(_) => "http",
            Error::Io(_) => "io",
            Error::Yaml(_) => "yaml",
            Error::Json(_) => "json",
        }
    }

    /// True when the failure is only a missing implementation.
    pub fn is_not_implemented(&self) -> bool {
        matches!(self, Error::NotImplemented(_))
    }
}
