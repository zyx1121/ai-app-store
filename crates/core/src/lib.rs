//! Core logic of the AI App Store.
//!
//! Every capability lives in one module here. The Tauri app and the `aias` CLI
//! are thin shells over this crate, so anything that can be verified over SSH
//! without a GUI is verified through the CLI.
//!
//! The product plan is `PLAN.md` at the repo root; the module names follow the
//! object model in its section 2.

pub mod api;
pub mod apps;
pub mod error;
pub mod hardware;
pub mod instances;
pub mod jobs;
pub mod mcp;
pub mod models;
pub mod paths;
mod process;
pub mod publish;
pub mod runtime;
pub mod services;

pub use error::{Error, Result};
