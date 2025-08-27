// neo-orchestrator/src/lib.rs
// Thin re-export façade - all SNARK logic now lives in neo-fold

// Re-export the main SNARK interface from neo-fold
pub use neo_fold::snark::{prove, verify, Metrics, OrchestratorError};

// Re-export neutronnova_integration for tests that might import it
pub use neo_fold::neutronnova_integration;

// Keep the spartan2 module for backward compatibility
pub mod spartan2;