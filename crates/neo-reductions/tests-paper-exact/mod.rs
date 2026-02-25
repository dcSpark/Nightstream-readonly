// Legacy paper-exact micro-fixtures (many use pre-SuperNeo tiny widths) are
// opt-in under `testing`. Default release runs keep only SuperNeo-compatible suites.
#![cfg(all(feature = "paper-exact", feature = "testing"))]

// Paper-exact test modules for the CCS-based folding protocol
mod oracle_self_check;
mod paper_ccs_dec_tests;
mod paper_ccs_rlc_tests;
mod paper_ccs_tests;
