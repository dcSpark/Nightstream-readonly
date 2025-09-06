//! Property-based and QuickCheck tests for the Neo protocol.
//!
//! This crate provides comprehensive property verification through randomized testing,
//! complementing the deterministic unit tests with broader input space exploration.
//!
//! The tests are organized into three main categories:
//! - Bridge parity tests: Verify consistency between different protocol interfaces
//! - Decomposition properties: Test mathematical identities in field arithmetic
//! - Security guard rails: Ensure fail-closed behavior for security-critical checks
//!
//! All tests are designed to be fast and CI-friendly while maintaining good coverage.

#![warn(missing_docs)]
#![warn(clippy::all)]

// This is a test-only crate, so we don't export any public APIs.
// All functionality is in the test files.
