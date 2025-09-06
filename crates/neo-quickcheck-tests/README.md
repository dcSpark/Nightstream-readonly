# neo-quickcheck-tests

Property-based and QuickCheck tests for the Neo protocol, providing comprehensive verification of mathematical identities and security properties through randomized testing.

## Overview

This crate contains property-based tests that verify critical protocol invariants using both [Proptest](https://github.com/proptest-rs/proptest) and [QuickCheck](https://github.com/BurntSushi/quickcheck) libraries. These tests complement the deterministic unit tests by exploring a much larger input space through randomized generation.

## Test Categories

### Bridge Parity Tests (`bridge_parity.rs`)
- **Header encoding consistency**: Verifies that `encode_bridge_io_header(me)` produces byte-identical output to the circuit's `public_values()` 
- **Padding verification**: Tests proper handling of variable-length inputs and padding-before-digest
- **Digest placement**: Ensures digest limbs are correctly positioned in the serialized output

### Decomposition Properties (`dec_props.rs`)
- **Base-b recomposition over F**: Verifies `recombine(base, limbs) == parent` for field elements
- **Base-b recomposition over K**: Tests the same property over the extension field
- **Range polynomial (b=2)**: Validates that `v(v-1)(v+1) = 0` iff `v ∈ {-1,0,1}` over Goldilocks
- **Negative cases**: Confirms that corrupted inputs properly fail verification

### Security Guard Rails (`security_red_team.rs`)
- **Ajtai binding enforcement**: Tests that the bridge properly rejects instances with missing or empty Ajtai commitment rows
- **Fail-closed behavior**: Ensures security-critical checks fail safely when preconditions are violated

## Features

- **Default**: Basic functionality
- **quickcheck**: Enable QuickCheck-specific tests (similar to `neo-redteam-tests`)

## Usage

**By design, these tests only run when explicitly requested:**

```bash
# Skip all quickcheck tests (default - compiles 0 tests)
cargo test -p neo-quickcheck-tests

# Run all property-based tests (requires quickcheck flag)
cargo test -p neo-quickcheck-tests --features quickcheck
```

Run specific test categories:
```bash
# Bridge parity tests only
cargo test -p neo-quickcheck-tests --features quickcheck bridge_parity

# Decomposition properties only  
cargo test -p neo-quickcheck-tests --features quickcheck dec_props

# Security guard rails only
cargo test -p neo-quickcheck-tests --features quickcheck security_red_team
```

**Why this flag?**
Neo's property-based tests are intended to be invoked explicitly in a "security/property verification" mode rather than during routine development cycles.

## Test Strategy

- **Small input spaces**: Tests use bounded generators to remain CI-friendly while still covering edge cases
- **Focused properties**: Each test verifies a specific mathematical identity or security property
- **Negative testing**: Where applicable, tests verify that corrupted inputs properly fail validation
- **Public API only**: Tests interact only with public APIs, avoiding dependency on internal implementation details

## Design Principles

1. **Fast execution**: Test parameters are tuned for quick CI runs while maintaining coverage
2. **Property verification**: Focus on mathematical identities rather than specific implementation details  
3. **Complementary coverage**: Works alongside deterministic unit tests to provide broader verification
4. **Security-first**: Explicit testing of security guard rails and fail-closed behaviors

## Dependencies

This crate depends on all major Neo protocol crates to provide comprehensive cross-module property testing:

- `neo`: Main protocol facade
- `neo-ccs`: Constraint system representations
- `neo-fold`: Folding scheme implementation  
- `neo-spartan-bridge`: Circuit compilation bridge
- `neo-math`: Field arithmetic and extensions
- `neo-ajtai`: Lattice-based commitments

The property-based approach ensures that interfaces between these modules maintain their mathematical guarantees across refactoring and optimization.
