# AGENTS.md

## Testing
- Never add tests in the same implementation file, always prefer to add them to a file inside tests/ (current or new)

## Build & Test Commands
- When running tests use --release eg cargo test --workspace --release

## Profiling

- You can easily profile with an easy to read output by just using this script: ./scripts/profile_for_ai.sh e.g., ./scripts/profile_for_ai.sh neo-fold test_sha256_single_step test_sha256_preimage_64_bytes --ignored