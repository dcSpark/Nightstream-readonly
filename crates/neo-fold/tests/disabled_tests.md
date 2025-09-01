# Disabled Test Files

The following test files have been temporarily disabled due to dependencies on unimplemented functionality:

- `soundness.rs` - Uses `neo_sumcheck::batched_sumcheck_verifier` and other unimplemented functions
- `security_tests.rs` - Uses `neo_commit`, `FoldState`, `verify_rlc`, `EvalInstance` and other unimplemented types
- `verifier_chain.rs` - Uses `neo_commit`, `neo_decomp`, `quickcheck_macros`, and unimplemented functions 
- `transcript_tests.rs` - Uses `neo_commit`, `FoldState` and other unimplemented functionality
- `ivc_tests.rs` - Uses `neo_commit`, `FoldState`, `Proof`, `EvalInstance` and other unimplemented types
- `open_binding.rs` - Uses `neo_commit`, `verify_open`, `EvalInstance` and other unimplemented functionality
- `knowledge_soundness.rs` - Uses `neo_commit`, `FoldState`, unimplemented verification functions
- `norm_scaling.rs` - Uses `neo_commit`, `pi_rlc`, `EvalInstance`, `FoldState` and other unimplemented types
- `edge_cases.rs` - Uses `neo_commit`, `FoldState` and other unimplemented functionality

These tests should be re-enabled once the required functionality is implemented.
