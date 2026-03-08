# SuperNeo Lean Cross-Check (Standalone)

This folder is intentionally outside `crates/` and independent from the Rust workspace.

It provides the theorem-facing Lean implementation of core SuperNeo/Neo math surfaces.
Lean is the mathematical source of truth.

Operationally:
- Lean theorem/definition surfaces are authoritative.
- Golden vectors from this formalization are used to validate Rust implementations.
- Rust-generated vectors (`neo-math`) are used as executable regression evidence.
- Target direction: run corresponding Lean and Rust computations in parallel and
  require value equality at each compared surface.

## What is checked

- SuperNeo inner-product transform identity:
  - `ct(cf_inv(bar(a)) * cf_inv(b)) == dot(a, b)`
- Ring multiplication in `R_q = F_q[X]/(X^54 + X^27 + 1)`
- Centered `l_inf` norm on field/ring elements
- Balanced `split_b` decomposition and recomposition
- `eq(x,y)` polynomial behavior (Boolean and non-Boolean inputs)
- MLE identity via inner-product form vs. folding form
- Definition 7 coefficient embedding round-trips (element/vector/matrix)
- Definition 8 bar-lift transform checks (vector and matrix forms)
- Theorem 4 computational identity checks: `Mz = ct(bar(M)z)`
- Remark 2 evaluation/`ct` linkage checks
- Theorem 5 linear-combination evaluation homomorphism checks
- Module-homomorphism sanity checks used by evaluation homomorphism
- Theorem 8 assumption boundary + concrete precondition checks
- Definition 17/Theorem 9 strong-sampling + expansion-factor checks
- Appendix C Lemma 6-style eq-lifting table checks (+ SZ bound interface sanity)
- Neo-style polynomial interpolation/evaluation helpers (base field)

## Layout

The module structure mirrors the paper's four main sections.

### Barrel files (top-level re-exports)

| Barrel | Paper section | Re-exports |
|---|---|---|
| `SuperNeo/Primitives.lean` | Section 4 (Preliminaries) | Goldilocks, Field, Dimensions, Ring, CoeffMaps, Norm, Decomp, EqPoly, MLE, SumCheck, PolyLemmas, Interp, Parameters |
| `SuperNeo/EmbeddingTheory.lean` | Section 5 (Embedding + Eval Hom) | Embedding, Thm3Core, BarLift, MatrixTransform, EvalLink, ModuleHom, EvalHom |
| `SuperNeo/SecurityModel.lean` | Section 6 + Appendix C security | InteractiveReductions, ProofSystem/{Types,Security,Negligible,Lattice,LatticeReductions}, InvertibilityAxioms, SamplingSet |
| `SuperNeo/FoldingProtocol.lean` | Section 7 (Folding scheme) | ProofSystem/{ConstraintSystem,SumCheck,Folding}, ProtocolRelations, PiCCS, PiRLC, PiDEC, ArithmeticBundle, ArithmeticObligations, ProtocolTarget, ProtocolMathTarget, ProtocolTheorem, ProofSystem/Protocol |

### Section 4: Primitives (Definitions 1-6, polynomial tools)

- `SuperNeo/Goldilocks.lean`: Goldilocks field constants
- `SuperNeo/Field.lean`: Goldilocks modular arithmetic implementation
- `SuperNeo/Dimensions.lean`: concrete `η`, `d`, and shape helpers
- `SuperNeo/Ring.lean`: ring multiplication/reduction, `ct`, bar-block mat-vec
- `SuperNeo/CoeffMaps.lean`: `cf` / `cf⁻¹` map definitions and round-trips
- `SuperNeo/Norm.lean`: centered-representative norms
- `SuperNeo/Decomp.lean`: balanced base-`b` decomposition (`split_b`) helpers
- `SuperNeo/EqPoly.lean`: `eq` polynomial helpers
- `SuperNeo/MLE.lean`: multilinear-extension identities (`r_hat`, folding)
- `SuperNeo/SumCheck.lean`: SuperNeo-specialized sum-check scaffold plus paper-facing verifier view; accepted for the SuperNeo protocol path, not as a fully generic standalone Definition-6 library formalization
- `SuperNeo/PolyLemmas.lean`: reusable polynomial helpers for Lemma 5/6
- `SuperNeo/Interp.lean`: polynomial eval + interpolation
- `SuperNeo/Parameters.lean`: Appendix B.2 concrete parameter constants

### Section 5: Embedding Theory (Definitions 7-8, Theorems 3-5)

- `SuperNeo/Embedding.lean`: Definition 7 element/vector/matrix embeddings
- `SuperNeo/Thm3Core.lean`: Theorem 3 core proposition + dimensional preconditions
- `SuperNeo/BarLift.lean`: Definition 8 blockwise lifting transform
- `SuperNeo/MatrixTransform.lean`: Theorem 4 computational transform identity
- `SuperNeo/EvalLink.lean`: Remark 2 coefficientwise evaluation linkage
- `SuperNeo/ModuleHom.lean`: Definition 15 module-hom interfaces + linearity
- `SuperNeo/EvalHom.lean`: Theorem 5 computational evaluation homomorphism

### Section 6: Security Model (Definitions 9-10, 16-18, Theorems 2, 6, 8-9)

- `SuperNeo/InteractiveReductions.lean`: Definitions 9-10 weak/strong reductions, Theorem 6 composition
- `SuperNeo/ProofSystem/Types.lean`: proof-system facade types (Context, Claim, Witness)
- `SuperNeo/ProofSystem/Security.lean`: probability/error model
- `SuperNeo/ProofSystem/Negligible.lean`: `ErrorFn`, `IsNegligible`
- `SuperNeo/ProofSystem/Lattice.lean`: Definition 16 (MSIS), Definition 18 (Ajtai), Theorem 2
- `SuperNeo/ProofSystem/LatticeReductions.lean`: MSIS-to-Ajtai security reductions
- `SuperNeo/InvertibilityAxioms.lean`: Theorem 8 assumption boundary and concrete checks
- `SuperNeo/SamplingSet.lean`: Definition 17/Theorem 9 sampling-set and expansion checks

### Section 7: Folding Protocol (Definitions 11-14, Lemmas 3-4, Theorem 7)

- `SuperNeo/ProofSystem/ConstraintSystem/CCS.lean`: Definitions 11-13 (CCS structure)
- `SuperNeo/ProofSystem/SumCheck/`: proof-system-level sum-check facade
- `SuperNeo/ProtocolRelations.lean`: Section 7.1 relation predicates
- `SuperNeo/PiCCS.lean`: Section 7.3 Π_CCS, Lemma 3
- `SuperNeo/PiRLC.lean`: Section 7.4 Π_RLC, Lemma 4
- `SuperNeo/PiDEC.lean`: Section 7.5 Π_DEC, Theorem 7
- `SuperNeo/ArithmeticBundle.lean`: bundled arithmetic prerequisites
- `SuperNeo/ArithmeticObligations.lean`: arithmetic side-conditions for protocol reduction
- `SuperNeo/ProtocolTarget.lean`: protocol-target bridge (Thm 3 + obligations)
- `SuperNeo/ProtocolMathTarget.lean`: protocol math-target from arithmetic bundle
- `SuperNeo/ProtocolTheorem.lean`: final theorem shape (completeness + knowledge-soundness)
- `SuperNeo/ProofSystem/Protocol.lean`: proof-system entrypoint (final theorem wiring)
- `SuperNeo/ProofSystem/Folding/`: proof-system folding wrappers

### Infrastructure

- `SuperNeo/ProtocolReduction.lean`: medium-term theorem skeletons
- `SuperNeo/Checks.lean`: cross-check predicates against generated vectors
- `SuperNeo/Regression.lean`: regression harness
- `SuperNeo/Generated/Vectors.lean`: Rust-generated constants (bar matrix + cases)
- `rust-vectors/`: standalone Rust generator crate

## Regenerate vectors from Rust

```bash
cargo run --manifest-path formal/superneo-lean/rust-vectors/Cargo.toml
```

## Run Lean checks

```bash
cd formal/superneo-lean
lake build
lake exe check
```

## Run SumCheck tests

The SumCheck test suite lives under `tests/` and is elaboration-driven:
`#guard` checks and `example` proofs fail at compile time.

```bash
cd formal/superneo-lean
lake build SumCheckTests
```

`lake build SumCheckTests` currently includes:
- standalone/core SumCheck smoke tests,
- proof-style SumCheck examples,
- prefix-soundness smoke tests for the executable proof-system path.

This complements `lake exe check`; it does not replace the global regression gate.

Expected output ends with:

```text
all_checks=true
```

## Check Output Breakdown (`lake exe check`)

`lake exe check` reports 21 atomic checks plus one aggregate gate.
`true` means the corresponding executable predicate passed on all configured inputs.
This is stronger than unit smoke tests, but weaker than full universal theorem proofs.

| Output flag | `true` means (exactly what passed) | Evidence type | Remaining gap to a full SuperNeo proof |
|---|---|---|---|
| `superneo_cases` | For every generated `(a,b)`: `ct(mulRqPhi(superneoBarBlock(bar,a), b))`, `dot(a,b)`, and expected values all agree. | Rust-generated vectors + Lean recomputation | Prove identity for all valid inputs, not only sampled/generated cases. |
| `ring_mul_cases` | `mulRqPhi a b` matches expected coefficient vectors for all generated multiplication cases. | Rust-generated vectors | Prove quotient-ring multiplication semantics universally. |
| `norm_cases` | `normInfCoeffs` equals expected norms on all generated norm cases. | Rust-generated vectors | Prove general norm properties/bounds used in later theorems. |
| `split_cases` | `splitBalancedVec` digits match expected, recomposition equals expected and original input, and per-digit bounds hold. | Rust-generated vectors + invariant check | Prove reconstruction and bound theorems for all inputs. |
| `eq_cases` | `eqPoly x y` matches expected; Boolean points also satisfy indicator behavior check. | Rust-generated vectors + Boolean invariant | Prove full hypercube-indicator theorem. |
| `mle_cases` | Inner-product MLE and folding MLE both match expected and match each other (`mleIdentity`). | Rust-generated vectors + identity check | Lift to quantified theorem over all vectors/points. |
| `embedding_vec_cases` | `embedVec` matches expected blocks and `unembedVec (embedVec v) = v`. | Rust-generated vectors + round-trip invariant | Prove embedding bijection/linearity generally. |
| `embedding_matrix_cases` | `embedMatrix` matches expected blocks and `unembedMatrix (embedMatrix M) = M`. | Rust-generated vectors + round-trip invariant | Prove matrix-level embedding theorems generally. |
| `bar_lift_vec_cases` | Bar-lift outputs for `v`, `w`, `v+w`, `s*v` match expected and satisfy add/scale linearity checks. | Rust-generated vectors + linearity invariant | Prove Definition 8 properties for all vectors/scalars. |
| `bar_lift_matrix_cases` | `barLiftMatrix` matches expected lifted matrices on all generated cases. | Rust-generated vectors | Prove matrix lift correctness and algebraic properties generally. |
| `matrix_transform_cases` | `matrixVecDirect M z`, `matrixVecCtBar bar M z`, and expected vectors all agree; identity predicate also holds. | Rust-generated vectors + identity check | Formalize Theorem 4 universally from lower lemmas. |
| `eval_link_cases` | Evaluation-link computations (`evalRingVec`, `ct`, expected outputs) agree and `evalLinkIdentity`/`evalLinkForMatrix` checks pass. | Rust-generated vectors + identity checks | Replace computational checks with quantified Remark 2 proof. |
| `eval_hom_cases` | Evaluation homomorphism outputs (`Y1`, `Y2`, linear combo, direct combo) all match expected and each other; `evalHom2` holds. | Rust-generated vectors + homomorphism invariant | Prove full Theorem 5 algebraically. |
| `module_hom_cases` | `moduleHomSanity` passes add/scale preservation for representative concrete homomorphisms. | Fixed sanity witnesses (not generated) | Prove abstract module-hom lemmas, not only witness instances. |
| `invertibility_cases` | Concrete Goldilocks arithmetic side-conditions for the low-norm invertibility interface are satisfied. | Deterministic constant checks | Replace the remaining external theorem boundary with a proved low-norm invertibility theorem (Theorem 8 core). |
| `sampling_cases` | Strong-sampling predicate, max norm, bound, empirical expansion, and `empirical <= bound` all match expected/hold. | Rust-generated vectors + bound check | Prove Theorem 9 bound universally over required set class. |
| `eq_lift_cases` | `eqLiftFromTable` matches expected sums; Boolean-point behavior matches expected values when applicable. | Rust-generated vectors + Boolean-point check | Prove Appendix C eq-lifting lemmas for all tables/points. |
| `poly_lemma_cases` | `polyLemmaSanity` passes (`eqLiftAllBoolean` on a sample table + SZ interface condition). | Fixed sanity witnesses | Prove Schwartz-Zippel and related lemmas in general form. |
| `coeff_map_cases` | Coefficient-map round-trip checks pass on superneo/ring-generated data; additional sanity predicates pass. | Mixed: generated data + sanity predicates | Complete formal inverse/linearity proofs for `cf`, `cf^-1`, `ct`. |
| `parameter_cases` | Shape sanity, concrete parameter sanity, and norm sanity predicates all hold. | Deterministic constant/invariant checks | Prove all downstream inequalities that depend on these constants. |
| `interp_cases` | Interpolation coefficients and evaluation at a test point match expected values for all generated interpolation cases. | Rust-generated vectors | Prove interpolation correctness/uniqueness generally. |
| `all_checks` | Logical conjunction of every check above is `true`. | Aggregate gate | No new math content; only reports that all current executable checks passed. |

## Paper-Faithful Proof-Complete Goal (Project Policy)

The target for this project is **paper-faithful proof-complete closure** of the protocol and its subparts.
This means:

1. Final protocol goal:
   - `S7.6` (Protocol Theorem) is proved from theorem-native dependencies (not only executable checks/skeleton wrappers).
2. Subpart goal:
   - each required milestone on the critical path is closed with quantified theorem surfaces.
3. Boundary policy:
   - `Done (Boundary)` is an intermediate milestone (interface/assumption boundary closed),
     **not** the final project state.
4. Check policy:
   - `lake exe check` must remain green, but checks are regression evidence, not substitutes for universal proofs.
5. Paper-faithfulness policy:
   - completion requires proving the same mathematical construction stated in the paper,
     not only an equivalent-by-definition interface surface.
   - if an executable construction exists (e.g. folding evaluator), the theorem-facing claim must include
     a proved bridge from that executable construction to the paper formula.
6. Trusted assumptions:
   - any remaining trusted assumption must be explicit, minimized, and documented with closure intent.

## Status Labels

| Label | Meaning | Final for project? |
|---|---|---|
| `Done (Boundary)` | Boundary/interface closure is complete and consumable; deeper theorem chain may still be open. | No |
| `Accepted (SuperNeo path)` | Proof-complete and paper-faithful for the concrete SuperNeo dependency chain; generic standalone generalization may remain open. | Yes, for current repo scope |
| `In progress` | Some theorem surfaces exist, but proof obligations remain open. | No |
| `Good shell` | Composition skeleton exists; full derivation is not complete. | No |
| `Done (Proof-Complete)` | Paper-faithful universal theorem closure achieved for that milestone with explicit assumptions only. | Yes |

## Proof Dependency Map (Section-Aligned)

Milestones are aligned to paper sections. Each ID `S<section>.<item>` maps to a specific
paper Definition, Theorem, or Lemma.

```text
    SECTION 4: PRIMITIVES                SECTION 5: EMBEDDING THEORY
    ========================             ============================

    S4.1 Field/Ring/Dims                 S5.1 Embedding (Def 7)
      |                                    |
      +---> S4.2 Norm/Decomp              v
      |       |                          S5.2 Thm 3 core <--- S4.6
      +---> S4.3 EqPoly/MLE               |
      |       |                            v
      |       +---> S4.5 PolyLemmas      S5.3 BarLift (Def 8)
      |                Interp              |
      |                                    v
      +---> S4.4 SumCheck               S5.4 MatrixTransform (Thm 4)
      |                                    |
      +---> S4.6 Parameters               +---> S5.5 EvalLink + ModuleHom
                                           |       |
                                           |       v
                                           +---> S5.6 EvalHom (Thm 5)


    SECTION 6: SECURITY MODEL           SECTION 7: FOLDING PROTOCOL
    ==========================           ============================

    S6.1 InteractiveReductions           S7.1 CCS Relations (Defs 11-14)
         (Defs 9-10, Thm 6)               |
                                           v
    S6.2 Lattice/MSIS/Ajtai             S7.2 Π_CCS  (Lemma 3) <--- S4.4
         (Defs 4, 16, 18, Thm 2)          |
      |                                    v
      +---> S6.3 Invertibility           S7.3 Π_RLC  (Lemma 4)
      |      (Thm 8) <--- S4.2            |
      |       |                            v
      +---> S6.4 Sampling               S7.4 Π_DEC  (Thm 7)
             (Def 17, Thm 9)              |
                                           v
    S6.5 Error/Negligible Model          S7.5 Arithmetic Obligations
                                              <--- S4.2, S4.5, S5.4, S5.6,
                                                   S6.3, S6.4
                                           |
                                           v
                                         S7.6 Protocol Theorem (Thm 1)
                                              <--- S5.2, S6.1, S6.2, S6.5,
                                                   S7.2, S7.3, S7.4, S7.5
```

## Milestone Table

| ID | Paper item | Lean modules | Core claim target | Depends on | Enables | Status |
|---|---|---|---|---|---|---|
| `S4.1` | Defs 1-2 (Field/Ring/Dims) | `Field.lean`, `Dimensions.lean`, `Ring.lean`, `CoeffMaps.lean` | Base field/ring algebra, coefficient maps, `ct` bridge. | - | S4.2, S4.3, S4.4, S5.1 | In progress (Field/Dimensions/Ring are proof-complete at module level; CoeffMaps linearity remains pending). |
| `S4.2` | Def 3 + decomposition | `Norm.lean`, `Decomp.lean` | Centered `l_∞` norm bounds, `split_b` recomposition. | S4.1 | S6.3, S6.4, S7.5 | In progress (Norm is proof-complete at module level; decomposition universal closure remains pending). |
| `S4.3` | `eq` polynomial + MLE | `EqPoly.lean`, `MLE.lean` | `eq` is Boolean-cube selector; MLE identity `ṽ(r) = ⟨v, r̂⟩`. | S4.1 | S4.5, S5.5 | EqPoly: Done (Proof-Complete). MLE: In progress (executable-folding bridge pending). |
| `S4.4` | Def 6 (sum-check) | `SumCheck.lean` | Sum-check soundness/completeness boundary. | S4.1 | S7.2 | Accepted (SuperNeo path): the prefix-dependent proof-system route and protocol integration are proof-complete for SuperNeo's actual use path; the standalone core remains a table/MLE-specialized scaffold rather than a fully generic `SumCheck(T; Q)` library formalization. |
| `S4.5` | Lemmas 5-6, interpolation | `PolyLemmas.lean`, `Interp.lean` | Schwartz-Zippel, eq-lifting, interpolation correctness. | S4.3 | S7.5 | In progress (sanity checks pass; quantified lemmas pending). |
| `S4.6` | App B.2 parameters | `Parameters.lean`, `Goldilocks.lean` | Concrete constants and bound checks. | - | S5.2, S6.3 | Done (Boundary). |
| `S5.1` | Def 7 (embedding) | `Embedding.lean` | Element/vector/matrix embedding bijection + linearity. | S4.1 | S5.2 | In progress (parity passing; proof layer pending). |
| `S5.2` | Thm 3 (inner-product transform) | `Thm3Core.lean` | `ct(cf⁻¹(bar(a)) · cf⁻¹(b)) = ⟨a, b⟩`. | S4.1, S4.6, S5.1 | S5.3, S7.6 | Done (Proof-Complete for the native paper instance via `thm3CoreAssumption_native`; generic closure is provided as finite basis criterion/checker `thm3BasisKernelCheck`). |
| `S5.3` | Def 8 (bar-lift) | `BarLift.lean` | Blockwise lifting is correct and linear. | S5.1, S5.2 | S5.4 | Done (Proof-Complete) for module-level theorem closure (`barLiftVector_add_constructive`, `barLiftVector_scale_constructive`, `barLiftLinearityAssumption_closed`). |
| `S5.4` | Thm 4 (matrix-vector transform) | `MatrixTransform.lean` | `Mz = ct(bar(M)z)` for all valid M, z. | S5.2 | S5.5, S5.6, S7.5 | Done (Boundary): theorem-native closure from P10 is proved; remaining open surface is upstream generic `thm3CoreAssumption` closure. |
| `S5.5` | Remark 2 + Def 15 | `EvalLink.lean`, `ModuleHom.lean` | Eval/`ct` linkage; module-hom linearity. | S4.1, S5.4 | S5.6 | In progress (check-level; quantified proofs pending). |
| `S5.6` | Thm 5 (eval homomorphism) | `EvalHom.lean` | Linear-combination preservation under evaluation. | S5.4, S5.5 | S7.5 | Done (Proof-Complete): theorem-native closure is proved constructively from MLE linearity, and all eval-hom boundary constructors are derived in-module. |
| `S6.1` | Defs 5, 9-10, Thm 6 | `InteractiveReductions.lean` | Weak/strong reductions compose correctly. | - | S7.6 | Done (Boundary): strong/weak composition theorems are proved from `InteractiveReductionAssumptions`; remaining work is instantiating the bundle on the final protocol path. |
| `S6.2` | Defs 4, 16, 18, Thm 2 | `ProofSystem/Lattice.lean`, `LatticeReductions.lean` | Ajtai commitment properties, MSIS hardness, binding reductions. | - | S6.3, S7.6 | Done (Boundary): Defs 4/16/18 and the Thm 2 reduction chain are proved; on the active `paperCarrier`/Goldilocks final-theorem path the Ajtai reduction package is derived directly from the MSIS boundary with theorem-native strong-sampling, while the generic `LatticeReductionLaws` surface remains only for abstract carrier generalization. |
| `S6.3` | Thm 8 (invertibility) | `InvertibilityAxioms.lean`, `InvertibilityGoldilocks.lean` | Low-norm invertibility preconditions and interface. | S4.2, S4.6, S6.2 | S6.4, S7.5 | Accepted (SuperNeo path): the active Goldilocks `paperCarrier`-difference theorem is now proved in-repo and consumed directly on the protocol path; the remaining generic gap is the standalone external low-norm invertibility theorem itself. |
| `S6.4` | Def 17 + Thm 9 (sampling) | `SamplingSet.lean` | Strong-sampling + expansion-factor interface. | S4.2, S6.3 | S7.5 | Done (Proof-Complete) for module-level contract surfaces (`samplingDiffSet`, `strongSamplingExpansionProp`, and associated theorem wrappers). |
| `S6.5` | Error/negligible model | `ProofSystem/{Types,Security,Negligible}.lean` | `ProbModel`, `ErrorModel`, `IsNegligible`. | - | S7.6 | Done (Proof-Complete): the canonical `ErrorModel` now derives `epsTotal` and its negligibility internally from the five component boundaries, and the final theorem consumes that model directly on the active protocol path. |
| `S7.1` | Defs 11-14 (CCS) | `ProofSystem/ConstraintSystem/CCS.lean`, `ProtocolRelations.lean` | Norm-bounded CCS structure and evaluation relations. | - | S7.2, S7.3 | In progress (definitions exist). |
| `S7.2` | Sec 7.3, Lemma 3 (Π_CCS) | `PiCCS.lean`, `ProofSystem/Folding/PiCCS.lean` | Π_CCS is a strong interactive reduction. | S4.4, S7.1 | S7.4 | Done (Boundary): theorem is proved from protocol-target assumptions plus an accepted SumCheck witness; no separate SumCheck boundary bundle remains on this path. |
| `S7.3` | Sec 7.4, Lemma 4 (Π_RLC) | `PiRLC.lean`, `ProofSystem/Folding/PiRLC.lean` | Π_RLC is a weak interactive reduction. | S7.2 | S7.4 | Done (Boundary): theorem is proved directly from `ProtocolTargetAssumptions` plus an accepted transition witness. |
| `S7.4` | Sec 7.5, Thm 7 (Π_DEC) | `PiDEC.lean`, `ProofSystem/Folding/PiDEC.lean` | Π_DEC is a reduction of knowledge. | S7.3 | S7.6 | Done (Boundary): theorem is proved directly from `ProtocolTargetAssumptions` plus an accepted transition witness, using invertibility already packaged in `protocolTargetProp`. |
| `S7.5` | Arithmetic obligations | `ArithmeticBundle.lean`, `ArithmeticObligations.lean`, `ProtocolTarget.lean`, `ProtocolMathTarget.lean` | Side-conditions compose cleanly for protocol reduction. | S4.2, S4.5, S5.4, S5.6, S6.3, S6.4 | S7.6 | Done (Boundary): theorem-native arithmetic bundles and protocol-target constructors are proved; remaining work is upstream theorem-provider closure, not a local shell. |
| `S7.6` | Thm 1 (protocol theorem) | `ProtocolTheorem.lean`, `ProofSystem/Protocol.lean` | End-to-end completeness + knowledge-soundness. | S5.2, S6.1, S6.2, S6.5, S7.2, S7.3, S7.4, S7.5 | Final claim | Done (Boundary): theorem shape and canonical final-assumption assembly are proved; on the active `paperCarrier` path the final package now derives Ajtai reduction data directly from the MSIS boundary, the narrowed Goldilocks Appendix B.2 route fixes the concrete paper lattice constants while leaving only message length explicit, and the active `paperCarrier`-difference route now consumes the proved Goldilocks invertibility theorem directly rather than an external invertibility boundary. Remaining work is instantiating the upstream lattice/error bundles on the final path. |

### Tracked Status and Exit Criteria

Completion policy reminder: every row below targets `Done (Proof-Complete)` as the terminal state.
Rows marked `Done (Boundary)` are intentionally intermediate.

| ID | Status now | Missing now | Exit criteria |
|---|---|---|---|
| `S4.1` | In progress (Field/Dimensions/Ring are Done (Proof-Complete) at module contract level; CoeffMaps remains open). | CoeffMaps: linearity lemmas and full `cf`/`cf⁻¹` theorem closure. | Complete theorem API for `cf`/`cf⁻¹`/`ct` + ring semantics consumed by S5.2/S5.5. |
| `S4.2` | In progress (Norm is Done (Proof-Complete) at module contract level; Decomp remains open). | Universal decomposition theorem with terminal-state closure and bound threading. | Norm/decomp obligations in S6.3/S6.4/S7.5 discharged from theorem lemmas. |
| `S4.3` | EqPoly: Done (Proof-Complete). MLE: In progress. | MLE: prove `mleByFoldingExec = mleByInnerProduct` universally. | Both modules proof-complete. |
| `S4.4` | Accepted (SuperNeo path). | Optional generalization only: replace the standalone table/MLE scaffold with a generic Definition-6 `SumCheck(T; Q)` semantic object if a reusable library formalization is desired. | Sum-check remains theorem-native and paper-faithful for the SuperNeo dependency chain consumed by S7.2. |
| `S4.5` | In progress. | Quantified SZ, eq-lift, interpolation correctness/uniqueness. | Full polynomial lemma set consumed by S7.5. |
| `S4.6` | Done (Boundary). | None for boundary closure. | Parameter inequalities used by S5.2/S6.3 come from theorem constants. |
| `S5.1` | In progress. | Embedding bijection + linearity not theorem-native. | General embedding/unembedding theorem suite. |
| `S5.2` | Done (Proof-Complete for native paper instance). | Optional extension only: prove basis criterion for additional concrete bar constructions. | Native Theorem-3 remains constructive while preserving downstream theorem interfaces. |
| `S5.3` | Done (Proof-Complete) for module-level closure. | None for the module-level theorem closure; optional extension is additional non-native bar design validation. | Keep bar-lift linearity theorem-native and reused directly by S5.4/S5.5 constructors. |
| `S5.4` | Done (Boundary). | Upstream generic `thm3CoreAssumption` closure remains open. | Full Theorem-4 proof remains discharged for native/bar-closed P10 paths; generic closure finalizes when S5.2 generic boundary is discharged. |
| `S5.5` | In progress. | Eval-link and module-hom remain check-oriented. | Quantified proofs for Remark 2 and module-hom linearity. |
| `S5.6` | Done (Proof-Complete). | None at the module level. | Theorem-5 remains proved constructively and feeds S7.5 without additional local boundaries. |
| `S6.1` | Done (Boundary). | Instantiate `InteractiveReductionAssumptions` on the final protocol path and collapse the remaining boundary bundle into S7.6. | Theorem-6 composition remains proved from the reduction bundle and is consumed directly by S7.6. |
| `S6.2` | Done (Boundary). | Optional extension only: discharge the remaining generic `LatticeReductionLaws` abstraction if a fully carrier-parametric reduction library is desired beyond the active `paperCarrier`/Goldilocks route. | MSIS-to-Ajtai binding theorems consumed by S7.6 from the accepted base MSIS boundary on the concrete protocol path. |
| `S6.3` | Accepted (SuperNeo path). | The active Goldilocks `paperCarrier`-difference route is now closed in-repo via `paperCarrierDiffInvertibilityAssumption_goldilocks`; the remaining generic gap is the standalone external low-norm invertibility theorem itself. | Decide whether to keep the generic theorem as an explicit trusted boundary or formalize it beyond the active path. |
| `S6.4` | Done (Proof-Complete) for module-level theorem surfaces. | Downstream protocol threading (S7.5) still needs full theorem-only closure. | Universal sampling expansion theorem wired into S7.5. |
| `S6.5` | Done (Proof-Complete). | None at the module level. | Error model derives total-error decomposition/negligibility internally and is consumed directly by S7.6. |
| `S7.1` | In progress. | CCS/relation definitions exist but not theorem-complete. | CCS relation predicates consumed by S7.2/S7.3. |
| `S7.2` | Done (Boundary). | Upstream closure still sits in `ProtocolTargetAssumptions`; no separate SumCheck boundary remains on this path. | Π_CCS strong reduction proved from protocol-target assumptions plus an accepted transition witness. |
| `S7.3` | Done (Boundary). | Instantiate `ProtocolTargetAssumptions` on the intended protocol path; no local theorem gap remains. | Π_RLC weak reduction remains proved from the narrowed protocol-target boundary. |
| `S7.4` | Done (Boundary). | Instantiate `ProtocolTargetAssumptions` on the intended protocol path; no local theorem gap remains. | Π_DEC reduction-of-knowledge remains proved from the narrowed protocol-target boundary. |
| `S7.5` | Done (Boundary). | Discharge upstream theorem providers (`S4.5`, `S6.3`) that feed the arithmetic/protocol-target bundle. | Arithmetic obligations and protocol-target composition remain theorem-native once those upstream providers are instantiated. |
| `S7.6` | Done (Boundary). | Instantiate the remaining upstream boundary bundle `S6.5` on the final protocol path; the narrowed Goldilocks Appendix B.2 route is now available as the canonical concrete entry point, and its active `paperCarrier`-difference branch uses the proved Goldilocks invertibility theorem directly. | End-to-end protocol theorem consumed from explicit assumptions only, with final boundary packages assembled canonically in-module. |

## Math Breakdown (Current Status)

Source references:
- `docs/superneo-paper/04_4_Preliminaries.md`
- `docs/superneo-paper/05_5_Embedding_products_with_evaluation_homomorphism.md`
- `docs/superneo-paper/11_B_Concrete_parameters.md`
- `docs/superneo-paper/12_C_Additional_Background.md`
- `docs/superneo-paper/13_D_Deferred_theorems_and_proofs.md`

### Section 4: Preliminaries

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M1 | Definition 1 (field/ring/dimension) | `Field.lean` + `Dimensions.lean` | S4.1 | Done (Boundary) |
| M2 | Definition 2 (`cf`, `cf⁻¹`, `ct`) | `CoeffMaps.lean` + `Ring.lean` | S4.1 | In progress |
| M3 | Ring arithmetic in `R_q` | `Ring.lean` | S4.1 | Done (Proof-Complete) |
| M4 | Definition 3 (centered `l_∞` norm) | `Norm.lean` | S4.2 | Done (Proof-Complete) |
| M5 | `split_b` decomposition | `Decomp.lean` | S4.2 | In progress |
| M6 | `eq` polynomial on Boolean hypercube | `EqPoly.lean` | S4.3 | Done (Proof-Complete) |
| M7 | MLE identity | `MLE.lean` | S4.3 | In progress |
| M8 | Definition 6 (sum-check protocol) | `SumCheck.lean` | S4.4 | Accepted (SuperNeo path) |
| M9 | Lemma 5 (Schwartz-Zippel) | `PolyLemmas.lean` | S4.5 | In progress |
| M10 | Lemma 6 (eq-lifting) | `PolyLemmas.lean` | S4.5 | In progress |
| M11 | Polynomial interpolation/evaluation | `Interp.lean` | S4.5 | In progress |
| M12 | Appendix B.2 concrete parameters | `Parameters.lean` + `Goldilocks.lean` | S4.6 | Done (Boundary) |

### Section 5: Embedding Products with Evaluation Homomorphism

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M13 | Definition 7 (coefficient embedding) | `Embedding.lean` | S5.1 | In progress |
| M14 | Theorem 3 (inner-product transform) | `Thm3Core.lean` | S5.2 | Done (Proof-Complete for native paper instance) |
| M15 | Definition 8 (lifting transform) | `BarLift.lean` | S5.3 | Done (Proof-Complete) |
| M16 | Theorem 4 (matrix-vector product transform) | `MatrixTransform.lean` | S5.4 | Done (Boundary) |
| M17 | Remark 2 (evaluation/ct linkage) | `EvalLink.lean` | S5.5 | In progress |
| M18 | Definition 15 (module homomorphisms) | `ModuleHom.lean` | S5.5 | In progress |
| M19 | Theorem 5 (evaluation homomorphism) | `EvalHom.lean` | S5.6 | Done (Proof-Complete) |

### Section 6: Security Model

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M20 | Definition 5 (interactive reductions) | `InteractiveReductions.lean` | S6.1 | Done (Boundary) |
| M21 | Definitions 9-10 (weak/strong reductions) | `InteractiveReductions.lean` | S6.1 | Done (Boundary) |
| M22 | Theorem 6 (strong-weak composition) | `InteractiveReductions.lean` | S6.1 | Done (Boundary) |
| M23 | Definition 4 (ring commitment scheme) | `ProofSystem/Lattice.lean` | S6.2 | Done (Proof-Complete) |
| M24 | Definition 16 (MSIS) | `ProofSystem/Lattice.lean` | S6.2 | Done (Proof-Complete) |
| M25 | Definition 18 (Ajtai commitment) | `ProofSystem/Lattice.lean` | S6.2 | Done (Proof-Complete) |
| M26 | Theorem 2 (Ajtai properties) | `ProofSystem/LatticeReductions.lean` | S6.2 | In progress |
| M27 | Theorem 8 (low-norm invertibility) | `InvertibilityAxioms.lean`, `InvertibilityGoldilocks.lean` | S6.3 | Accepted (SuperNeo path) |
| M28 | Definition 17 (strong sampling sets) | `SamplingSet.lean` | S6.4 | Done (Proof-Complete) |
| M29 | Theorem 9 (expansion factors) | `SamplingSet.lean` | S6.4 | Done (Proof-Complete) |

### Section 7: Folding Protocol

| ID | Math item (paper) | Lean target | Milestone | Status |
|---|---|---|---|---|
| M30 | Definition 11 (structure) | `ProofSystem/ConstraintSystem/CCS.lean` | S7.1 | In progress |
| M31 | Definition 12 (norm-bounded CCS) | `ProofSystem/ConstraintSystem/CCS.lean` | S7.1 | In progress |
| M32 | Definition 13 (CCS evaluation relation) | `ProtocolRelations.lean` | S7.1 | In progress |
| M33 | Definition 14 (global parameters) | `ProtocolRelations.lean` | S7.1 | In progress |
| M34 | Lemma 3 (Π_CCS is strong) | `PiCCS.lean` | S7.2 | Done (Boundary) |
| M35 | Lemma 4 (Π_RLC is weak) | `PiRLC.lean` | S7.3 | Done (Boundary) |
| M36 | Theorem 7 (Π_DEC reduction of knowledge) | `PiDEC.lean` | S7.4 | Done (Boundary) |
| M37 | Arithmetic obligations | `ArithmeticBundle.lean`, `ArithmeticObligations.lean` | S7.5 | Done (Boundary) |
| M38 | Theorem 1 (full composition) | `ProtocolTheorem.lean`, `ProofSystem/Protocol.lean` | S7.6 | Done (Boundary) |

### Infrastructure

| ID | Item | Lean target | Status |
|---|---|---|---|
| M39 | Executable cross-check harness | `Main.lean` + `Checks.lean` | Done (all checks pass) |

### Status Summary

| State | Count |
|---|---|
| Accepted (SuperNeo path) | 2 (M8, M27) |
| Done (Boundary) | 12 (M1, M12, M16, M20, M21, M22, M34, M35, M36, M37, M38, M39) |
| Done (Proof-Complete) | 11 (M3, M4, M6, M14, M15, M19, M23, M24, M25, M28, M29) |
| In progress | 14 |
| Good shell | 0 |
| Not started | 0 |
