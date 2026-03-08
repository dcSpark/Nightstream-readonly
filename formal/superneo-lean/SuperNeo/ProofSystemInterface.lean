import SuperNeo.ProofSystem

/-!
Contract interface for `SuperNeo.ProofSystem`.

Spec: `specs/ProofSystem.spec.md`

Paper anchors:
- Section 6 (Security Model), lines 404-447.
- Section 7 (Folding Protocol), lines 449-596.
-/

namespace SuperNeo

namespace ProofSystemInterface

/-- [Role: Theorem-Target] No curated module-level surface extracted yet. -/
def moduleContractPending : Prop := True

theorem moduleContractPending_true : moduleContractPending := by
  trivial

end ProofSystemInterface

end SuperNeo
