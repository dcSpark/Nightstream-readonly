import SuperNeo.Primitives

/-!
Contract interface for `SuperNeo.Primitives`.

Paper anchors:
- `./formal/superneo-lean/SuperNeo.pdf.md`, Section 4 (Preliminaries), lines 268-353
- Definitions 1-6 (fields, rings, dimensions, coefficient maps, norms,
  commitment scheme, interactive reductions, sum-check)
- Lemma 5 (Schwartz-Zippel), Lemma 6 (eq-lifting)
-/

namespace SuperNeo

namespace PrimitivesInterface

/-- [Role: Theorem-Target] No curated module-level surface extracted yet. -/
def moduleContractPending : Prop := True

theorem moduleContractPending_true : moduleContractPending := by
  trivial

end PrimitivesInterface

end SuperNeo
