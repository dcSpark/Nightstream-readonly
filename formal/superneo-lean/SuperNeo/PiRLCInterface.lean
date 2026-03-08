import SuperNeo.PiRLC

/-!
Contract interface for `SuperNeo.PiRLC`.

Spec: ./formal/superneo-lean/specs/PiRLC.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.4 (Π_RLC), lines 549-583.
- Lemma 4 (Π_RLC is weak), lines 582-583.
-/

namespace SuperNeo

namespace PiRLCInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piRLCWeakStatement`. -/
abbrev piRLCWeakStatement := SuperNeo.piRLCWeakStatement

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `PiRLCAssumptions` requiring closure. -/
abbrev PiRLCAssumptions := SuperNeo.PiRLCAssumptions

/-- [Role: Theorem-Target] Curated theorem surface `piRLCWeak_of_assumptions`. -/
abbrev piRLCWeak_of_assumptions := SuperNeo.piRLCWeak_of_assumptions

end PiRLCInterface

end SuperNeo
