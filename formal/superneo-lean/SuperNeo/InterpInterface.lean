import SuperNeo.Interp

/-!
Contract interface for `SuperNeo.Interp`.

Spec: `specs/Interp.spec.md`

Paper anchors:
- Infrastructure module supporting evaluation checks in Sections 7.3-7.4.
-/

namespace SuperNeo

namespace InterpInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `interpolationProp`. -/
abbrev interpolationProp := SuperNeo.interpolationProp

/-- [Role: Theorem-Target] Curated re-export of `interpolationCase`. -/
abbrev interpolationCase := SuperNeo.interpolationCase

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `interpolationCase_sound`. -/
abbrev interpolationCase_sound := SuperNeo.interpolationCase_sound

/-- [Role: Theorem-Target] Curated theorem surface `interpolationCase_complete`. -/
abbrev interpolationCase_complete := SuperNeo.interpolationCase_complete

/-- [Role: Theorem-Target] Curated theorem surface `interpolationCase_eq_true_iff`. -/
abbrev interpolationCase_eq_true_iff := SuperNeo.interpolationCase_eq_true_iff

/-- [Role: Theorem-Target] Curated theorem surface `interpolationProp_sizes`. -/
abbrev interpolationProp_sizes := SuperNeo.interpolationProp_sizes

/-- [Role: Theorem-Target] Curated theorem surface `interpolationProp_eval_eq`. -/
abbrev interpolationProp_eval_eq := SuperNeo.interpolationProp_eval_eq

/-- [Role: Theorem-Target] Curated theorem surface `not_interpolationAssumption`. -/
abbrev not_interpolationAssumption := SuperNeo.not_interpolationAssumption

/-! ## Boundary Surfaces -/

/-- [Role: Legacy Boundary / Refuted] Legacy surface `interpolationAssumption`; do not use it as a real closure target. -/
abbrev interpolationAssumption := SuperNeo.interpolationAssumption

end InterpInterface

end SuperNeo
