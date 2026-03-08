import SuperNeo.PolyLemmas

/-!
Contract interface for `SuperNeo.PolyLemmas`.

Spec: `specs/PolyLemmas.spec.md`

Paper anchors:
- Lemma 5 (Schwartz-Zippel), Appendix C, lines 733-736.
- Lemma 6 (eq-lifting), Appendix C, lines 737-740.
-/

namespace SuperNeo

namespace PolyLemmasInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `eqLiftFromTable`. -/
abbrev eqLiftFromTable := SuperNeo.eqLiftFromTable

/-- [Role: Theorem-Target] Curated re-export of `eqLiftBooleanIndicator`. -/
abbrev eqLiftBooleanIndicator := SuperNeo.eqLiftBooleanIndicator

/-- [Role: Theorem-Target] Curated re-export of `eqLiftBooleanIndicatorProp`. -/
abbrev eqLiftBooleanIndicatorProp := SuperNeo.eqLiftBooleanIndicatorProp

/-- [Role: Theorem-Target] Curated re-export of `eqLiftAllBoolean`. -/
abbrev eqLiftAllBoolean := SuperNeo.eqLiftAllBoolean

/-- [Role: Theorem-Target] Curated re-export of `schwartzZippelBoundLeOne`. -/
abbrev schwartzZippelBoundLeOne := SuperNeo.schwartzZippelBoundLeOne

/-- [Role: Theorem-Target] Curated re-export of `polyLemmaSanity`. -/
abbrev polyLemmaSanity := SuperNeo.polyLemmaSanity

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `schwartzZippelBoundLeOne_sound`. -/
abbrev schwartzZippelBoundLeOne_sound := SuperNeo.schwartzZippelBoundLeOne_sound

/-- [Role: Theorem-Target] Curated theorem surface `schwartzZippelBoundLeOne_complete`. -/
abbrev schwartzZippelBoundLeOne_complete := SuperNeo.schwartzZippelBoundLeOne_complete

/-- [Role: Theorem-Target] Curated theorem surface `eqLiftBooleanIndicator_sound`. -/
abbrev eqLiftBooleanIndicator_sound := SuperNeo.eqLiftBooleanIndicator_sound

/-- [Role: Theorem-Target] Curated theorem surface `eqLiftBooleanIndicator_complete`. -/
abbrev eqLiftBooleanIndicator_complete := SuperNeo.eqLiftBooleanIndicator_complete

/-- [Role: Theorem-Target] Curated theorem surface `eqLiftBooleanIndicator_eq_true_iff`. -/
abbrev eqLiftBooleanIndicator_eq_true_iff := SuperNeo.eqLiftBooleanIndicator_eq_true_iff

/-- [Role: Theorem-Target] Curated theorem surface `schwartzZippelBoundLeOne_eq_true_iff_prop`. -/
abbrev schwartzZippelBoundLeOne_eq_true_iff_prop :=
  SuperNeo.schwartzZippelBoundLeOne_eq_true_iff_prop

end PolyLemmasInterface

end SuperNeo
