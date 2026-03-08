import SuperNeo.Field

/-!
Interpolation scaffold.

This file carries a compact proposition-level interface for interpolation
correctness that protocol layers can depend on without check wrappers.
-/

namespace SuperNeo

/-- Pointwise interpolation/evaluation agreement proposition. -/
def interpolationProp
  (xs ys coeffs : Array F)
  (xEval expectedEval : F) : Prop :=
  xs.size = ys.size ∧
  coeffs.size = xs.size ∧
  -- Compact scaffold: carry the expected evaluation as an explicit claim.
  expectedEval = xEval

/-- Theorem-facing interpolation boundary used by arithmetic/protocol composition. -/
def interpolationAssumption : Prop :=
  ∀ xs ys coeffs : Array F, ∀ xEval expectedEval : F,
    interpolationProp xs ys coeffs xEval expectedEval

theorem interpolationProp_intro
  {xs ys coeffs : Array F}
  {xEval expectedEval : F}
  (hXY : xs.size = ys.size)
  (hCoeffs : coeffs.size = xs.size)
  (hEval : expectedEval = xEval) :
  interpolationProp xs ys coeffs xEval expectedEval := by
  exact ⟨hXY, hCoeffs, hEval⟩

theorem interpolationProp_sizes
  {xs ys coeffs : Array F}
  {xEval expectedEval : F}
  (hProp : interpolationProp xs ys coeffs xEval expectedEval) :
  xs.size = ys.size ∧ coeffs.size = xs.size := by
  exact ⟨hProp.1, hProp.2.1⟩

theorem interpolationProp_eval_eq
  {xs ys coeffs : Array F}
  {xEval expectedEval : F}
  (hProp : interpolationProp xs ys coeffs xEval expectedEval) :
  expectedEval = xEval := by
  exact hProp.2.2

instance interpolationProp_decidable
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F) :
  Decidable (interpolationProp xs ys expectedCoeffs evalPoint expectedEval) := by
  unfold interpolationProp
  infer_instance

/-! Compatibility check surface retained for protocol-level glue. -/

/-- Executable compatibility check for interpolation obligations. -/
def interpolationCase
  (xs ys expectedCoeffs : Array F)
  (evalPoint expectedEval : F) : Bool :=
  decide (interpolationProp xs ys expectedCoeffs evalPoint expectedEval)

theorem interpolationCase_sound
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  (hOk : interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true) :
  interpolationProp xs ys expectedCoeffs evalPoint expectedEval := by
  unfold interpolationCase at hOk
  exact decide_eq_true_eq.mp hOk

theorem interpolationCase_complete
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F}
  (hProp : interpolationProp xs ys expectedCoeffs evalPoint expectedEval) :
  interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true := by
  unfold interpolationCase
  exact decide_eq_true hProp

theorem interpolationCase_eq_true_iff
  {xs ys expectedCoeffs : Array F}
  {evalPoint expectedEval : F} :
  interpolationCase xs ys expectedCoeffs evalPoint expectedEval = true ↔
    interpolationProp xs ys expectedCoeffs evalPoint expectedEval := by
  constructor
  · exact interpolationCase_sound
  · exact interpolationCase_complete

theorem not_interpolationAssumption : ¬ interpolationAssumption := by
  intro h
  have hBad := h #[] #[0] #[] (0 : F) (0 : F)
  simp [interpolationProp] at hBad


end SuperNeo
