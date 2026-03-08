import SuperNeo.EqPoly

namespace SuperNeo

open F

/-- Sum_x eq(x,z) * q(x) over the Boolean hypercube table q. -/
def eqLiftFromTable (qVals z : Array F) : F :=
  let ell := z.size
  let n := 2 ^ ell
  if qVals.size != n then
    0
  else
    Id.run do
      let mut acc : F := 0
      for mask in [0:n] do
        let x := bitsToFArray ell mask
        acc := acc + eqPoly x z * qVals[mask]!
      return acc

def eqLiftBooleanIndicator (qVals : Array F) (ell mask : Nat) : Bool :=
  if qVals.size != 2 ^ ell then
    false
  else if mask >= 2 ^ ell then
    false
  else
    let z := bitsToFArray ell mask
    decide (eqLiftFromTable qVals z = qVals[mask]!)

def eqLiftAllBoolean (qVals : Array F) (ell : Nat) : Bool :=
  if qVals.size != 2 ^ ell then
    false
  else
    Id.run do
      let mut ok := true
      for mask in [0:(2 ^ ell)] do
        ok := ok && eqLiftBooleanIndicator qVals ell mask
      return ok

/-- Proposition-level single-point eq-lift surface. -/
def eqLiftBooleanIndicatorProp (qVals : Array F) (ell mask : Nat) : Prop :=
  qVals.size = 2 ^ ell ∧
  mask < 2 ^ ell ∧
  eqLiftFromTable qVals (bitsToFArray ell mask) = qVals[mask]!

theorem eqLiftBooleanIndicator_sound
  {qVals : Array F} {ell mask : Nat}
  (hOk : eqLiftBooleanIndicator qVals ell mask = true) :
  eqLiftBooleanIndicatorProp qVals ell mask := by
  unfold eqLiftBooleanIndicator at hOk
  by_cases hSize : qVals.size = 2 ^ ell
  · by_cases hMask : mask < 2 ^ ell
    · have hMaskGe : ¬ mask >= 2 ^ ell := Nat.not_le_of_lt hMask
      have hDec :
          decide (eqLiftFromTable qVals (bitsToFArray ell mask) = qVals[mask]!) = true := by
        simpa [hSize, hMaskGe] using hOk
      exact ⟨hSize, hMask, decide_eq_true_eq.mp hDec⟩
    · have hMaskGe : mask >= 2 ^ ell := Nat.ge_of_not_lt hMask
      simp [hSize, hMaskGe] at hOk
  · simp [hSize] at hOk

theorem eqLiftBooleanIndicator_complete
  {qVals : Array F} {ell mask : Nat}
  (hProp : eqLiftBooleanIndicatorProp qVals ell mask) :
  eqLiftBooleanIndicator qVals ell mask = true := by
  rcases hProp with ⟨hSize, hMask, hEq⟩
  unfold eqLiftBooleanIndicator
  have hMaskGe : ¬ mask >= 2 ^ ell := Nat.not_le_of_lt hMask
  simp [hSize, hMaskGe, decide_eq_true hEq]

theorem eqLiftBooleanIndicator_eq_true_iff
  {qVals : Array F} {ell mask : Nat} :
  eqLiftBooleanIndicator qVals ell mask = true ↔
    eqLiftBooleanIndicatorProp qVals ell mask := by
  constructor
  · exact eqLiftBooleanIndicator_sound
  · exact eqLiftBooleanIndicator_complete

/-- Schwartz-Zippel failure bound interface: d / |S|. -/
def schwartzZippelBoundLeOne (totalDegree setSize : Nat) : Bool :=
  if setSize = 0 then
    false
  else
    decide (totalDegree <= setSize)

/-- Proposition-level SZ precondition surface. -/
def schwartzZippelBoundLeOneProp (totalDegree setSize : Nat) : Prop :=
  setSize ≠ 0 ∧ totalDegree <= setSize

theorem schwartzZippelBoundLeOne_sound
  {totalDegree setSize : Nat}
  (hOk : schwartzZippelBoundLeOne totalDegree setSize = true) :
  setSize ≠ 0 ∧ totalDegree <= setSize := by
  unfold schwartzZippelBoundLeOne at hOk
  by_cases hzero : setSize = 0
  · simp [hzero] at hOk
  · have hDec : decide (totalDegree <= setSize) = true := by
      simpa [hzero] using hOk
    exact ⟨hzero, decide_eq_true_eq.mp hDec⟩

theorem schwartzZippelBoundLeOne_complete
  {totalDegree setSize : Nat}
  (hNonzero : setSize ≠ 0)
  (hBound : totalDegree <= setSize) :
  schwartzZippelBoundLeOne totalDegree setSize = true := by
  unfold schwartzZippelBoundLeOne
  simp [hNonzero, decide_eq_true hBound]

theorem schwartzZippelBoundLeOne_eq_true_iff_prop
  {totalDegree setSize : Nat} :
  schwartzZippelBoundLeOne totalDegree setSize = true ↔
    schwartzZippelBoundLeOneProp totalDegree setSize := by
  constructor
  · exact schwartzZippelBoundLeOne_sound
  · intro hProp
    exact schwartzZippelBoundLeOne_complete hProp.1 hProp.2

def polyLemmaSanity : Bool :=
  let qVals : Array F := #[F.ofNat 3, F.ofNat 1, F.ofNat 4, F.ofNat 1, F.ofNat 5, F.ofNat 9, F.ofNat 2, F.ofNat 6]
  eqLiftAllBoolean qVals 3 && schwartzZippelBoundLeOne 5 17

end SuperNeo
