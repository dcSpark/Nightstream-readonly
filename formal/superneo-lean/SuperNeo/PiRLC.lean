import SuperNeo.PiCCS

/-!
Weak interactive-reduction step `Π_RLC`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_RLC` reduction step. -/
abbrev PiRLCAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetAssumptions ctx

/-- Native assumptions consumed by the `Π_RLC` reduction step. -/
abbrev PiRLCNativeAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetNativeAssumptions ctx

/-- Weak `Π_RLC` target statement. -/
def piRLCWeakStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelaxedRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive weak `Π_RLC` statement from strong `Π_CCS`. -/
theorem piRLCWeak_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiRLCAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  have hStrong : piCCSStrongStatement ctx :=
    piCCSStrong_of_assumptions h hWitness
  exact ⟨ceRelaxedRelation_of_ce hStrong.1, hStrong.2⟩

/-- Derive weak `Π_RLC` statement from native strong `Π_CCS`. -/
theorem piRLCWeak_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiRLCNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  have hStrong : piCCSStrongStatement ctx :=
    piCCSStrong_of_native_assumptions h hWitness
  exact ⟨ceRelaxedRelation_of_ce hStrong.1, hStrong.2⟩

end SuperNeo
