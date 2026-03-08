import SuperNeo.PiRLC

/-!
Reduction-of-knowledge step `Π_DEC`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_DEC` step. -/
abbrev PiDECAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetAssumptions ctx

/-- Native assumptions consumed by the `Π_DEC` step. -/
abbrev PiDECNativeAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetNativeAssumptions ctx

/-- Knowledge-style `Π_DEC` target statement. -/
def piDECKnowledgeStatement (ctx : ProtocolTargetContext) : Prop :=
  ∃ deltaInv : Coeffs,
    mulRq ctx.invDelta deltaInv = oneRq ∧
    ceRelaxedRelation ctx ∧
    SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive `Π_DEC` statement from weak relation and invertibility boundary. -/
theorem piDEC_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiDECAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  have hWeak : piRLCWeakStatement ctx :=
    piRLCWeak_of_assumptions h hWitness
  have hTarget : protocolTargetProp ctx := hWeak.1
  rcases hTarget with ⟨_hThm3, _hSplit, _hEvalHom, _hVecMod, _hScalMod, _hSampling,
      _hMleSize, _hMleId, _hInterp, hInvDelta⟩
  rcases hInvDelta with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hWeak.1, hWeak.2⟩

/-- Derive `Π_DEC` statement from native weak relation and invertibility boundary. -/
theorem piDEC_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiDECNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  have hWeak : piRLCWeakStatement ctx :=
    piRLCWeak_of_native_assumptions h hWitness
  have hTarget : protocolTargetProp ctx := hWeak.1
  rcases hTarget with ⟨_hThm3, _hSplit, _hEvalHom, _hVecMod, _hScalMod, _hSampling,
      _hMleSize, _hMleId, _hInterp, hInvDelta⟩
  rcases hInvDelta with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hWeak.1, hWeak.2⟩

end SuperNeo
