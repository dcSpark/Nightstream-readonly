import SuperNeo.ProtocolRelations

/-!
Strong interactive-reduction step `Π_CCS`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_CCS` reduction step. -/
abbrev PiCCSAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetAssumptions ctx

/-- Native assumptions consumed by the `Π_CCS` reduction step. -/
abbrev PiCCSNativeAssumptions (ctx : ProtocolTargetContext) :=
  ProtocolTargetNativeAssumptions ctx

/-- Strong `Π_CCS` target statement. -/
def piCCSStrongStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive the strong `Π_CCS` statement directly from the CE relation. -/
theorem piCCSStrong_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piCCSStrongStatement ctx := by
  exact ⟨hCE, ceClaimTrue_of_ce hCE⟩

/-- Derive strong `Π_CCS` statement from relation assumptions and transcript witness. -/
theorem piCCSStrong_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiCCSAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  have hCCS : ccsRelation ctx := protocolTargetProp_of_assumptions h
  exact piCCSStrong_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

/-- Derive strong `Π_CCS` statement from native relation assumptions. -/
theorem piCCSStrong_of_native_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiCCSNativeAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  have hCCS : ccsRelation ctx := protocolTargetProp_of_native_assumptions h
  exact piCCSStrong_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

end SuperNeo
