import SuperNeo.SumCheck

namespace SuperNeo

/-- Paper-facing degree bound for a coefficient-array encoding of a round polynomial. -/
def sumcheckRoundPolyDegreeLe (inst : SumCheckInstance) (poly : Array F) : Prop :=
  poly.size ≤ inst.maxDegree + 1

theorem sumcheckRoundPolyShape_degreeLe
  {inst : SumCheckInstance}
  {poly : Array F}
  (h : sumcheckRoundPolyShape inst poly) :
  sumcheckRoundPolyDegreeLe inst poly := by
  simpa [sumcheckRoundPolyDegreeLe, sumcheckRoundPolyShape] using Nat.le_of_eq h

/-- Paper-facing per-round degree bound extracted from the transcript encoding. -/
def sumcheckRoundDegrees
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  ∀ i : Fin tr.roundPolys.size,
    sumcheckRoundPolyDegreeLe inst tr.roundPolys[i.1]

/-- Paper-facing verifier checks: transcript shape, degree bound, initial, and fold. -/
def sumcheckVerifierAccepted
  (inst : SumCheckInstance)
  (tr : SumCheckTranscript) : Prop :=
  sumcheckRoundConsistent inst tr ∧
  sumcheckRoundDegrees inst tr ∧
  sumcheckInitialRoundConsistent inst tr ∧
  sumcheckFoldConsistent tr

theorem sumcheckVerifierAccepted_of_accepted
  {inst : SumCheckInstance}
  {tr : SumCheckTranscript}
  (hAcc : sumcheckAccepted inst tr) :
  sumcheckVerifierAccepted inst tr := by
  refine ⟨hAcc.2.2.1, ?_, hAcc.2.2.2.2.1, hAcc.2.2.2.2.2⟩
  intro i
  exact sumcheckRoundPolyShape_degreeLe (hAcc.2.2.2.1 i)

/--
The current standalone paper-claim surface is a scaffold:
it is equivalent to the legacy base-claim well-formedness conditions.
-/
theorem sumcheckPaperClaimTrue_iff_baseClaim
  {inst : SumCheckInstance} :
  sumcheckPaperClaimTrue inst ↔ sumcheckBaseClaimTrue inst := by
  constructor
  · intro h
    rcases h with ⟨stmt⟩
    exact ⟨stmt.parameterConsistent, stmt.degreeCompatible⟩
  · intro h
    exact sumcheckPaperClaimTrue_of_baseClaim_constructive h

theorem sumcheckClaimTrue_iff_baseClaim
  {inst : SumCheckInstance} :
  sumcheckClaimTrue inst ↔ sumcheckBaseClaimTrue inst :=
  sumcheckPaperClaimTrue_iff_baseClaim

end SuperNeo
