import SuperNeo.PiRLC
import SuperNeo.ProofSystem.Types

namespace SuperNeo.ProofSystem.Folding

abbrev PiRLCAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.PiRLCAssumptions ctx

abbrev WeakStatement (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.piRLCWeakStatement ctx

/-- Proof-system wrapper: weak relation theorem for `Π_RLC`. -/
theorem weak_relaxed
  {ctx : SuperNeo.ProtocolTargetContext}
  (h : PiRLCAssumptions ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  WeakStatement ctx := by
  exact SuperNeo.piRLCWeak_of_assumptions h hWitness

end SuperNeo.ProofSystem.Folding
