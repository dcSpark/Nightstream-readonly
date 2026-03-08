import SuperNeo.ProofSystem.Folding.PiRLC

/-!
Interface for `SuperNeo.ProofSystem.Folding.PiRLC`.

Spec: `specs/ProofSystem/Folding/PiRLC.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.4 (Π_RLC), lines 549–583
- Lemma 4 (Π_RLC is weak), line 581

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.Folding.PiRLCInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding.PiRLC"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "/Users/nicolasarqueros/starstream/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["PiRLCAssumptions", "WeakStatement", "weak_relaxed"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["PiRLCAssumptions"]

/-- Interface scaffold marker; replace with concrete typed wrappers as closure progresses. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.Folding.PiRLCInterface

end SuperNeo
