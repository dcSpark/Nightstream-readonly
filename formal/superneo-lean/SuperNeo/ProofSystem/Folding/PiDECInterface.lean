import SuperNeo.ProofSystem.Folding.PiDEC

/-!
Interface for `SuperNeo.ProofSystem.Folding.PiDEC`.

Spec: `specs/ProofSystem/Folding/PiDEC.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.5 (Π_DEC), lines 585–593
- Theorem 7 (Π_DEC is a reduction of knowledge), line 593

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.Folding.PiDECInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding.PiDEC"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "/Users/nicolasarqueros/starstream/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["PiDECAssumptions", "FinalStatement", "final_of_assumption"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["PiDECAssumptions", "final_of_assumption"]

/-- Interface scaffold marker; replace with concrete typed wrappers as closure progresses. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.Folding.PiDECInterface

end SuperNeo
