import SuperNeo.ProofSystem.Folding.PiCCS

/-!
Interface for `SuperNeo.ProofSystem.Folding.PiCCS`.

Spec: `specs/ProofSystem/Folding/PiCCS.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.3 (Π_CCS), lines 481–548
- Lemma 3 (Π_CCS is strong), line 545

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.Folding.PiCCSInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding.PiCCS"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "/Users/nicolasarqueros/starstream/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["PiCCSAssumptions", "StrongStatement", "soundness_relations"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["PiCCSAssumptions", "soundness_relations"]

/-- Interface scaffold marker; replace with concrete typed wrappers as closure progresses. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.Folding.PiCCSInterface

end SuperNeo
