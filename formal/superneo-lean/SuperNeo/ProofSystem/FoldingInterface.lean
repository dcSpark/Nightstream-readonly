import SuperNeo.ProofSystem.Folding

/-!
Interface for `SuperNeo.ProofSystem.Folding`.

Spec: `specs/ProofSystem/Folding.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7.2–7.5 (Folding reductions), lines 468–593
- Lemma 3 (Π_CCS), Lemma 4 (Π_RLC), Theorem 7 (Π_DEC)

Barrel re-export of PiCCS, PiRLC, PiDEC. This interface file is the typed boundary companion.
-/

namespace SuperNeo

namespace ProofSystem.FoldingInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Folding"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "/Users/nicolasarqueros/starstream/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 interactive reductions", "§7.2-§7.5 folding reductions"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := []

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface scaffold marker; replace with concrete typed wrappers as closure progresses. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.FoldingInterface

end SuperNeo
