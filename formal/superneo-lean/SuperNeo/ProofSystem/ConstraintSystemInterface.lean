import SuperNeo.ProofSystem.ConstraintSystem

/-!
Interface for `SuperNeo.ProofSystem.ConstraintSystem`.

Spec: `specs/ProofSystem/ConstraintSystem.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 12 (Norm-bounded CCS), lines 457–459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461–465

Barrel re-export of CCS. This interface file is the typed boundary companion.
-/

namespace SuperNeo

namespace ProofSystem.ConstraintSystemInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.ConstraintSystem"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "/Users/nicolasarqueros/starstream/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§2.1 Breaking down HyperNova (CCS/CE relations)", "§7.1 Relations"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := []

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface scaffold marker; replace with concrete typed wrappers as closure progresses. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.ConstraintSystemInterface

end SuperNeo
