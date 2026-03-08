import SuperNeo.ProofSystem.ConstraintSystem.CCS

/-!
Interface for `SuperNeo.ProofSystem.ConstraintSystem.CCS`.

Spec: `specs/ProofSystem/ConstraintSystem/CCS.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
- Definition 12 (Norm-bounded CCS), lines 457–459
- Definition 13 (Norm-bounded CCS Evaluation Relation), lines 461–465

This interface file is the typed boundary companion for the implementation.
-/

namespace SuperNeo

namespace ProofSystem.ConstraintSystem.CCSInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.ConstraintSystem.CCS"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "/Users/nicolasarqueros/starstream/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§2.1 Breaking down HyperNova (CCS/CE relations)", "§7.1 Relations"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["CCS", "CE", "CERelaxed", "CERelaxed.ofCE"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := []

/-- Interface scaffold marker; replace with concrete typed wrappers as closure progresses. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.ConstraintSystem.CCSInterface

end SuperNeo
