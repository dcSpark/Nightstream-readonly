import Mathlib.Algebra.Polynomial.Coeff
import Mathlib.Algebra.Polynomial.Div
import Mathlib.GroupTheory.OrderOfElement
import Mathlib.RingTheory.Coprime.Basic
import Mathlib.RingTheory.Polynomial.Cyclotomic.Factorization
import SuperNeo.InvertibilityAxioms
import SuperNeo.PolynomialBridge

namespace SuperNeo

open Polynomial

noncomputable section

/-- Concrete Goldilocks cube root of unity used to factor `Φ₈₁`. -/
def goldilocksOmega : Fq := ((2 ^ 32 - 1 : Nat) : Fq)

/-- First Goldilocks degree-27 factor of `Φ₈₁`. -/
def phiA : Polynomial Fq := X ^ 27 - C goldilocksOmega

/-- Second Goldilocks degree-27 factor of `Φ₈₁`. -/
def phiB : Polynomial Fq := X ^ 27 - C (goldilocksOmega ^ 2)

private def quadA : Polynomial Fq := X - C goldilocksOmega
private def quadB : Polynomial Fq := X - C (goldilocksOmega ^ 2)

/-- `q mod 81` as a unit, used for the exact-order argument. -/
def modulusUnit81 : (ZMod Parameters.Goldilocks.eta)ˣ :=
  ZMod.unitOfCoprime Parameters.Goldilocks.modulus (by native_decide)

private theorem goldilocksOmega_quad :
    goldilocksOmega ^ 2 + goldilocksOmega + 1 = 0 := by
  native_decide

private theorem goldilocksOmega_cubic :
    goldilocksOmega ^ 3 = 1 := by
  native_decide

private theorem goldilocksOmega_ne_one :
    goldilocksOmega ≠ 1 := by
  native_decide

private theorem goldilocksOmega_sub_sq_ne_zero :
    goldilocksOmega - goldilocksOmega ^ 2 ≠ 0 := by
  native_decide

private theorem modulusUnit81_order :
    orderOf modulusUnit81 = 27 := by
  have h9 : modulusUnit81 ^ 3 ^ 2 ≠ 1 := by
    simpa using (show modulusUnit81 ^ 9 ≠ 1 by native_decide)
  have h27 : modulusUnit81 ^ 3 ^ 3 = 1 := by
    simpa using (show modulusUnit81 ^ 27 = 1 by native_decide)
  simpa [pow_succ'] using orderOf_eq_prime_pow (x := modulusUnit81) (p := 3) (n := 2) h9 h27

private theorem qUnitMod81_order :
    orderOf (ZMod.unitOfCoprime Goldilocks.q
      ((Goldilocks.q_prime).coprime_iff_not_dvd.mpr (by native_decide : ¬ Goldilocks.q ∣ 81))) = 27 := by
  simpa [modulusUnit81, Parameters.Goldilocks.modulus, Parameters.Goldilocks.eta_eq_81] using
    modulusUnit81_order

private theorem quad_factor :
    quadA * quadB = X ^ 2 + X + 1 := by
  have hsum : goldilocksOmega + goldilocksOmega ^ 2 = (-1 : Fq) := by
    calc
      goldilocksOmega + goldilocksOmega ^ 2
          = (goldilocksOmega ^ 2 + goldilocksOmega + 1) - 1 := by ring
      _ = -1 := by simp [goldilocksOmega_quad]
  have hprod : goldilocksOmega * goldilocksOmega ^ 2 = (1 : Fq) := by
    simpa [pow_succ', mul_assoc, mul_comm] using goldilocksOmega_cubic
  have hlin :
      -(X * C goldilocksOmega) - X * C (goldilocksOmega ^ 2) = (X : Polynomial Fq) := by
    calc
      -(X * C goldilocksOmega) - X * C (goldilocksOmega ^ 2)
          = -(X * C goldilocksOmega + X * C (goldilocksOmega ^ 2)) := by ring
      _ = -(X * (C goldilocksOmega + C (goldilocksOmega ^ 2))) := by rw [mul_add]
      _ = -(X * C (goldilocksOmega + goldilocksOmega ^ 2)) := by rw [← C_add]
      _ = -(X * C (-1 : Fq)) := by rw [hsum]
      _ = X := by simp
  unfold quadA quadB
  ring_nf
  rw [hlin]
  rw [show C goldilocksOmega * C (goldilocksOmega ^ 2) = (1 : Polynomial Fq) by
    simpa [C_mul] using congrArg C hprod]
  ring

private theorem cyclotomic81_eq_phiPolynomial :
    Polynomial.cyclotomic (3 ^ 4) Fq = phiPolynomial := by
  have hgeom :
      Polynomial.cyclotomic (3 ^ 4) Fq =
        ∑ i ∈ Finset.range 3, (X ^ (3 ^ 3)) ^ i := by
    simpa using
      (Polynomial.cyclotomic_prime_pow_eq_geom_sum (R := Fq) (p := 3) (n := 3)
        (by native_decide : Nat.Prime 3))
  rw [hgeom]
  rw [phiPolynomial_def]
  simp [Finset.sum_range_succ]
  have hsq : ((X ^ 27 : Polynomial Fq) ^ 2) = X ^ 54 := by
    rw [pow_two, ← pow_add]
  rw [hsq]
  ring

private theorem phi_factor :
    phiA * phiB = phiPolynomial := by
  have hcomp :
      (quadA * quadB).comp (X ^ 27) = phiA * phiB := by
    rw [Polynomial.mul_comp]
    simp [phiA, phiB, quadA, quadB]
  calc
    phiA * phiB = (quadA * quadB).comp (X ^ 27) := hcomp.symm
    _ = (X ^ 2 + X + 1).comp (X ^ 27) := by rw [quad_factor]
    _ = phiPolynomial := by
      rw [phiPolynomial_def]
      have hsq : ((X ^ 27 : Polynomial Fq) ^ 2) = X ^ 54 := by
        rw [pow_two, ← pow_add]
      simp [hsq]

private theorem phiA_monic : phiA.Monic := by
  change (X ^ 27 - C goldilocksOmega).Monic
  exact Polynomial.monic_X_pow_sub_C (a := goldilocksOmega)
    (n := 27) (show (27 : Nat) ≠ 0 by decide)

private theorem phiB_monic : phiB.Monic := by
  change (X ^ 27 - C (goldilocksOmega ^ 2)).Monic
  exact Polynomial.monic_X_pow_sub_C (a := goldilocksOmega ^ 2)
    (n := 27) (show (27 : Nat) ≠ 0 by decide)

private theorem phiA_natDegree : phiA.natDegree = 27 := by
  change (X ^ 27 - C goldilocksOmega).natDegree = 27
  exact Polynomial.natDegree_X_pow_sub_C (R := Fq) (n := 27) (r := goldilocksOmega)

private theorem phiB_natDegree : phiB.natDegree = 27 := by
  change (X ^ 27 - C (goldilocksOmega ^ 2)).natDegree = 27
  exact Polynomial.natDegree_X_pow_sub_C (R := Fq) (n := 27) (r := goldilocksOmega ^ 2)

private theorem phiA_dvd_cyclotomic81 :
    phiA ∣ Polynomial.cyclotomic 81 Fq := by
  have hcyclo : Polynomial.cyclotomic 81 Fq = phiPolynomial := by
    simpa using cyclotomic81_eq_phiPolynomial
  rw [hcyclo, ← phi_factor]
  exact dvd_mul_right _ _

private theorem phiB_dvd_cyclotomic81 :
    phiB ∣ Polynomial.cyclotomic 81 Fq := by
  have hcyclo : Polynomial.cyclotomic 81 Fq = phiPolynomial := by
    simpa using cyclotomic81_eq_phiPolynomial
  rw [hcyclo, ← phi_factor]
  exact dvd_mul_left _ _

private theorem phiA_irreducible : Irreducible phiA := by
  have hpn : ¬ Goldilocks.q ∣ 81 := by
    native_decide
  exact ZMod.irreducible_of_dvd_cyclotomic_of_natDegree (p := Goldilocks.q) (n := 81)
    hpn phiA_dvd_cyclotomic81 (by simpa using phiA_natDegree.trans qUnitMod81_order.symm)

private theorem phiB_irreducible : Irreducible phiB := by
  have hpn : ¬ Goldilocks.q ∣ 81 := by
    native_decide
  exact ZMod.irreducible_of_dvd_cyclotomic_of_natDegree (p := Goldilocks.q) (n := 81)
    hpn phiB_dvd_cyclotomic81 (by simpa using phiB_natDegree.trans qUnitMod81_order.symm)

private theorem phiA_degree :
    phiA.degree = 27 := by
  simpa [phiA_natDegree] using (Polynomial.degree_eq_natDegree phiA_monic.ne_zero)

private theorem phiB_degree :
    phiB.degree = 27 := by
  simpa [phiB_natDegree] using (Polynomial.degree_eq_natDegree phiB_monic.ne_zero)

/-- Lower 27 coefficients as a polynomial. -/
private noncomputable def loPoly (a : Coeffs) : Polynomial Fq :=
  Finset.sum (Finset.range 27) fun i => Polynomial.monomial i (fToZMod (coeffAt a i))

/-- Upper 27 coefficients as a polynomial. -/
private noncomputable def hiPoly (a : Coeffs) : Polynomial Fq :=
  Finset.sum (Finset.range 27) fun i => Polynomial.monomial i (fToZMod (coeffAt a (i + 27)))

private theorem coeff_loPoly (a : Coeffs) (i : Nat) :
    (loPoly a).coeff i = if i < 27 then fToZMod (coeffAt a i) else 0 := by
  by_cases hi : i < 27
  · have hiMem : i ∈ Finset.range 27 := by simp [hi]
    calc
      (loPoly a).coeff i
          = Finset.sum (Finset.range 27) (fun j =>
              (Polynomial.monomial j (fToZMod (coeffAt a j))).coeff i) := by
                simp [loPoly]
      _ = (Polynomial.monomial i (fToZMod (coeffAt a i))).coeff i := by
            apply Finset.sum_eq_single i
            · intro j hj hji
              simp [Polynomial.coeff_monomial, hji]
            · intro hNotMem
              exact False.elim (hNotMem hiMem)
      _ = fToZMod (coeffAt a i) := by simp [Polynomial.coeff_monomial]
      _ = if i < 27 then fToZMod (coeffAt a i) else 0 := by simp [hi]
  · have hiNotMem : i ∉ Finset.range 27 := by simp [hi]
    calc
      (loPoly a).coeff i
          = Finset.sum (Finset.range 27) (fun j =>
              (Polynomial.monomial j (fToZMod (coeffAt a j))).coeff i) := by
                simp [loPoly]
      _ = 0 := by
            apply Finset.sum_eq_zero
            intro j hj
            have hji : j ≠ i := by
              intro hEq
              exact hiNotMem (by simpa [hEq] using hj)
            simp [Polynomial.coeff_monomial, hji]
      _ = if i < 27 then fToZMod (coeffAt a i) else 0 := by simp [hi]

private theorem coeff_hiPoly (a : Coeffs) (i : Nat) :
    (hiPoly a).coeff i = if i < 27 then fToZMod (coeffAt a (i + 27)) else 0 := by
  by_cases hi : i < 27
  · have hiMem : i ∈ Finset.range 27 := by simp [hi]
    calc
      (hiPoly a).coeff i
          = Finset.sum (Finset.range 27) (fun j =>
              (Polynomial.monomial j (fToZMod (coeffAt a (j + 27)))).coeff i) := by
                simp [hiPoly]
      _ = (Polynomial.monomial i (fToZMod (coeffAt a (i + 27)))).coeff i := by
            apply Finset.sum_eq_single i
            · intro j hj hji
              simp [Polynomial.coeff_monomial, hji]
            · intro hNotMem
              exact False.elim (hNotMem hiMem)
      _ = fToZMod (coeffAt a (i + 27)) := by simp [Polynomial.coeff_monomial]
      _ = if i < 27 then fToZMod (coeffAt a (i + 27)) else 0 := by simp [hi]
  · have hiNotMem : i ∉ Finset.range 27 := by simp [hi]
    calc
      (hiPoly a).coeff i
          = Finset.sum (Finset.range 27) (fun j =>
              (Polynomial.monomial j (fToZMod (coeffAt a (j + 27)))).coeff i) := by
                simp [hiPoly]
      _ = 0 := by
            apply Finset.sum_eq_zero
            intro j hj
            have hji : j ≠ i := by
              intro hEq
              exact hiNotMem (by simpa [hEq] using hj)
            simp [Polynomial.coeff_monomial, hji]
      _ = if i < 27 then fToZMod (coeffAt a (i + 27)) else 0 := by simp [hi]

private theorem coeffsToPolynomial_eq_lo_add_X27_hi
    {a : Coeffs}
    (ha : hasRingDegreeShape a) :
    coeffsToPolynomial a = loPoly a + X ^ 27 * hiPoly a := by
  ext n
  by_cases h27 : n < 27
  · have hd : 27 < d := by native_decide
    have hltD : n < d := lt_trans h27 hd
    have hMul : (X ^ 27 * hiPoly a).coeff n = 0 := by
      rw [coeff_X_pow_mul', if_neg (Nat.not_le_of_lt h27)]
    simp [coeff_coeffsToPolynomial, coeff_loPoly, coeff_add, h27, hltD, hMul]
  · by_cases h54 : n < 54
    · have hge27 : 27 ≤ n := Nat.le_of_not_lt h27
      have hsub : n - 27 < 27 := by omega
      have hltD : n < d := by simpa [d] using h54
      have hLo : (loPoly a).coeff n = 0 := by simp [coeff_loPoly, h27]
      have hHi : (hiPoly a).coeff (n - 27) = fToZMod (coeffAt a n) := by
        have : n - 27 + 27 = n := by omega
        simpa [coeff_hiPoly, hsub, this]
      rw [coeff_coeffsToPolynomial, if_pos hltD, coeff_add, hLo, zero_add,
        coeff_X_pow_mul', if_pos hge27, hHi]
    · have hCoeff : (coeffsToPolynomial a).coeff n = 0 := by
        have hNotLtD : ¬ n < d := by simpa [d] using h54
        simp [coeff_coeffsToPolynomial, hNotLtD]
      have hLo : (loPoly a).coeff n = 0 := by simp [coeff_loPoly, h27]
      have hHi : (X ^ 27 * hiPoly a).coeff n = 0 := by
        have hsub : ¬ (n - 27 < 27) := by omega
        rw [coeff_X_pow_mul', if_pos (by omega), coeff_hiPoly]
        simp [hsub]
      simp [hCoeff, coeff_add, hLo, hHi]

private theorem degree_lo_add_C_mul_hi_lt
    (a : Coeffs) (c : Fq) :
    (loPoly a + C c * hiPoly a).degree < (27 : WithBot ℕ) := by
  refine (degree_lt_iff_coeff_zero _ _).2 ?_
  intro n hn
  rw [coeff_add, coeff_loPoly, coeff_C_mul, coeff_hiPoly]
  have hNot : ¬ n < 27 := not_lt_of_ge hn
  simp [hNot]

private theorem mod_phiA_eq_lo_add_omega_hi
    {a : Coeffs}
    (ha : hasRingDegreeShape a) :
    coeffsToPolynomial a %ₘ phiA = loPoly a + C goldilocksOmega * hiPoly a := by
  refine (Polynomial.div_modByMonic_unique (f := coeffsToPolynomial a) (g := phiA)
    (q := hiPoly a) (r := loPoly a + C goldilocksOmega * hiPoly a) phiA_monic ?_).2
  constructor
  · calc
      (loPoly a + C goldilocksOmega * hiPoly a) + phiA * hiPoly a
          = loPoly a + X ^ 27 * hiPoly a := by
              simp [phiA, sub_eq_add_neg]
              ring
      _ = coeffsToPolynomial a := by simpa using (coeffsToPolynomial_eq_lo_add_X27_hi ha).symm
  · rw [phiA_degree]
    exact degree_lo_add_C_mul_hi_lt a goldilocksOmega

private theorem mod_phiB_eq_lo_add_omegaSq_hi
    {a : Coeffs}
    (ha : hasRingDegreeShape a) :
    coeffsToPolynomial a %ₘ phiB = loPoly a + C (goldilocksOmega ^ 2) * hiPoly a := by
  refine (Polynomial.div_modByMonic_unique (f := coeffsToPolynomial a) (g := phiB)
    (q := hiPoly a) (r := loPoly a + C (goldilocksOmega ^ 2) * hiPoly a) phiB_monic ?_).2
  constructor
  · calc
      (loPoly a + C (goldilocksOmega ^ 2) * hiPoly a) + phiB * hiPoly a
          = loPoly a + X ^ 27 * hiPoly a := by
              simp [phiB, sub_eq_add_neg]
              ring
      _ = coeffsToPolynomial a := by simpa using (coeffsToPolynomial_eq_lo_add_X27_hi ha).symm
  · rw [phiB_degree]
    exact degree_lo_add_C_mul_hi_lt a (goldilocksOmega ^ 2)

@[simp] private theorem fToZMod_ofNat (n : Nat) :
    fToZMod (F.ofNat n) = (n : Fq) := by
  simp [fToZMod, F.ofNat]

@[simp] private theorem fToZMod_neg_ofNat (n : Nat) :
    fToZMod (-F.ofNat n) = -((n : Nat) : Fq) := by
  simp [fToZMod_neg]

private theorem nat_add_omega_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (m : Fq) + goldilocksOmega * (n : Fq) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((m : Fq) + goldilocksOmega * (n : Fq) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_add_neg_omega_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (m : Fq) + goldilocksOmega * (-(n : Fq)) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((m : Fq) + goldilocksOmega * (-(n : Fq)) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_neg_add_omega_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (-(m : Fq)) + goldilocksOmega * (n : Fq) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((-(m : Fq)) + goldilocksOmega * (n : Fq) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_neg_add_neg_omega_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (-(m : Fq)) + goldilocksOmega * (-(n : Fq)) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((-(m : Fq)) + goldilocksOmega * (-(n : Fq)) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_add_omegaSq_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (m : Fq) + goldilocksOmega ^ 2 * (n : Fq) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((m : Fq) + goldilocksOmega ^ 2 * (n : Fq) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_add_neg_omegaSq_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (m : Fq) + goldilocksOmega ^ 2 * (-(n : Fq)) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((m : Fq) + goldilocksOmega ^ 2 * (-(n : Fq)) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_neg_add_omegaSq_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (-(m : Fq)) + goldilocksOmega ^ 2 * (n : Fq) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((-(m : Fq)) + goldilocksOmega ^ 2 * (n : Fq) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem nat_neg_add_neg_omegaSq_eq_zero
    {m n : Nat}
    (hm : m ≤ 4)
    (hn : n ≤ 4)
    (h : (-(m : Fq)) + goldilocksOmega ^ 2 * (-(n : Fq)) = 0) :
    m = 0 ∧ n = 0 := by
  have hAll : ∀ m n : Nat,
      m ≤ 4 →
      n ≤ 4 →
      ((-(m : Fq)) + goldilocksOmega ^ 2 * (-(n : Fq)) = 0 → m = 0 ∧ n = 0) := by
    intro m n hm hn
    interval_cases m <;> interval_cases n <;> native_decide
  exact hAll m n hm hn h

private theorem small_add_omega_eq_zero
    {x y : F}
    (hx : normInfF x ≤ 4)
    (hy : normInfF y ≤ 4)
    (hxy : fToZMod x + goldilocksOmega * fToZMod y = 0) :
    x = 0 ∧ y = 0 := by
  rcases F.exists_smallNat_or_neg_of_centeredAbs_le_four x (by simpa [normInfF] using hx) with
    ⟨m, hm, hxPos | hxNeg⟩
  · rcases F.exists_smallNat_or_neg_of_centeredAbs_le_four y (by simpa [normInfF] using hy) with
      ⟨n, hn, hyPos | hyNeg⟩
    · subst hxPos
      subst hyPos
      rcases nat_add_omega_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]
    · subst hxPos
      subst hyNeg
      rcases nat_add_neg_omega_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]
  · rcases F.exists_smallNat_or_neg_of_centeredAbs_le_four y (by simpa [normInfF] using hy) with
      ⟨n, hn, hyPos | hyNeg⟩
    · subst hxNeg
      subst hyPos
      rcases nat_neg_add_omega_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]
    · subst hxNeg
      subst hyNeg
      rcases nat_neg_add_neg_omega_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]

private theorem small_add_omegaSq_eq_zero
    {x y : F}
    (hx : normInfF x ≤ 4)
    (hy : normInfF y ≤ 4)
    (hxy : fToZMod x + goldilocksOmega ^ 2 * fToZMod y = 0) :
    x = 0 ∧ y = 0 := by
  rcases F.exists_smallNat_or_neg_of_centeredAbs_le_four x (by simpa [normInfF] using hx) with
    ⟨m, hm, hxPos | hxNeg⟩
  · rcases F.exists_smallNat_or_neg_of_centeredAbs_le_four y (by simpa [normInfF] using hy) with
      ⟨n, hn, hyPos | hyNeg⟩
    · subst hxPos
      subst hyPos
      rcases nat_add_omegaSq_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]
    · subst hxPos
      subst hyNeg
      rcases nat_add_neg_omegaSq_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]
  · rcases F.exists_smallNat_or_neg_of_centeredAbs_le_four y (by simpa [normInfF] using hy) with
      ⟨n, hn, hyPos | hyNeg⟩
    · subst hxNeg
      subst hyPos
      rcases nat_neg_add_omegaSq_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]
    · subst hxNeg
      subst hyNeg
      rcases nat_neg_add_neg_omegaSq_eq_zero hm hn (by simpa using hxy) with ⟨hm0, hn0⟩
      constructor <;> simp [hm0, hn0]

private theorem not_dvd_phiA_of_shape_norm_le_four_ne_zeroRq
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a ≤ 4)
    (hNe : a ≠ zeroRq) :
    ¬ phiA ∣ coeffsToPolynomial a := by
  intro hDiv
  have hmod0 : coeffsToPolynomial a %ₘ phiA = 0 := by
    exact (Polynomial.modByMonic_eq_zero_iff_dvd phiA_monic).2 hDiv
  have hrem : loPoly a + C goldilocksOmega * hiPoly a = 0 := by
    simpa [mod_phiA_eq_lo_add_omega_hi ha] using hmod0
  have hCoeffZero :
      ∀ i : Fin 27, coeffAt a i.1 = 0 ∧ coeffAt a (i.1 + 27) = 0 := by
    intro i
    have hx : normInfF (coeffAt a i.1) ≤ 4 := by
      exact Nat.le_trans (normInfF_coeffAt_le_normInfCoeffs a i.1) hNorm
    have hy : normInfF (coeffAt a (i.1 + 27)) ≤ 4 := by
      exact Nat.le_trans (normInfF_coeffAt_le_normInfCoeffs a (i.1 + 27)) hNorm
    have hcoeff : fToZMod (coeffAt a i.1) + goldilocksOmega * fToZMod (coeffAt a (i.1 + 27)) = 0 := by
      have hcoeff0 : (loPoly a).coeff i.1 + (C goldilocksOmega * hiPoly a).coeff i.1 = 0 := by
        simpa [coeff_add] using congrArg (fun p => p.coeff i.1) hrem
      have hmult : (C goldilocksOmega * hiPoly a).coeff i.1 =
          goldilocksOmega * fToZMod (coeffAt a (i.1 + 27)) := by
        rw [coeff_C_mul, coeff_hiPoly]
        simp [i.2]
      calc
        fToZMod (coeffAt a i.1) + goldilocksOmega * fToZMod (coeffAt a (i.1 + 27))
            = (loPoly a).coeff i.1 + (C goldilocksOmega * hiPoly a).coeff i.1 := by
                calc
                  fToZMod (coeffAt a i.1) + goldilocksOmega * fToZMod (coeffAt a (i.1 + 27))
                      = (loPoly a).coeff i.1 + goldilocksOmega * fToZMod (coeffAt a (i.1 + 27)) := by
                          simp [coeff_loPoly, i.2]
                  _ = (loPoly a).coeff i.1 + (C goldilocksOmega * hiPoly a).coeff i.1 := by
                          rw [hmult]
        _ = 0 := hcoeff0
    exact small_add_omega_eq_zero hx hy hcoeff
  have hZero : a = zeroRq := by
    apply Array.ext
    · simpa [hasRingDegreeShape] using ha
    · intro j hjA hjZ
      have hjA' := hjA
      have hj : j < d := by
        rw [ha] at hjA'
        exact hjA'
      by_cases h27 : j < 27
      · let i : Fin 27 := ⟨j, h27⟩
        have hEq0 : coeffAt a j = 0 := by
          simpa [i] using (hCoeffZero i).1
        have hAj : a[j]'hjA = 0 := by
          calc
            a[j]'hjA = a.getD j 0 := Array.getElem_eq_getD (xs := a) (i := j) (h := hjA) (fallback := (0 : F))
            _ = coeffAt a j := by
                  simp [coeffAt, hj]
            _ = 0 := hEq0
        have hZj : zeroRq[j]'hjZ = 0 := by
          simp [zeroRq]
        exact hAj.trans hZj.symm
      · have hge27 : 27 ≤ j := Nat.le_of_not_lt h27
        have hjlt54 : j < 54 := by simpa [d] using hj
        have hm : j - 27 < 27 := by omega
        let i : Fin 27 := ⟨j - 27, hm⟩
        have hij : i.1 + 27 = j := by
          simp [i]
          omega
        have hEq0 : coeffAt a j = 0 := by
          simpa [i, hij] using (hCoeffZero i).2
        have hAj : a[j]'hjA = 0 := by
          calc
            a[j]'hjA = a.getD j 0 := Array.getElem_eq_getD (xs := a) (i := j) (h := hjA) (fallback := (0 : F))
            _ = coeffAt a j := by
                  simp [coeffAt, hj]
            _ = 0 := hEq0
        have hZj : zeroRq[j]'hjZ = 0 := by
          simp [zeroRq]
        exact hAj.trans hZj.symm
  exact hNe hZero

private theorem not_dvd_phiB_of_shape_norm_le_four_ne_zeroRq
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a ≤ 4)
    (hNe : a ≠ zeroRq) :
    ¬ phiB ∣ coeffsToPolynomial a := by
  intro hDiv
  have hmod0 : coeffsToPolynomial a %ₘ phiB = 0 := by
    exact (Polynomial.modByMonic_eq_zero_iff_dvd phiB_monic).2 hDiv
  have hrem : loPoly a + C (goldilocksOmega ^ 2) * hiPoly a = 0 := by
    simpa [mod_phiB_eq_lo_add_omegaSq_hi ha] using hmod0
  have hCoeffZero :
      ∀ i : Fin 27, coeffAt a i.1 = 0 ∧ coeffAt a (i.1 + 27) = 0 := by
    intro i
    have hx : normInfF (coeffAt a i.1) ≤ 4 := by
      exact Nat.le_trans (normInfF_coeffAt_le_normInfCoeffs a i.1) hNorm
    have hy : normInfF (coeffAt a (i.1 + 27)) ≤ 4 := by
      exact Nat.le_trans (normInfF_coeffAt_le_normInfCoeffs a (i.1 + 27)) hNorm
    have hcoeff : fToZMod (coeffAt a i.1) + goldilocksOmega ^ 2 * fToZMod (coeffAt a (i.1 + 27)) = 0 := by
      have hcoeff0 : (loPoly a).coeff i.1 + (C (goldilocksOmega ^ 2) * hiPoly a).coeff i.1 = 0 := by
        simpa [coeff_add] using congrArg (fun p => p.coeff i.1) hrem
      have hmult : (C (goldilocksOmega ^ 2) * hiPoly a).coeff i.1 =
          goldilocksOmega ^ 2 * fToZMod (coeffAt a (i.1 + 27)) := by
        rw [coeff_C_mul, coeff_hiPoly]
        simp [i.2]
      calc
        fToZMod (coeffAt a i.1) + goldilocksOmega ^ 2 * fToZMod (coeffAt a (i.1 + 27))
            = (loPoly a).coeff i.1 + (C (goldilocksOmega ^ 2) * hiPoly a).coeff i.1 := by
                calc
                  fToZMod (coeffAt a i.1) + goldilocksOmega ^ 2 * fToZMod (coeffAt a (i.1 + 27))
                      = (loPoly a).coeff i.1 + goldilocksOmega ^ 2 * fToZMod (coeffAt a (i.1 + 27)) := by
                          simp [coeff_loPoly, i.2]
                  _ = (loPoly a).coeff i.1 + (C (goldilocksOmega ^ 2) * hiPoly a).coeff i.1 := by
                          rw [hmult]
        _ = 0 := hcoeff0
    exact small_add_omegaSq_eq_zero hx hy hcoeff
  have hZero : a = zeroRq := by
    apply Array.ext
    · simpa [hasRingDegreeShape] using ha
    · intro j hjA hjZ
      have hjA' := hjA
      have hj : j < d := by
        rw [ha] at hjA'
        exact hjA'
      by_cases h27 : j < 27
      · let i : Fin 27 := ⟨j, h27⟩
        have hEq0 : coeffAt a j = 0 := by
          simpa [i] using (hCoeffZero i).1
        have hAj : a[j]'hjA = 0 := by
          calc
            a[j]'hjA = a.getD j 0 := Array.getElem_eq_getD (xs := a) (i := j) (h := hjA) (fallback := (0 : F))
            _ = coeffAt a j := by
                  simp [coeffAt, hj]
            _ = 0 := hEq0
        have hZj : zeroRq[j]'hjZ = 0 := by
          simp [zeroRq]
        exact hAj.trans hZj.symm
      · have hge27 : 27 ≤ j := Nat.le_of_not_lt h27
        have hjlt54 : j < 54 := by simpa [d] using hj
        have hm : j - 27 < 27 := by omega
        let i : Fin 27 := ⟨j - 27, hm⟩
        have hij : i.1 + 27 = j := by
          simp [i]
          omega
        have hEq0 : coeffAt a j = 0 := by
          simpa [i, hij] using (hCoeffZero i).2
        have hAj : a[j]'hjA = 0 := by
          calc
            a[j]'hjA = a.getD j 0 := Array.getElem_eq_getD (xs := a) (i := j) (h := hjA) (fallback := (0 : F))
            _ = coeffAt a j := by
                  simp [coeffAt, hj]
            _ = 0 := hEq0
        have hZj : zeroRq[j]'hjZ = 0 := by
          simp [zeroRq]
        exact hAj.trans hZj.symm
  exact hNe hZero

private theorem phiPolynomial_monic : phiPolynomial.Monic := by
  rw [← phi_factor]
  exact phiA_monic.mul phiB_monic

private theorem phiPolynomial_natDegree :
    phiPolynomial.natDegree = d := by
  rw [← phi_factor, phiA_monic.natDegree_mul' phiB_monic.ne_zero, phiA_natDegree, phiB_natDegree]
  norm_num [d]

private theorem phiPolynomial_degree :
    phiPolynomial.degree = d := by
  simpa [phiPolynomial_natDegree] using
    (Polynomial.degree_eq_natDegree phiPolynomial_monic.ne_zero)

private theorem degree_polyMulMod_lt (a b : Coeffs) :
    ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial).degree < d := by
  have h := Polynomial.degree_modByMonic_lt
    (coeffsToPolynomial a * coeffsToPolynomial b) phiPolynomial_monic
  simpa [phiPolynomial_degree] using h

private theorem coeffsToPolynomial_polyMulMod
    (a b : Coeffs) :
    coeffsToPolynomial
      (polynomialToCoeffs ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial)) =
        ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial) := by
  exact coeffsToPolynomial_polynomialToCoeffs_of_degree_lt (degree_polyMulMod_lt a b)

private def basisVecNat (i : Nat) : Array F :=
  Array.ofFn (fun j : Fin d => if j.1 = i then (1 : F) else 0)

@[simp] private theorem basisVecNat_size (i : Nat) : (basisVecNat i).size = d := by
  simp [basisVecNat]

private def monomialReduce (n : Nat) : Coeffs :=
  let r := n % 81
  if hLt : r < d then
    basisVecNat r
  else
    vecAdd
      (vecScale (-1 : F) (basisVecNat (r - 54)))
      (vecScale (-1 : F) (basisVecNat (r - 27)))

@[simp] private theorem monomialReduce_size (n : Nat) :
    (monomialReduce n).size = d := by
  unfold monomialReduce
  dsimp
  split
  · simp [basisVecNat]
  · calc
      (vecAdd
        (vecScale (-1 : F) (basisVecNat (n % 81 - 54)))
        (vecScale (-1 : F) (basisVecNat (n % 81 - 27)))).size
          = (vecScale (-1 : F) (basisVecNat (n % 81 - 54))).size := by
              exact vecAdd_size_of_eq (by simp [basisVecNat])
      _ = d := by simp [basisVecNat]

set_option maxRecDepth 4096 in
private theorem mulRqPhi_basis_basis :
    ∀ i j : Fin d,
      mulRqPhi (basisVecNat i.1) (basisVecNat j.1) =
        monomialReduce (i.1 + j.1) := by
  native_decide

private theorem f_mul_zero (a : F) : a * 0 = 0 := by
  simpa using (Fin.mul_zero (n := Goldilocks.q) a)

private theorem f_add_zero (a : F) : a + 0 = a := by
  simpa using (Fin.add_zero (n := Goldilocks.q) a)

private def basisExpandPrefix (a : Array F) : Nat → Array F
  | 0 => Array.replicate d (0 : F)
  | n + 1 => vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))

private def basisExpand (a : Array F) : Array F :=
  basisExpandPrefix a d

private theorem basisExpandPrefix_size (a : Array F) (n : Nat) :
    (basisExpandPrefix a n).size = d := by
  induction n with
  | zero => simp [basisExpandPrefix, d]
  | succ n ih =>
      calc
        (basisExpandPrefix a (n + 1)).size
            = (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))).size := by
                simp [basisExpandPrefix]
        _ = (basisExpandPrefix a n).size := by
              apply vecAdd_size_of_eq
              simp [ih, basisVecNat]
        _ = d := ih

private theorem coeffAt_basisExpandPrefix_of_lt
    (a : Array F)
    (j n : Nat)
    (hj : j < d) :
    coeffAt (basisExpandPrefix a n) j = if j < n then a[j]! else 0 := by
  induction n with
  | zero =>
      simp [basisExpandPrefix, coeffAt, hj]
  | succ n ih =>
      have hSize : (basisExpandPrefix a n).size = d := by simp [basisExpandPrefix_size]
      have hScaledSize : (vecScale a[n]! (basisVecNat n)).size = d := by
        simp [basisVecNat]
      calc
        coeffAt (basisExpandPrefix a (n + 1)) j
            = coeffAt (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) j := by
                simp [basisExpandPrefix]
        _ = coeffAt (basisExpandPrefix a n) j +
              coeffAt (vecScale a[n]! (basisVecNat n)) j := by
                exact coeffAt_vecAdd_of_size_d
                  (basisExpandPrefix a n)
                  (vecScale a[n]! (basisVecNat n))
                  hSize hScaledSize j hj
        _ = (if j < n then a[j]! else 0) +
              coeffAt (vecScale a[n]! (basisVecNat n)) j := by
                simp [ih]
        _ = (if j < n then a[j]! else 0) +
              a[n]! * coeffAt (basisVecNat n) j := by
                rw [coeffAt_vecScale_of_size_d (s := a[n]!) (x := basisVecNat n)
                  (hx := by simp [basisVecNat]) (k := j) hj]
        _ = if j < n + 1 then a[j]! else 0 := by
              by_cases hjn : j < n
              · have hne : j ≠ n := by omega
                have hlt : j < n + 1 := by omega
                calc
                  (if j < n then a[j]! else 0) + a[n]! * coeffAt (basisVecNat n) j
                      = a[j]! + a[n]! * 0 := by
                          simp [hjn, hne, basisVecNat, coeffAt, hj]
                  _ = a[j]! + 0 := by simp [f_mul_zero]
                  _ = a[j]! := by simp [f_add_zero]
                  _ = if j < n + 1 then a[j]! else 0 := by simp [hlt]
              · have hge : n ≤ j := Nat.le_of_not_gt hjn
                by_cases hEq : j = n
                · subst hEq
                  have hnot : ¬ j < j := by omega
                  have hlt : j < j + 1 := Nat.lt_succ_self j
                  simp [hnot, hlt, basisVecNat, coeffAt, hj,
                    Lean.Grind.Fin.mul_one]
                · have hnotlt : ¬ j < n + 1 := by omega
                  calc
                    (if j < n then a[j]! else 0) + a[n]! * coeffAt (basisVecNat n) j
                        = 0 + a[n]! * 0 := by
                            simp [hjn, hEq, basisVecNat, coeffAt, hj]
                    _ = 0 + 0 := by simp [f_mul_zero]
                    _ = 0 := by simp
                    _ = if j < n + 1 then a[j]! else 0 := by simp [hnotlt]

attribute [irreducible] basisExpandPrefix basisExpand

set_option maxRecDepth 4096 in
set_option maxHeartbeats 1200000 in
private theorem basisExpand_eq_of_size_d
    (a : Array F)
    (ha : a.size = d) :
    basisExpand a = a := by
  have hSize : (basisExpand a).size = d := by
    unfold basisExpand
    exact basisExpandPrefix_size a d
  apply Array.ext
  · exact hSize.trans ha.symm
  · intro j hjL hjR
    have hjExpand : j < (basisExpand a).size := by
      simpa [hSize] using hjL
    have hj : j < d := lt_of_lt_of_eq hjExpand hSize
    have hLGet : (basisExpand a)[j]'hjL = (basisExpand a).getD j 0 := by
      exact Array.getElem_eq_getD
        (xs := basisExpand a) (i := j) (h := hjL) (fallback := (0 : F))
    have hCoeff : coeffAt (basisExpand a) j = a[j]! := by
      have hCoeff0 := coeffAt_basisExpandPrefix_of_lt a j d hj
      rw [if_pos hj] at hCoeff0
      simpa [basisExpand] using hCoeff0
    have hRGet : a[j]'hjR = a.getD j 0 := by
      exact Array.getElem_eq_getD (xs := a) (i := j) (h := hjR) (fallback := (0 : F))
    have hRBang : a[j]! = a.getD j 0 := by
      simpa using (Array.getElem!_eq_getD (xs := a) (i := j))
    calc
      (basisExpand a)[j]'hjL = (basisExpand a).getD j 0 := hLGet
      _ = coeffAt (basisExpand a) j := by simp [basisExpand, coeffAt, hj]
      _ = a[j]! := hCoeff
      _ = a.getD j 0 := hRBang
      _ = a[j]'hjR := hRGet.symm

private theorem vecScale_zero_basis (i : Nat) :
    vecScale (0 : F) (basisVecNat i) = Array.replicate d (0 : F) := by
  apply Array.ext
  · simp [vecScale, basisVecNat, d]
  · intro j hjL hjR
    have hj : j < d := by simpa [vecScale, basisVecNat] using hjL
    have hCoeff : coeffAt (vecScale (0 : F) (basisVecNat i)) j = 0 := by
      rw [coeffAt_vecScale_of_size_d (s := (0 : F)) (x := basisVecNat i)
        (hx := by simp [basisVecNat]) (k := j) hj]
      simpa using (Lean.Grind.Fin.zero_mul (n := Goldilocks.q) (coeffAt (basisVecNat i) j))
    have hGet : (vecScale (0 : F) (basisVecNat i))[j]'hjL =
        coeffAt (vecScale (0 : F) (basisVecNat i)) j := by
      unfold coeffAt
      simp [hj, Array.getD]
    calc
      (vecScale (0 : F) (basisVecNat i))[j]'hjL = coeffAt (vecScale (0 : F) (basisVecNat i)) j := hGet
      _ = 0 := hCoeff
      _ = (Array.replicate d (0 : F))[j]'hjR := by simp

private theorem linear_eq_of_basis
    (K L : Array F → Array F)
    (hKSize : ∀ a, a.size = d → (K a).size = d)
    (hLSize : ∀ a, a.size = d → (L a).size = d)
    (hKAdd : ∀ x y, x.size = d → y.size = d → K (vecAdd x y) = vecAdd (K x) (K y))
    (hLAdd : ∀ x y, x.size = d → y.size = d → L (vecAdd x y) = vecAdd (L x) (L y))
    (hKScale : ∀ s x, x.size = d → K (vecScale s x) = vecScale s (K x))
    (hLScale : ∀ s x, x.size = d → L (vecScale s x) = vecScale s (L x))
    (hBasis : ∀ n, n < d → K (basisVecNat n) = L (basisVecNat n)) :
    ∀ a, a.size = d → K a = L a := by
  intro a ha
  have hPrefix : ∀ n, n ≤ d → K (basisExpandPrefix a n) = L (basisExpandPrefix a n) := by
    intro n hn
    induction n with
    | zero =>
        have h0d : 0 < d := by decide
        have hZeroK : K (basisExpandPrefix a 0) = K (vecScale (0 : F) (basisVecNat 0)) := by
          simp [basisExpandPrefix, vecScale_zero_basis]
        have hZeroL : L (basisExpandPrefix a 0) = L (vecScale (0 : F) (basisVecNat 0)) := by
          simp [basisExpandPrefix, vecScale_zero_basis]
        calc
          K (basisExpandPrefix a 0) = K (vecScale (0 : F) (basisVecNat 0)) := hZeroK
          _ = vecScale (0 : F) (K (basisVecNat 0)) := by
                exact hKScale (0 : F) (basisVecNat 0) (by simp [basisVecNat])
          _ = vecScale (0 : F) (L (basisVecNat 0)) := by
                simp [hBasis 0 h0d]
          _ = L (vecScale (0 : F) (basisVecNat 0)) := by
                symm
                exact hLScale (0 : F) (basisVecNat 0) (by simp [basisVecNat])
          _ = L (basisExpandPrefix a 0) := hZeroL.symm
    | succ n ih =>
        have hnlt : n < d := by omega
        have hPrevSize : (basisExpandPrefix a n).size = d := by simp [basisExpandPrefix_size]
        calc
          K (basisExpandPrefix a (n + 1))
              = K (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) := by
                  simp [basisExpandPrefix]
          _ = vecAdd (K (basisExpandPrefix a n)) (K (vecScale a[n]! (basisVecNat n))) := by
                exact hKAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)) hPrevSize (by simp [basisVecNat])
          _ = vecAdd (L (basisExpandPrefix a n)) (K (vecScale a[n]! (basisVecNat n))) := by
                simp [ih (by omega)]
          _ = vecAdd (L (basisExpandPrefix a n)) (vecScale a[n]! (K (basisVecNat n))) := by
                rw [hKScale]
                simp [basisVecNat]
          _ = vecAdd (L (basisExpandPrefix a n)) (vecScale a[n]! (L (basisVecNat n))) := by
                simp [hBasis n hnlt]
          _ = vecAdd (L (basisExpandPrefix a n)) (L (vecScale a[n]! (basisVecNat n))) := by
                rw [hLScale]
                simp [basisVecNat]
          _ = L (vecAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n))) := by
                exact (hLAdd (basisExpandPrefix a n) (vecScale a[n]! (basisVecNat n)) hPrevSize (by simp [basisVecNat])).symm
          _ = L (basisExpandPrefix a (n + 1)) := by
                simp [basisExpandPrefix]
  calc
    K a = K (basisExpand a) := by rw [basisExpand_eq_of_size_d a ha]
    _ = L (basisExpand a) := by
          simpa [basisExpand] using hPrefix d (Nat.le_refl d)
    _ = L a := by rw [basisExpand_eq_of_size_d a ha]

private theorem bilinear_eq_of_basis
    (K L : Array F → Array F → Array F)
    (hKSize : ∀ a b, a.size = d → b.size = d → (K a b).size = d)
    (hLSize : ∀ a b, a.size = d → b.size = d → (L a b).size = d)
    (hKAddLeft : ∀ x y b, x.size = d → y.size = d → b.size = d → K (vecAdd x y) b = vecAdd (K x b) (K y b))
    (hLAddLeft : ∀ x y b, x.size = d → y.size = d → b.size = d → L (vecAdd x y) b = vecAdd (L x b) (L y b))
    (hKScaleLeft : ∀ s x b, x.size = d → b.size = d → K (vecScale s x) b = vecScale s (K x b))
    (hLScaleLeft : ∀ s x b, x.size = d → b.size = d → L (vecScale s x) b = vecScale s (L x b))
    (hKAddRight : ∀ a x y, a.size = d → x.size = d → y.size = d → K a (vecAdd x y) = vecAdd (K a x) (K a y))
    (hLAddRight : ∀ a x y, a.size = d → x.size = d → y.size = d → L a (vecAdd x y) = vecAdd (L a x) (L a y))
    (hKScaleRight : ∀ s a x, a.size = d → x.size = d → K a (vecScale s x) = vecScale s (K a x))
    (hLScaleRight : ∀ s a x, a.size = d → x.size = d → L a (vecScale s x) = vecScale s (L a x))
    (hBasis : ∀ i j, i < d → j < d → K (basisVecNat i) (basisVecNat j) = L (basisVecNat i) (basisVecNat j)) :
    ∀ a b, a.size = d → b.size = d → K a b = L a b := by
  intro a b ha hb
  have hLeftBasis : ∀ i, i < d → K (basisVecNat i) b = L (basisVecNat i) b := by
    intro i hi
    exact linear_eq_of_basis
      (K := fun x => K (basisVecNat i) x)
      (L := fun x => L (basisVecNat i) x)
      (hKSize := fun x hx => hKSize (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hLSize := fun x hx => hLSize (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hKAdd := fun x y hx hy => hKAddRight (basisVecNat i) x y (by simp [basisVecNat]) hx hy)
      (hLAdd := fun x y hx hy => hLAddRight (basisVecNat i) x y (by simp [basisVecNat]) hx hy)
      (hKScale := fun s x hx => hKScaleRight s (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hLScale := fun s x hx => hLScaleRight s (basisVecNat i) x (by simp [basisVecNat]) hx)
      (hBasis := fun j hj => hBasis i j hi hj)
      b hb
  exact linear_eq_of_basis
    (K := fun x => K x b)
    (L := fun x => L x b)
    (hKSize := fun x hx => hKSize x b hx hb)
    (hLSize := fun x hx => hLSize x b hx hb)
    (hKAdd := fun x y hx hy => hKAddLeft x y b hx hy hb)
    (hLAdd := fun x y hx hy => hLAddLeft x y b hx hy hb)
    (hKScale := fun s x hx => hKScaleLeft s x b hx hb)
    (hLScale := fun s x hx => hLScaleLeft s x b hx hb)
    (hBasis := hLeftBasis)
    a ha

private def polyMulCoeffs (a b : Coeffs) : Coeffs :=
  polynomialToCoeffs ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial)

@[simp] private theorem polyMulCoeffs_size (a b : Coeffs) :
    (polyMulCoeffs a b).size = d := by
  simp [polyMulCoeffs, polynomialToCoeffs_size]

private theorem coeffsToPolynomial_polyMulCoeffs
    (a b : Coeffs) :
    coeffsToPolynomial (polyMulCoeffs a b) =
      ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial) := by
  simpa [polyMulCoeffs] using coeffsToPolynomial_polyMulMod a b

private theorem polyMulCoeffs_vecAdd_left
    {x y b : Coeffs}
    (hx : x.size = d)
    (hy : y.size = d)
    (hb : b.size = d) :
    polyMulCoeffs (vecAdd x y) b = vecAdd (polyMulCoeffs x b) (polyMulCoeffs y b) := by
  apply coeffsToPolynomial_injective_of_size_d
  · simp [polyMulCoeffs_size, vecAdd_size_of_eq, hx, hy]
  · simp [polyMulCoeffs_size, vecAdd_size_of_eq]
  · calc
      coeffsToPolynomial (polyMulCoeffs (vecAdd x y) b)
          = ((coeffsToPolynomial (vecAdd x y) * coeffsToPolynomial b) %ₘ phiPolynomial) := by
              simp [coeffsToPolynomial_polyMulCoeffs]
      _ = (((coeffsToPolynomial x + coeffsToPolynomial y) * coeffsToPolynomial b) %ₘ phiPolynomial) := by
              rw [coeffsToPolynomial_vecAdd_of_size_d hx hy]
      _ = (((coeffsToPolynomial x * coeffsToPolynomial b) +
            (coeffsToPolynomial y * coeffsToPolynomial b)) %ₘ phiPolynomial) := by
              rw [add_mul]
      _ = (((coeffsToPolynomial x * coeffsToPolynomial b) %ₘ phiPolynomial) +
            ((coeffsToPolynomial y * coeffsToPolynomial b) %ₘ phiPolynomial)) := by
              rw [Polynomial.add_modByMonic]
      _ = coeffsToPolynomial (vecAdd (polyMulCoeffs x b) (polyMulCoeffs y b)) := by
              rw [coeffsToPolynomial_vecAdd_of_size_d (by simp [polyMulCoeffs_size]) (by simp [polyMulCoeffs_size])]
              simp [coeffsToPolynomial_polyMulCoeffs]

private theorem polyMulCoeffs_vecAdd_right
    {a x y : Coeffs}
    (ha : a.size = d)
    (hx : x.size = d)
    (hy : y.size = d) :
    polyMulCoeffs a (vecAdd x y) = vecAdd (polyMulCoeffs a x) (polyMulCoeffs a y) := by
  apply coeffsToPolynomial_injective_of_size_d
  · simp [polyMulCoeffs_size, vecAdd_size_of_eq, hx, hy]
  · simp [polyMulCoeffs_size, vecAdd_size_of_eq]
  · calc
      coeffsToPolynomial (polyMulCoeffs a (vecAdd x y))
          = ((coeffsToPolynomial a * coeffsToPolynomial (vecAdd x y)) %ₘ phiPolynomial) := by
              simp [coeffsToPolynomial_polyMulCoeffs]
      _ = ((coeffsToPolynomial a * (coeffsToPolynomial x + coeffsToPolynomial y)) %ₘ phiPolynomial) := by
              rw [coeffsToPolynomial_vecAdd_of_size_d hx hy]
      _ = (((coeffsToPolynomial a * coeffsToPolynomial x) +
            (coeffsToPolynomial a * coeffsToPolynomial y)) %ₘ phiPolynomial) := by
              rw [mul_add]
      _ = (((coeffsToPolynomial a * coeffsToPolynomial x) %ₘ phiPolynomial) +
            ((coeffsToPolynomial a * coeffsToPolynomial y) %ₘ phiPolynomial)) := by
              rw [Polynomial.add_modByMonic]
      _ = coeffsToPolynomial (vecAdd (polyMulCoeffs a x) (polyMulCoeffs a y)) := by
              rw [coeffsToPolynomial_vecAdd_of_size_d (by simp [polyMulCoeffs_size]) (by simp [polyMulCoeffs_size])]
              simp [coeffsToPolynomial_polyMulCoeffs]

private theorem polyMulCoeffs_vecScale_left
    {s : F} {x b : Coeffs}
    (hx : x.size = d)
    (hb : b.size = d) :
    polyMulCoeffs (vecScale s x) b = vecScale s (polyMulCoeffs x b) := by
  apply coeffsToPolynomial_injective_of_size_d
  · simp [polyMulCoeffs_size]
  · simp [polyMulCoeffs_size, vecScale_size]
  · calc
      coeffsToPolynomial (polyMulCoeffs (vecScale s x) b)
          = ((coeffsToPolynomial (vecScale s x) * coeffsToPolynomial b) %ₘ phiPolynomial) := by
              simp [coeffsToPolynomial_polyMulCoeffs]
      _ = (((Polynomial.C (fToZMod s) * coeffsToPolynomial x) * coeffsToPolynomial b) %ₘ phiPolynomial) := by
              rw [coeffsToPolynomial_vecScale_of_size_d hx]
      _ = ((Polynomial.C (fToZMod s) * (coeffsToPolynomial x * coeffsToPolynomial b)) %ₘ phiPolynomial) := by
              ring
      _ = Polynomial.C (fToZMod s) * ((coeffsToPolynomial x * coeffsToPolynomial b) %ₘ phiPolynomial) := by
              rw [← smul_eq_C_mul, Polynomial.smul_modByMonic, smul_eq_C_mul]
      _ = coeffsToPolynomial (vecScale s (polyMulCoeffs x b)) := by
              rw [coeffsToPolynomial_vecScale_of_size_d (by simp [polyMulCoeffs_size])]
              simp [coeffsToPolynomial_polyMulCoeffs]

private theorem polyMulCoeffs_vecScale_right
    {s : F} {a x : Coeffs}
    (ha : a.size = d)
    (hx : x.size = d) :
    polyMulCoeffs a (vecScale s x) = vecScale s (polyMulCoeffs a x) := by
  apply coeffsToPolynomial_injective_of_size_d
  · simp [polyMulCoeffs_size]
  · simp [polyMulCoeffs_size, vecScale_size]
  · calc
      coeffsToPolynomial (polyMulCoeffs a (vecScale s x))
          = ((coeffsToPolynomial a * coeffsToPolynomial (vecScale s x)) %ₘ phiPolynomial) := by
              simp [coeffsToPolynomial_polyMulCoeffs]
      _ = ((coeffsToPolynomial a * (Polynomial.C (fToZMod s) * coeffsToPolynomial x)) %ₘ phiPolynomial) := by
              rw [coeffsToPolynomial_vecScale_of_size_d hx]
      _ = ((Polynomial.C (fToZMod s) * (coeffsToPolynomial a * coeffsToPolynomial x)) %ₘ phiPolynomial) := by
              ring
      _ = Polynomial.C (fToZMod s) * ((coeffsToPolynomial a * coeffsToPolynomial x) %ₘ phiPolynomial) := by
              rw [← smul_eq_C_mul, Polynomial.smul_modByMonic, smul_eq_C_mul]
      _ = coeffsToPolynomial (vecScale s (polyMulCoeffs a x)) := by
              rw [coeffsToPolynomial_vecScale_of_size_d (by simp [polyMulCoeffs_size])]
              simp [coeffsToPolynomial_polyMulCoeffs]

private theorem degree_X_pow_lt_d
    {n : Nat}
    (hn : n < d) :
    ((X ^ n : Polynomial Fq)).degree < d := by
  rw [Polynomial.degree_lt_iff_coeff_zero]
  intro m hm
  by_cases hEq : m = n
  · subst hEq
    omega
  · simp [Polynomial.coeff_X_pow, hEq]

private theorem degree_coeffsToPolynomial_lt_d
    (a : Coeffs) :
    (coeffsToPolynomial a).degree < d := by
  rw [Polynomial.degree_lt_iff_coeff_zero]
  intro m hm
  have hnot : ¬ m < d := by omega
  simp [coeff_coeffsToPolynomial, hnot]

private theorem coeffsToPolynomial_basisVecNat
    {i : Nat}
    (hi : i < d) :
    coeffsToPolynomial (basisVecNat i) = X ^ i := by
  ext n
  by_cases hn : n < d
  · by_cases hEq : n = i
    · subst hEq
      simp [coeff_coeffsToPolynomial, basisVecNat, coeffAt, hi]
    · simp [coeff_coeffsToPolynomial, basisVecNat, coeffAt, hn, hEq, Polynomial.coeff_X_pow]
  · have hNe : n ≠ i := by omega
    simp [coeff_coeffsToPolynomial, hn, hNe, Polynomial.coeff_X_pow]

private theorem coeffsToPolynomial_monomialReduce_of_lt_d
    {n : Nat}
    (hn : n < d) :
    coeffsToPolynomial (monomialReduce n) = X ^ n := by
  have h54 : n < 54 := by simpa [d] using hn
  have h81 : n < 81 := by omega
  unfold monomialReduce
  simp [Nat.mod_eq_of_lt h81, hn, coeffsToPolynomial_basisVecNat hn]

private theorem coeffsToPolynomial_monomialReduce_of_ge54_lt81
    {n : Nat}
    (h54 : 54 ≤ n)
    (h81 : n < 81) :
    coeffsToPolynomial (monomialReduce n) =
      -(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27) := by
  have hNotLt : ¬ n < d := by simpa [d] using (show ¬ n < 54 by omega)
  have h54lt : n - 54 < d := by simpa [d] using (show n - 54 < 54 by omega)
  have h27lt : n - 27 < d := by simpa [d] using (show n - 27 < 54 by omega)
  calc
    coeffsToPolynomial (monomialReduce n)
        = coeffsToPolynomial
            (vecAdd
              (vecScale (-1 : F) (basisVecNat (n - 54)))
              (vecScale (-1 : F) (basisVecNat (n - 27)))) := by
              simp [monomialReduce, Nat.mod_eq_of_lt h81, hNotLt]
    _ = coeffsToPolynomial (vecScale (-1 : F) (basisVecNat (n - 54))) +
          coeffsToPolynomial (vecScale (-1 : F) (basisVecNat (n - 27))) := by
            rw [coeffsToPolynomial_vecAdd_of_size_d]
            · simp [basisVecNat]
            · simp [basisVecNat]
    _ = -coeffsToPolynomial (basisVecNat (n - 54)) +
          -coeffsToPolynomial (basisVecNat (n - 27)) := by
            rw [coeffsToPolynomial_vecScale_of_size_d, coeffsToPolynomial_vecScale_of_size_d]
            · simp [basisVecNat, fToZMod_neg, fToZMod_one]
            · simp [basisVecNat]
            · simp [basisVecNat]
    _ = -(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27) := by
          rw [coeffsToPolynomial_basisVecNat h54lt, coeffsToPolynomial_basisVecNat h27lt]
          ring

private theorem coeffsToPolynomial_monomialReduce_of_ge81_lt108
    {n : Nat}
    (h81 : 81 ≤ n)
    (h108 : n < 2 * d) :
    coeffsToPolynomial (monomialReduce n) = X ^ (n - 81) := by
  have hn108 : n < 108 := by simpa [d] using h108
  have hSub : n - 81 < d := by simpa [d] using (show n - 81 < 54 by omega)
  have hLt81 : n - 81 < 81 := by omega
  have hmod : n % 81 = n - 81 := by
    calc
      n % 81 = (81 + (n - 81)) % 81 := by congr; omega
      _ = (n - 81) % 81 := by simp
      _ = n - 81 := Nat.mod_eq_of_lt hLt81
  unfold monomialReduce
  simp [hmod, hSub, coeffsToPolynomial_basisVecNat hSub]

private theorem X_pow_mod_phiPolynomial_of_lt_d
    {n : Nat}
    (hn : n < d) :
    ((X ^ n : Polynomial Fq) %ₘ phiPolynomial) = X ^ n := by
  exact (Polynomial.modByMonic_eq_self_iff phiPolynomial_monic).2 <|
    by simpa [phiPolynomial_degree] using degree_X_pow_lt_d hn

private theorem degree_neg_Xpow_sub_Xpow_lt_d
    {n : Nat}
    (h54 : 54 ≤ n)
    (h81 : n < 81) :
    (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)).degree < (d : WithBot Nat) := by
  rw [Polynomial.degree_lt_iff_coeff_zero]
  intro m hm
  have hmGe : d ≤ m := hm
  have h54lt : n - 54 < d := by simpa [d] using (show n - 54 < 54 by omega)
  have h27lt : n - 27 < d := by simpa [d] using (show n - 27 < 54 by omega)
  have hm1 : m ≠ n - 54 := by
    intro hEq
    subst hEq
    exact (Nat.not_lt_of_ge hmGe h54lt).elim
  have hm2 : m ≠ n - 27 := by
    intro hEq
    subst hEq
    exact (Nat.not_lt_of_ge hmGe h27lt).elim
  simp [Polynomial.coeff_sub, Polynomial.coeff_neg, Polynomial.coeff_X_pow, hm1, hm2]

  private theorem X_pow_mod_phiPolynomial_of_ge54_lt81
    {n : Nat}
    (h54 : 54 ≤ n)
    (h81 : n < 81) :
    ((X ^ n : Polynomial Fq) %ₘ phiPolynomial) =
      (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) := by
  have h1 : (X ^ 54 : Polynomial Fq) * X ^ (n - 54) = X ^ n := by
    rw [← pow_add]
    congr
    omega
  have h2 : (X ^ 27 : Polynomial Fq) * X ^ (n - 54) = X ^ (n - 27) := by
    rw [← pow_add]
    congr
    omega
  refine (Polynomial.div_modByMonic_unique (q := X ^ (n - 54))
    (r := (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)))
    phiPolynomial_monic ?_).2
  constructor
  · calc
      (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) + phiPolynomial * X ^ (n - 54)
          = (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) +
              ((X ^ 54 : Polynomial Fq) * X ^ (n - 54) +
                (X ^ 27 : Polynomial Fq) * X ^ (n - 54) +
                X ^ (n - 54)) := by
                  rw [phiPolynomial_def, add_mul, add_mul, one_mul]
      _ = (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) +
            (X ^ n + X ^ (n - 27) + X ^ (n - 54)) := by rw [h1, h2]
      _ = X ^ n := by ring_nf
  · exact by simpa [phiPolynomial_degree] using degree_neg_Xpow_sub_Xpow_lt_d h54 h81

  private theorem X_pow_mod_phiPolynomial_of_ge81_lt108
    {n : Nat}
    (h81 : 81 ≤ n)
    (h108 : n < 2 * d) :
    ((X ^ n : Polynomial Fq) %ₘ phiPolynomial) = X ^ (n - 81) := by
  have h54 : 54 ≤ n := by omega
  have h1 : (X ^ 54 : Polynomial Fq) * X ^ (n - 54) = X ^ n := by
    rw [← pow_add]
    congr
    omega
  have h2 : (X ^ 27 : Polynomial Fq) * X ^ (n - 54) = X ^ (n - 27) := by
    rw [← pow_add]
    congr
    omega
  have h3 : (X ^ 54 : Polynomial Fq) * X ^ (n - 81) = X ^ (n - 27) := by
    rw [← pow_add]
    congr
    omega
  have h4 : (X ^ 27 : Polynomial Fq) * X ^ (n - 81) = X ^ (n - 54) := by
    rw [← pow_add]
    congr
    omega
  refine (Polynomial.div_modByMonic_unique
    (q := X ^ (n - 54) - X ^ (n - 81))
    (r := X ^ (n - 81))
    phiPolynomial_monic ?_).2
  constructor
  · calc
      X ^ (n - 81) + phiPolynomial * (X ^ (n - 54) - X ^ (n - 81))
          = X ^ (n - 81) +
              (((X ^ 54 : Polynomial Fq) + X ^ 27 + 1) *
                (X ^ (n - 54) - X ^ (n - 81))) := by
                  rw [phiPolynomial_def]
      _ = X ^ (n - 81) +
            (((X ^ 54 : Polynomial Fq) * (X ^ (n - 54) - X ^ (n - 81))) +
              (X ^ 27 * (X ^ (n - 54) - X ^ (n - 81))) +
              (X ^ (n - 54) - X ^ (n - 81))) := by
                rw [add_mul, add_mul, one_mul]
      _ = X ^ (n - 81) +
            ((X ^ n - X ^ (n - 27)) + (X ^ (n - 27) - X ^ (n - 54)) +
              (X ^ (n - 54) - X ^ (n - 81))) := by
                rw [mul_sub, mul_sub, h1, h3, h2, h4]
      _ = X ^ n := by ring_nf
  · have hn108 : n < 108 := by simpa [d] using h108
    have hSub54 : n - 81 < 54 := by omega
    have hSub' : n - 81 < d := by simpa [d] using hSub54
    simpa [phiPolynomial_degree] using degree_X_pow_lt_d hSub'

set_option maxRecDepth 4096 in
private theorem coeffsToPolynomial_basis_basis_mod :
    ∀ i j : Fin d,
      ((coeffsToPolynomial (basisVecNat i.1) * coeffsToPolynomial (basisVecNat j.1)) %ₘ phiPolynomial) =
        coeffsToPolynomial (monomialReduce (i.1 + j.1)) := by
  intro i j
  let n := i.1 + j.1
  have hn108 : n < 2 * d := by
    dsimp [n]
    omega
  calc
    ((coeffsToPolynomial (basisVecNat i.1) * coeffsToPolynomial (basisVecNat j.1)) %ₘ phiPolynomial)
        = ((X ^ i.1 : Polynomial Fq) * X ^ j.1) %ₘ phiPolynomial := by
            rw [coeffsToPolynomial_basisVecNat i.2, coeffsToPolynomial_basisVecNat j.2]
    _ = (X ^ n : Polynomial Fq) %ₘ phiPolynomial := by
          dsimp [n]
          rw [← pow_add]
    _ = coeffsToPolynomial (monomialReduce n) := by
          by_cases hLt : n < d
          · rw [X_pow_mod_phiPolynomial_of_lt_d hLt, coeffsToPolynomial_monomialReduce_of_lt_d hLt]
          · by_cases hLt81 : n < 81
            · have h54 : 54 ≤ n := by simpa [d] using Nat.le_of_not_lt hLt
              rw [X_pow_mod_phiPolynomial_of_ge54_lt81 h54 hLt81,
                coeffsToPolynomial_monomialReduce_of_ge54_lt81 h54 hLt81]
            · have h81 : 81 ≤ n := by omega
              rw [X_pow_mod_phiPolynomial_of_ge81_lt108 h81 hn108,
                coeffsToPolynomial_monomialReduce_of_ge81_lt108 h81 hn108]

set_option maxHeartbeats 1200000 in
private theorem mulRq_basis_basis_eq_polyMulCoeffs
    (i j : Nat)
    (hi : i < d)
    (hj : j < d) :
    mulRq (basisVecNat i) (basisVecNat j) = polyMulCoeffs (basisVecNat i) (basisVecNat j) := by
  let ii : Fin d := ⟨i, hi⟩
  let jj : Fin d := ⟨j, hj⟩
  calc
    mulRq (basisVecNat i) (basisVecNat j)
        = monomialReduce (i + j) := by simpa [ii, jj] using mulRqPhi_basis_basis ii jj
    _ = polyMulCoeffs (basisVecNat i) (basisVecNat j) := by
          apply coeffsToPolynomial_injective_of_size_d
          · simp [monomialReduce_size]
          · simp [polyMulCoeffs_size]
          calc
            coeffsToPolynomial (monomialReduce (i + j))
                = ((coeffsToPolynomial (basisVecNat i) * coeffsToPolynomial (basisVecNat j)) %ₘ phiPolynomial) := by
                    symm
                    simpa [ii, jj] using coeffsToPolynomial_basis_basis_mod ii jj
            _ = coeffsToPolynomial (polyMulCoeffs (basisVecNat i) (basisVecNat j)) := by
                    symm
                    simp [coeffsToPolynomial_polyMulCoeffs]

set_option maxHeartbeats 4000000 in
private theorem mulRq_eq_polyMulCoeffs
    {a b : Coeffs}
    (ha : a.size = d)
    (hb : b.size = d) :
    mulRq a b = polyMulCoeffs a b := by
  exact bilinear_eq_of_basis
    (K := mulRq)
    (L := polyMulCoeffs)
    (hKSize := fun x y _ _ => by simp [mulRq_size])
    (hLSize := fun x y _ _ => by simp [polyMulCoeffs_size])
    (hKAddLeft := fun x y z hx hy _ => mulRqPhi_vecAdd_left_of_size_d x y z hx hy)
    (hLAddLeft := fun x y z hx hy hz => polyMulCoeffs_vecAdd_left hx hy hz)
    (hKScaleLeft := fun s x z hx _ => mulRqPhi_vecScale_left_of_size_d s x z hx)
    (hLScaleLeft := fun s x z hx hz => polyMulCoeffs_vecScale_left hx hz)
    (hKAddRight := fun z x y _ hx hy => mulRqPhi_vecAdd_right_of_size_d z x y hx hy)
    (hLAddRight := fun z x y hz hx hy => polyMulCoeffs_vecAdd_right hz hx hy)
    (hKScaleRight := fun s z x _ hx => mulRqPhi_vecScale_right_of_size_d s z x hx)
    (hLScaleRight := fun s z x hz hx => polyMulCoeffs_vecScale_right hz hx)
    (hBasis := fun i j hi hj => mulRq_basis_basis_eq_polyMulCoeffs i j hi hj)
    a b ha hb

private theorem coeffsToPolynomial_mulRq
    {a b : Coeffs}
    (ha : a.size = d)
    (hb : b.size = d) :
    coeffsToPolynomial (mulRq a b) =
      ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial) := by
  rw [mulRq_eq_polyMulCoeffs ha hb]
  exact coeffsToPolynomial_polyMulCoeffs a b

private theorem isCoprime_phiPolynomial_coeffsToPolynomial
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a ≤ 4)
    (hNe : a ≠ zeroRq) :
    IsCoprime phiPolynomial (coeffsToPolynomial a) := by
  have hA : IsCoprime phiA (coeffsToPolynomial a) := by
    exact (phiA_irreducible.coprime_iff_not_dvd).2
      (not_dvd_phiA_of_shape_norm_le_four_ne_zeroRq ha hNorm hNe)
  have hB : IsCoprime phiB (coeffsToPolynomial a) := by
    exact (phiB_irreducible.coprime_iff_not_dvd).2
      (not_dvd_phiB_of_shape_norm_le_four_ne_zeroRq ha hNorm hNe)
  simpa [phi_factor] using hA.mul_left hB

private theorem invertibleRq_of_isCoprime_coeffsToPolynomial
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hcop : IsCoprime (coeffsToPolynomial a) phiPolynomial) :
    invertibleRq a := by
  rcases hcop with ⟨u, v, huv⟩
  let ubar : Polynomial Fq := u %ₘ phiPolynomial
  refine ⟨polynomialToCoeffs ubar, ?_⟩
  apply coeffsToPolynomial_injective_of_size_d
  · simpa [hasRingDegreeShape] using ha
  · exact oneRq_size
  · calc
      coeffsToPolynomial (mulRq a (polynomialToCoeffs ubar))
          = ((coeffsToPolynomial a * coeffsToPolynomial (polynomialToCoeffs ubar)) %ₘ phiPolynomial) := by
              exact coeffsToPolynomial_mulRq (by simpa [hasRingDegreeShape] using ha) (by simp [polynomialToCoeffs_size])
      _ = ((coeffsToPolynomial a * ubar) %ₘ phiPolynomial) := by
            have hubarDeg : ubar.degree < d := by
              simpa [ubar, phiPolynomial_degree] using Polynomial.degree_modByMonic_lt u phiPolynomial_monic
            simpa [ubar] using congrArg
              (fun p => ((coeffsToPolynomial a * p) %ₘ phiPolynomial))
              (coeffsToPolynomial_polynomialToCoeffs_of_degree_lt hubarDeg)
      _ = ((coeffsToPolynomial a * (u %ₘ phiPolynomial)) %ₘ phiPolynomial) := by
            simp [ubar]
      _ = ((coeffsToPolynomial a * u) %ₘ phiPolynomial) := by
            have hpmod : coeffsToPolynomial a %ₘ phiPolynomial = coeffsToPolynomial a := by
              exact (Polynomial.modByMonic_eq_self_iff phiPolynomial_monic).2 <|
                by simpa [phiPolynomial_degree] using degree_coeffsToPolynomial_lt_d a
            have hubarMod : (u %ₘ phiPolynomial) %ₘ phiPolynomial = u %ₘ phiPolynomial := by
              exact (Polynomial.modByMonic_eq_self_iff phiPolynomial_monic).2 <|
                by simpa [phiPolynomial_degree] using Polynomial.degree_modByMonic_lt u phiPolynomial_monic
            rw [Polynomial.mul_modByMonic (p₁ := coeffsToPolynomial a) (p₂ := u) (q := phiPolynomial)]
            simp [hpmod, hubarMod]
      _ = 1 := by
            have huv' : coeffsToPolynomial a * u + v * phiPolynomial = 1 := by
              simpa [mul_comm] using huv
            have hrepr : (1 : Polynomial Fq) + phiPolynomial * (-v) = coeffsToPolynomial a * u := by
              calc
                (1 : Polynomial Fq) + phiPolynomial * (-v)
                    = (coeffsToPolynomial a * u + v * phiPolynomial) + phiPolynomial * (-v) := by
                        rw [huv']
                _ = coeffsToPolynomial a * u := by ring_nf
            have hdeg : ((1 : Polynomial Fq).degree) < phiPolynomial.degree := by
              simpa [phiPolynomial_degree, d]
            exact (Polynomial.div_modByMonic_unique
              (q := -v)
              (r := (1 : Polynomial Fq))
              phiPolynomial_monic
              ⟨hrepr, hdeg⟩).2
      _ = coeffsToPolynomial oneRq := by simp

theorem invertibleRq_of_shape_norm_le_four_ne_zeroRq
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a ≤ 4)
    (hNe : a ≠ zeroRq) :
    invertibleRq a := by
  exact invertibleRq_of_isCoprime_coeffsToPolynomial ha
    (isCoprime_phiPolynomial_coeffsToPolynomial ha hNorm hNe).symm

theorem paperCarrierDiffInvertibilityAssumption_goldilocks :
    paperCarrierDiffInvertibilityAssumption := by
  intro δ hDiff hNe
  rcases samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four hDiff with
    ⟨hShape, hNorm⟩
  exact invertibleRq_of_shape_norm_le_four_ne_zeroRq hShape hNorm hNe

end

end SuperNeo
