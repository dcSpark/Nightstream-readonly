# How SuperNeo Works

SuperNeo is Neo without the SIMD restriction.
Neo packs *d separate* field vectors into one ring vector (so the same constraint
system must apply to all d lanes). SuperNeo instead packs a **single** field
vector of length d*n into a ring vector of length n. This is the version we (plan to) use.

This document assumes familiarity with [how-neo-works.md](./how-neo-works.md).

---

## 1. The Six Desiderata

To our knowledge, SuperNeo is the first folding scheme to satisfy all six:

| Property | What it means | Who fails it |
|----------|---------------|-------------|
| D1: Post-quantum security | Secure against quantum adversaries | HyperNova, NeutronNova |
| D2: Pay-per-bit costs | Commitment cost scales with witness bit-width | LatticeFold, Arc |
| D3: Field-native arithmetic | Sumcheck and norm checks run over fields (not rings) | LatticeFold |
| D4: General constraints | Supports non-SIMD CCS over one witness vector | LatticeFold, **Neo** |
| D5: Small-field support | Works with Goldilocks, M61 | HyperNova, LatticeFold |
| D6: Low recursion overheads | Small verifier circuit for IVC | Arc, LatticeFold |

Neo satisfies D1–D3 and D5–D6 but fails D4. SuperNeo fixes D4.

---

## 2. The Core Problem: Embedding Field Vectors into Ring Vectors

Ajtai commitments live over ring vectors, but CCS constraints live over field
vectors. The embedding must:

- **Preserve norms** so low-norm field vectors map to low-norm ring vectors
  (pay-per-bit, and Ajtai binding remains meaningful).
- **Respect evaluation + linearity** so random linear combinations of committed
  ring vectors correspond to the same combinations of the underlying field
  evaluations (the key algebra needed for folding).
- **Enable field-native checks** so sumcheck runs over a field (possibly an
  extension), not over the ring.

Prior work (LatticeFold) used the NTT embedding, which breaks all three: it does
not preserve norms well, it forces ring-based sumcheck, and it requires SIMD.

---

## 3. Two Embeddings

### 3.1 The Neo Embedding (SIMD)

Pack d field vectors z^(1), ..., z^(d) in F^n into one ring vector z_R in R_F^n
by using coefficient slots. For each position i in [n]:

```
(z_R)_i = sum_{j=1}^{d} (z^(j))_i * X^{j-1}   in R_F
```

Equivalently, the coefficient matrix of z_R is the vertical stack of field
vectors:

```
Coeff(z_R) = [ z^(1) ; z^(2) ; ... ; z^(d) ]    (d x n matrix)
```

**Norm preservation:** trivial, since the field entries are literally the
coefficients.

**Evaluation homomorphism:** multiplying a ring element by a constant c in K
scales every coefficient by c. So the MLE satisfies:

```
(z_R)_tilde(r) = sum_{j=1}^{d} (z^(j))_tilde(r) * X^{j-1}
```

The d field evaluations appear as the d coefficients of a single ring element.
Taking ring linear combinations preserves this lane structure.

**Limitation:** you get d vectors of length n, not one vector of length d*n.
All lanes must satisfy the *same* constraint system (SIMD).

### 3.2 The SuperNeo Embedding (General)

Pack one field vector z in F^{d*n} into a ring vector z_R in R_F^n by chunking z
into n blocks of length d and using each block as the coefficients of one ring
element:

```
z = [z_1, z_2, ..., z_n],   z_i in F^d
(z_R)_i = sum_{j=1}^{d} (z_i)_j * X^{j-1}   in R_F
```

**Norm preservation:** same as Neo — coefficients are unchanged.

**Evaluation homomorphism:** the subtlety is multiplication. For a CCS matrix
M in F^{m x d*n}, folding needs evaluation claims like

```
y = (M z)_tilde(r),
```

where M z is an m-length field vector and the tilde is its MLE evaluation.
A naive coefficient embedding does not make ring products behave like field
inner products (coefficients mix via the cyclotomic relation), so we cannot
just "pack" and hope constant terms line up.

SuperNeo uses the **inner product transform**: there exists a linear transform
`bar(.) : F^d -> F^d` such that for all a, b in F^d,

```
ct( bar(a) * bar(b) ) = <a, b>
```

where `bar(a)` means "apply the transform to a and then embed into the ring",
and `ct(.)` extracts the constant term of a ring element.

Lift `bar(.)` blockwise to vectors/matrices, so a field matrix M becomes a ring
matrix `bar(M)` (same shape, but each d-block is transformed before embedding).
Then the matrix-vector product over the field is recovered as constant terms of
a ring matrix-vector product (taken rowwise):

```
M z = ct( bar(M) * z_R ).
```

Evaluating at r: the constant term of `(bar(M) * z_R)_tilde(r)` in R_K equals
`(M z)_tilde(r)`. Since evaluation and ring linear combinations are linear over
R_K, combining instances preserves these claims.

**Key advantage:** one witness vector of length d*n, one CCS, no SIMD. Packing is
optimal: d*n field elements into n ring elements.

---

## 4. The Relations (Updated for SuperNeo)

SuperNeo uses the same two relations as Neo but with the SuperNeo embedding and
the inner product transform. The witness is a single field vector z in F^{n_F}
(where n_F = d*n_R), packed/committed as a ring vector z_R in R_F^{n_R}.

### CCS(b, L) — Norm-Bounded CCS

```
CCS(b, L) = {
    (s; (c, x); w) :
        z = [x, w],  c = L(z),  ||z||_inf < b,
        f(M1*z_tilde, ..., Mt*z_tilde) vanishes on {0,1}^{log m}
}
```

### CE(b, L) — Norm-Bounded CCS Evaluation

```
CE(b, L) = {
    (c, x, r, {y_j in R_K}; z) :
        c = L(z),  x = L_in(z),  ||z||_inf < b,
        for all j: y_j = ( bar(M_j) * z_R )_tilde(r)
}
```

Note: each y_j is a ring element in R_K. The field evaluation `(M_j z)_tilde(r)`
is recovered as `ct(y_j)`.

---

## 5. The Three Reductions (Same Pipeline, Cleaner Security)

The pipeline is structurally identical to Neo:

```
CE(b,L)^k  x  CCS(b,L)^K
    |  Pi_CCS  (sumcheck: CCS + norm checks + re-randomize evaluations)
    v
CE(b,L)^{K+k}
    |  Pi_RLC  (ring linear combination, norm grows)
    v
CE(B,L)
    |  Pi_DEC  (split back to low-norm)
    v
CE(b,L)^k
```

What changes from Neo is only (i) how evaluation claims are represented
(ring elements + ct(.)), and (ii) how security is proven (interactive
reductions instead of ad-hoc composition).

### Pi_CCS

Constructs a polynomial Q encoding CCS constraints, norm checks, and
re-randomization of existing evaluation claims, and runs sumcheck over the
field extension K. Ring equalities like
`y_j = (bar(M_j)*z_R)_tilde(r)` are enforced coefficient-wise by extracting
`cf(y_j)_ell` for ell in [d], so the sumcheck itself remains field-native.

### Pi_RLC

Same structure as Neo: sample challenges rho_i from a strong sampling set C,
then take ring linear combinations of commitments, witnesses, and evaluations.
Evaluation homomorphism ensures the combined claim still satisfies
`y_j = (bar(M_j)*z_R)_tilde(r)`.

### Pi_DEC

Same as Neo: split_b(z) -> (z_1, ..., z_k) with ||z_i||_inf < b.
Verifier checks `c =? sum b^{i-1} * c_i` and `y_j =? sum b^{i-1} * y_{i,j}`.

---

## 6. Interactive Reductions (The Security Framework)

SuperNeo replaces Neo's ad-hoc knowledge-soundness variants with a modular
framework: **strong** and **weak** interactive reductions.

### The Problem

In the lattice setting, extracting from Pi_RLC yields openings that satisfy the
linear relations (e.g., `c = A z`, `y = (bar(M)*z_R)_tilde(r)`), but may have
*arbitrary norm*: the extractor must multiply by `(delta_1 - delta_2)^{-1}`,
whose norm can be huge even when delta_i are small. So Pi_RLC cannot, by itself,
be a standard reduction of knowledge.

The key observation is a uniqueness property: under MSIS hardness, there is
(essentially) only one opening per commitment (otherwise relaxed binding breaks).
This is what lets us compose the pieces cleanly.

### Strong Interactive Reductions

An interactive reduction Pi: R1 -> R2 is **strong** (w.r.t. phi) if:

1. **phi-restricted:** specified instance components (commitments) are always
   preserved in the output.
2. **Restricted extraction:** if the adversary is forced to always output the
   *same* (relaxed) witness for those components, an extractor can recover a
   valid witness for R1.

Pi_CCS is strong: sumcheck + norm checks work because the adversary cannot
"switch openings" for a fixed commitment.

### Weak Interactive Reductions

An interactive reduction Pi: R1 -> R2 is **weak** (w.r.t. the same phi) if:

1. **Relaxed extraction:** the extractor outputs witnesses for a relaxed
   relation R'1 (dropping the norm bound).
2. **Unique witness:** on inputs with the same phi-image (same commitments),
   the extractor can output at most one witness.

Pi_RLC is weak: it extracts arbitrary-norm openings, but only one per commitment.

### The Composition Theorem

**Theorem (Strong-Weak Composition):** If Pi_1 is strong w.r.t. phi and Pi_2 is
weak w.r.t. the same phi, then Pi_2 o Pi_1 is a standard reduction of knowledge.

Thus Pi_RLC o Pi_CCS is a RoK. Composing with Pi_DEC (directly a RoK) yields the
full folding scheme as a RoK.

### Why This Matters

This pattern is common in lattice protocols: an information-theoretic stage
(sumcheck, random projections) followed by a special-sound stage (RLC with
extraction). Strong/weak reductions make the analysis modular and reusable.
