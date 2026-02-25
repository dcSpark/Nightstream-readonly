# How Neo Works

Neo is a lattice-based folding scheme for CCS (Customizable Constraint Systems).
It can be understood as "HyperNova rebuilt on lattices instead of elliptic curves,"
giving two benefits: native support for small prime fields (Goldilocks, M61) and
plausible post-quantum security.

---

## 1. The Three Core Ingredients

Neo's folding scheme is built around three core ingredients:

**1. A commitment scheme that is linear/homomorphic in the right way, but whose
binding depends on a norm bound.**

Neo uses an Ajtai-style commitment, but *not* to arbitrary vectors directly.
Instead it commits to **low-norm matrices** produced by a **digit decomposition**
`Decomp_b(z)`, and it views the commitment as an **S-module homomorphism**
(a "matrix commitment scheme"). This gives you:

- efficient "pay-per-bit" committing (because digits are small),
- linear homomorphism needed for folding, and
- binding that stays meaningful because the committed object is norm-bounded.

**2. A target relation for openings/evaluations that folding operates on:
ME (Matrix Evaluation) claims.**

Neo defines ME claims of the form: you have a commitment `c = L(Z)` and you
want to assert **linear transforms** of the opening matrix evaluated at a random
point, written as `y_j = Z M_j^T r_hat` (their "partial evaluation" form).

**3. A three-step reduction pipeline that maintains soundness while controlling
norms:**

- `Pi_CCS`: uses **sumcheck** to reduce a CCS/MCS satisfaction claim to a set
  of ME claims.
- `Pi_RLC`: uses **random linear combinations** (with challenges from a
  **strong sampling set**) to collapse many ME claims into one larger-norm
  ME claim.
- `Pi_DEC`: **decomposes/splits** the large-norm claim back into many low-norm
  claims so binding stays within bounds.

That's "how Neo works." Everything must be expressible as "produce ME claims,
fold them with RLC, then DEC to keep norms bounded."

---

## 2. The Commitment Scheme

### Digit Decomposition

Given a witness vector z in F^m, Neo maps it to a low-norm matrix:

```
Z = Decomp_b(z) in F^{d x m}
```

Each element z_i is split into its base-b digits, producing d rows. The key
property: `||Z||_inf < b`. To reconstruct:
`z = sum_{i=1}^{d} b^{i-1} * Z^{(i)}`.

Each column of Z maps to one ring element via `cf^{-1}`, giving a vector
z' in R_q^m committed with Ajtai: `c = M * z'`.

### S-Module Homomorphism

The commitment L: F^{d x m} -> C is an S-module homomorphism, where
S = {rot(a) | a in R_q} is the ring of rotation matrices. This means:

```
rho_1 * L(Z_1) + rho_2 * L(Z_2) = L(rho_1 * Z_1 + rho_2 * Z_2)
```

This linearity is what makes folding possible: weighted sums of commitments
commit to the weighted sum of openings.

### Partial Evaluations

For a CCS constraint matrix M and evaluation point r:

```
y = Z * M^T * r_hat
```

This d-dimensional vector y is a "partial evaluation." The full evaluation
is recovered as `Mz_tilde(r) = sum b^{i-1} * y_i`. These partial evaluations
are the objects Neo's folding scheme manipulates.

---

## 3. The Two Relations

### MCS (Matrix Constraint System)

```
MCS(b, L) = {
    (s; (c, x); w) :
        Z = Decomp_b(x || w),  c = L(Z),
        f(M1*z_tilde, ..., Mt*z_tilde) in ZS_n
}
```

"I have a committed low-norm matrix whose underlying vector satisfies the
CCS constraints."

### ME (Matrix Evaluation)

```
ME(b, L) = {
    (c, X, r, {y_j}; Z) :
        c = L(Z),  X = L_x(Z),  ||Z||_inf < b,
        for all j: y_j = Z * M_j^T * r_hat
}
```

"I have a committed low-norm matrix whose partial evaluations at point r
are the claimed values y_j."

The folding scheme reduces MCS claims (constraint satisfaction) to ME claims
(evaluation assertions), then folds ME claims together.

---

## 4. The Three Reductions

### Pi_CCS: CCS Reduction (Sumcheck)

**Type:** ME(b, L)^k x MCS(b, L) -> ME(b, L)^{k+1}

Constructs a polynomial Q over {0,1}^{log(d*n)} encoding CCS constraints (F),
norm checks (NC_i), and re-randomization of existing evaluation claims (Eval_{i,j}).
Runs sumcheck to reduce everything to evaluations at a fresh random point r'.
The prover sends new partial evaluations at r', and the verifier checks
consistency.

Sumcheck runs over an extension of a small prime field (e.g., F_{q^2}), not
over polynomial rings.

### Pi_RLC: Random Linear Combination

**Type:** ME(b, L)^{k+1} -> ME(B, L)   where B = b^k

Verifier samples challenges rho_i from a **strong sampling set** C (subset of S).
Both parties compute combined commitment `c = sum rho_i * c_i` and evaluations
`y_j = sum rho_i * y_{i,j}`. Prover computes combined opening `Z = sum rho_i * Z_i`.

By S-homomorphism, `c = L(Z)` and `y_j = Z * M_j^T * r_hat` hold automatically.

**The catch:** `||Z||_inf` can grow up to B = b^k. The norm is no longer within
the low-norm bound b.

### Pi_DEC: Decomposition

**Type:** ME(B, L) -> ME(b, L)^k

Prover decomposes Z via `(Z_1, ..., Z_k) = split_b(Z)` where each `||Z_i||_inf < b`.
Sends fresh commitments and evaluations for each piece. Verifier checks
`c =? sum b^{i-1} * c_i` and `y_j =? sum b^{i-1} * y_{i,j}`.

No verifier randomness needed.

### The Composed Folding Scheme

```
ME(b,L)^k  x  MCS(b,L)
    |  Pi_CCS  (sumcheck: reduce CCS + re-randomize evaluations)
    v
ME(b,L)^{k+1}
    |  Pi_RLC  (combine into one claim, norm grows)
    v
ME(B,L)
    |  Pi_DEC  (split back to low-norm)
    v
ME(b,L)^k
```

Input and output both carry k ME claims with norm bound b, so the scheme folds
indefinitely: each new CCS step feeds into the running accumulator.

---

## 5. Norm Management and Security

### Why Norms Matter

Unlike Pedersen/KZG (where binding is unconditional on the committed value),
Ajtai's binding depends on the committed vector having norm below a bound.
If norms grow past the bound, the commitment is trivially breakable.

Every linear combination (Pi_RLC) inflates norms. Every decomposition (Pi_DEC)
brings them back down. The parameters must ensure that after a full fold cycle
the norm stays within the binding regime.

### Strong Sampling Sets

RLC challenges must come from a **strong sampling set** C: a subset of S where
all pairwise differences are invertible in R_q. This is needed for extraction
in the security proof. The expansion factor T controls norm inflation per
challenge multiplication. The constraint: `(k+1) * T * (b-1) < B = b^k`.

### Security Composition

Neither Pi_CCS nor Pi_RLC is individually a standard reduction of knowledge.
Neo introduces relaxed security notions (phi-restricted, phi-relaxed knowledge
soundness, restricted knowledge soundness) and a Composition Theorem proving
that Pi_RLC o Pi_CCS composes into a standard RoK. Pi_DEC is directly a
standard RoK. The full composition Pi_DEC o Pi_RLC o Pi_CCS is therefore a
sound folding scheme.
