## Twist and Shout Integration to Neo (Summary)

This summary outlines the strategy for implementing the Twist and Shout (T&S) memory-checking arguments within the Neo codebase. It is intended for review and validation by researchers familiar with folding schemes, zkVMs, and the relevant literature.

**Terminology upfront (to avoid overload):**

- *Folding step* / *folding chunk*: the unit processed by one iteration of the folding loop (one call to Π_CCS/Π_Twist/Π_Shout). A chunk can be one VM step or a user-defined batch of VM steps.
- *Shard* (in the code): a trace segment represented as a **collection** of folding chunks. Each chunk is a `StepWitnessBundle` containing one MCS plus the matching Twist/Shout instances for that same chunk. The folding loop advances chunk-by-chunk through the shard; there are no shard-wide Twist/Shout instances, only per-chunk ones.

### Summary: Integrating Twist and Shout into the Neo Folding Scheme (with index-based addressing + SPARK-style adapter)

#### 1. Context and Problem Statement

Neo is a folding scheme framework utilizing lattice-based commitments for Customizable Constraint Systems (CCS). Twist (for R/W memory) and Shout (for R/O memory/lookups) are highly efficient memory arguments based purely on sum-check, avoiding grand-product arguments.

The objective is to integrate T&S into Neo to leverage significant prover speedups. While Neo and T&S are compatible at the level of required primitives (sum-check + sparse-friendly commitments), the key implementation challenge is deriving T&S inputs from the Neo execution trace and committing them in a way that keeps **commitment keys and commitment costs small** in practice.

A concrete concern is that naïvely committing to one-hot address representations can make the commitment "ambient dimension" / key sizes large, despite the vectors being sparse. Concretely, one-hot addressing commits scale with the number of committed columns (\Theta(d\cdot m)), whereas index-bit addressing scales as (\Theta(d\cdot \ell)) where (\ell=\lceil\log_2 m\rceil).

We highly appreciate the support and guidance from Srinath Setty on how to do this implementation. This document puts into writting what we understood a this discussion about this topic with him.

#### 2. Architectural Foundations

**Neo's Folding Mechanism:**
Neo folds instances of the Matrix Constraint System relation into a running set of Matrix Evaluation relations, following:

1. (\Pi_{\mathsf{CCS}}): reduces a fresh CCS/MCS claim (and existing (\mathsf{ME}) claims) to a new set of (\mathsf{ME}(b,\mathcal{L})) claims via sum-check.
2. (\Pi_{\mathsf{RLC}}): combines all (\mathsf{ME}) claims into a single aggregated (\mathsf{ME}) claim via random linear combination.
3. (\Pi_{\mathsf{DEC}}): decomposes the aggregated claim back down to the base norm bound (b).

**Neo's Lattice Commitments:**
Neo uses a low-norm matrix commitment scheme with "pay-per-bit" behavior, making it well-suited for sparse/structured witness columns *when the committed representation itself is compact*.

**Twist and Shout:**
T&S rely on:

* Addressing-based constraints (often described with one-hot structure),
* Sparse increments (Twist),
* Virtual polynomials (read values, full memory state) accessed via sum-check rather than committed.

Twist models R/W memory via a recurrence: the memory state at step i equals the previous state plus a sparse increment, Val_{i+1} = Val_i + Inc_i. The full memory state vector is never committed; only the sparse increment columns are committed, and Val is computed virtually during sum-check as the prefix-sum of increments. This "rollover" semantic means that each chunk's memory state carries forward to the next; the builder advances `init_vals` between folding chunks so the recurrence spans the whole execution.

#### 3. The Integration Strategy: Unified Folding Interface

The core integration remains unchanged:
[
\Pi_{\mathsf{Shout}} : \mathsf{SHO}(b,\mathcal{L})
\rightsquigarrow \mathsf{ME}(b,\mathcal{L})^{t_{\mathsf{sho}}}
\qquad
\Pi_{\mathsf{Twist}} : \mathsf{TWI}(b,\mathcal{L})
\rightsquigarrow \mathsf{ME}(b,\mathcal{L})^{t_{\mathsf{twi}}}.
]

We instantiate Twist and Shout **per folding chunk**, not once for the whole shard. Each chunk carries its own Twist/Shout instance (per memory/table id), built from the slice of the trace that belongs to that chunk, with `init_vals` rolled forward from the previous chunk.

At the level of the folding interface, all ME claims (from CCS and from T&S) are treated uniformly. At a folding step we aggregate:

1. (k) running (\mathsf{ME}) instances,
2. fresh (\mathsf{ME}) instances from (\Pi_{\mathsf{CCS}}),
3. fresh (\mathsf{ME}) instances from (\Pi_{\mathsf{Twist}}) and (\Pi_{\mathsf{Shout}}) for this chunk (including their committed columns like addr bits, flags/values, Inc, and any adapter-derived openings),

and then fold the full batch using the standard Neo pipeline:
[
(\text{all ME claims}) \xrightarrow{\Pi_{\mathsf{RLC}}} \mathsf{ME}^{\text{agg}}
\xrightarrow{\Pi_{\mathsf{DEC}}} \mathsf{ME}(b,\mathcal{L}).
]

**Instantiation choice (current implementation – updated).** We now run Twist and Shout *per folding step*, over per-step witnesses that slice the VM trace at that folding chunk. A “step” here means a folding chunk chosen by the arithmetization/front-end; it can be one VM step or a user-defined batch of VM steps. For each step (i):

* Π_{\mathsf{CCS}} runs on MCS\_i and the current accumulator,
* Π_{\mathsf{Twist}} runs on the step-i memory slice (with rolled-forward `init_vals`),
* Π_{\mathsf{Shout}} runs on the step-i lookup slice,
* all resulting ME claims (running + CCS + Twist + Shout + any IDX→OH) are batched into a single Π_{\mathsf{RLC}} → Π_{\mathsf{DEC}} to produce the next k children.

There is no shard-wide “memory sidecar merge”; Twist/Shout participate in the same per-step folding loop as CCS.

**Per-step folding shape (current code):**

Step i:
  - k running ME
  - ME from Π_{\mathsf{CCS}}(chunk i)
  - ME from Π_{\mathsf{Twist}}(chunk i)
  - ME from Π_{\mathsf{Shout}}(chunk i)
  → Π_{\mathsf{RLC}} → Π_{\mathsf{DEC}} → k children

**Terminology note.** A shard is a trace segment modeled as a collection of folding chunks. Each `StepWitnessBundle` holds exactly one MCS (CPU chunk) plus the Twist/Shout instances for that *same folding chunk*. A folding chunk can be one VM step or a user-chosen grouping of VM steps. The folding loop consumes one chunk at a time from the shard. There are no shard-wide Twist/Shout instances—only per-chunk ones.

No modification to (\Pi_{\mathsf{RLC}}) or (\Pi_{\mathsf{DEC}}) is required.

#### 4. Trace Generation and Commitments (Updated Address Handling)

1. **Execution and Witness:**
   Prover executes the VM and generates the CCS witness (z) (as before).

2. **Deriving T&S Inputs:**
   * Twist: derive sparse increment trace (\widetilde{\text{Inc}}) from writes (as before).
   * Shout: derive lookup/RO trace columns (as before).
   * **Addresses (updated): do *not* materialize or commit one-hot vectors.**
     For each access (i), represent each address dimension (j\in[d]) by an index (\textsf{idx}^{(j)}_i\in[m)), where (m \approx K^{1/d}). Commit to a low-norm representation of (\textsf{idx}^{(j)}_i), e.g. its bit-decomposition:
     [
     \textsf{idx}^{(j)}_i = \sum_{k=0}^{\ell-1} 2^k \cdot b^{(j,k)}_i,\quad b^{(j,k)}_i \in \{0,1\},\ \ell=\lceil \log_2 m \rceil.
     ]
     (If (m) is not a power of two, enforce (\textsf{idx}^{(j)}_i<m) or pad to the next power of two and treat extra addresses as unused.)

3. **Committing the Inputs:**
   Commit to (\widetilde{\text{Inc}}) and other sparse T&S inputs using Neo's lattice commitment scheme, and commit to address **bit-columns** (b^{(j,k)}), rather than one-hot address columns.

4. **Virtualization (unchanged):**
   Read values (\widetilde{\text{rv}}) and full memory state (\widetilde{\text{Val}}) remain virtual and are only accessed through sum-check evaluations.

#### 5. New Component: Index→OneHot Adapter ("SPARK-style" bridge)

Although we do not commit one-hot address vectors, Twist/Shout use the *one-hot interface* (equivalently, MLE evaluations of conceptual one-hot objects). We introduce an adapter protocol that implements a **virtual one-hot oracle** defined by committed index bits.

**Conceptual object (not committed):** For each dimension (j), define the one-hot matrix (A^{(j)}):
[
A^{(j)}[i,y] = 1 \iff y = \textsf{idx}^{(j)}_i.
]

**Evaluation identity:** For verifier-chosen random points (r) (over the step index) and (u) (over the address index bits),
[
\widetilde{A}^{(j)}(r,u)
= \sum_{i\in\{0,1\}^{\log T}} \chi_i(r)\cdot \chi_{\textsf{idx}^{(j)}_i}(u),
]
with
[
\chi_{\textsf{idx}}(u)=\prod_{k=0}^{\ell-1}\Big(b_k u_k + (1-b_k)(1-u_k)\Big),
]
where (b_k) are the committed bit-columns for (\textsf{idx}) at row (i).

**Adapter protocol:** Define (\Pi_{\mathsf{IDX2OH}}) that proves:

* bitness of (b^{(j,k)}_i) (or relies on CCS to enforce bitness),
* and that the claimed (\widetilde{A}^{(j)}(r,u)) equals the sum/product expression above,
  reducing to a small batch of (\mathsf{ME}(b,\mathcal{L})) claims:
  [
  \Pi_{\mathsf{IDX2OH}}:\ \mathsf{IDX2OH}(b,\mathcal L) \rightsquigarrow \mathsf{ME}(b,\mathcal L)^{t_{\mathsf{idx}}}.
  ]

**Challenge alignment:** The adapter must be proven at the same ((r,u)) points that Twist/Shout use when querying the conceptual one-hot oracle. This can be achieved by sampling ((r,u)) once (Fiat–Shamir) and reusing it across the relevant sub-protocols, or by embedding (\Pi_{\mathsf{IDX2OH}}) as a sub-protocol inside Twist/Shout queries.

**Trade-off:** This adds extra sum-check work (foldable, prover-side), in exchange for substantially smaller committed representations: one-hot uses (\Theta(d\cdot m)) committed columns; index-bits use (\Theta(d\cdot \ell)).

#### 6. Examples (Toy + Scaling)

##### Example A: 4 accesses into 8 memory cells (one-hot vs index bits)

Memory size (m=8). Four accesses with indices:
[
\textsf{idx} = [5,\ 0,\ 5,\ 3].
]

**Conceptual one-hot address matrix** (A \in {0,1}^{4\times 8}) (not committed):

* row0 (addr 5): ([0,0,0,0,0,1,0,0])
* row1 (addr 0): ([1,0,0,0,0,0,0,0])
* row2 (addr 5): ([0,0,0,0,0,1,0,0])
* row3 (addr 3): ([0,0,0,1,0,0,0,0])

A naïve approach commits all 8 indicator columns (A_0,\dots,A_7).

**Updated approach:** Commit only (\ell=\log_2 8 = 3) bit-columns:

* 5 = 101, 0 = 000, 5 = 101, 3 = 011 gives:
  * (b_0 = [1,0,1,1])
  * (b_1 = [0,0,0,1])
  * (b_2 = [1,0,1,0])

When Twist/Shout need an MLE opening of the conceptual one-hot matrix at ((r,u)), the prover supplies the value computed from the identity in §5 and proves correctness with (\Pi_{\mathsf{IDX2OH}}), without committing (A).

##### Example B: Scaling intuition

Take memory size (K=2^{32}) and choose (d=4). Then per-dimension one-hot length is:
[
m = K^{1/d} = 2^8 = 256
\quad \Rightarrow \quad \ell=\log_2 m = 8.
]

Per access:

* One-hot committed columns per dimension: (m=256) (so (d\cdot m=1024)).
* Index-bit committed columns per dimension: (\ell=8) (so (d\cdot \ell=32)).

This is a **32× reduction** in committed address width, shifting work from commitments/keys to a foldable sum-check adapter.

#### 7. Conclusion

Twist and Shout integrate cleanly into Neo's folding architecture because they reduce to the same intermediate relation (\mathsf{ME}(b,\mathcal L)). To control commitment-key size and commitment costs:

* Do **not** commit to conceptual one-hot address vectors.
* Commit to compact index representations (bit-/digit-decomposed, low-norm).
* Add a **foldable Index→OneHot adapter** ("SPARK-style bridge") proving that MLE evaluations of the conceptual one-hot objects used by Twist/Shout are consistent with the committed indices. In the intended shape, adapter-derived openings (and Inc for Twist) are just additional ME claims emitted by Π_{\mathsf{Twist}}/Π_{\mathsf{Shout}} and folded alongside their other columns—no separate adapter batch is required.

In our instantiation, each folding chunk has its own Twist and Shout instances (per memory/table id), scoped to that chunk’s trace slice. Π_{\mathsf{Twist}} and Π_{\mathsf{Shout}} run once per chunk to produce ME claims, and these memory ME claims are folded together with CCS in that step’s (\Pi_{\mathsf{RLC}}\rightarrow\Pi_{\mathsf{DEC}}) round. All resulting (\mathsf{ME}) claims—CCS, Twist, Shout, and the adapter—are folded using Neo’s existing (\Pi_{\mathsf{RLC}}\rightarrow\Pi_{\mathsf{DEC}}) loop without modifying the core folding logic.
