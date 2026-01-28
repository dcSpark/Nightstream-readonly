/* tslint:disable */
/* eslint-disable */

/**
 * Opaque proof handle returned by `NeoFoldSession::fold_and_prove()`.
 */
export class NeoFoldProof {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    fold_step_ms(): any;
    folding_summary(): any;
    proof_estimate(): any;
    step_count(): number;
}

/**
 * Stateful JS-facing session wrapper.
 *
 * Construct once from a circuit JSON, then:
 * - `add_step_*` incrementally
 * - `fold_and_prove()` to obtain an opaque proof handle
 * - `verify(proof)` to check it
 */
export class NeoFoldSession {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Add one step from `(x, w)` encoded as JSON arrays of u64s.
     */
    add_step_io_json(x_json: string, w_json: string): void;
    /**
     * Add one step from a witness vector `z` encoded as JSON array of u64s.
     */
    add_step_witness_json(witness_json: string): void;
    /**
     * Add all witness steps from a full `TestExport` JSON (uses only `witness`).
     */
    add_steps_from_test_export_json(json: string): void;
    circuit_summary(): any;
    fold_and_prove(): NeoFoldProof;
    /**
     * Create a new session from a circuit JSON (same fields as `TestExport` but without `witness`).
     *
     * Note: serde ignores unknown fields, so passing a full `TestExport` JSON is also accepted.
     */
    constructor(circuit_json: string);
    params_summary(): any;
    /**
     * Set verifier-side step-linking equality pairs from JSON.
     *
     * Format: `[[prev_idx, next_idx], ...]` (must be non-empty for multi-step verification).
     */
    set_step_linking_pairs_json(json: string): void;
    setup_timings_ms(): any;
    /**
     * Compress a folding proof into a Spartan2 proof (Merkle-MLE engine).
     */
    spartan_prove(proof: NeoFoldProof): SpartanCompressedProof;
    spartan_verify(proof: SpartanCompressedProof): boolean;
    step_count(): number;
    verify(proof: NeoFoldProof): boolean;
}

/**
 * Opaque Spartan proof handle returned by `NeoFoldSession::spartan_prove()`.
 */
export class SpartanCompressedProof {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Downloadable Spartan proof bytes (SNARK proof only; excludes `vk`).
     */
    bytes(): Uint8Array;
    /**
     * Size of the downloadable artifact (SNARK proof only; excludes `vk`).
     */
    bytes_len(): number;
    /**
     * Total bytes if you bundled `(vk, snark)` together.
     */
    vk_and_snark_bytes_len(): number;
    /**
     * Verifier key size (useful for understanding total proof package size).
     */
    vk_bytes_len(): number;
}

export function init_panic_hook(): void;

/**
 * Parse a `TestExport` JSON (same schema as `crates/neo-fold/poseidon2-tests/*.json`),
 * then run prove+verify and return a small result object.
 */
export function prove_verify_test_export_json(json: string): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly init_panic_hook: () => void;
    readonly prove_verify_test_export_json: (a: number, b: number) => [number, number, number];
    readonly __wbg_neofoldsession_free: (a: number, b: number) => void;
    readonly neofoldsession_new: (a: number, b: number) => [number, number, number];
    readonly neofoldsession_step_count: (a: number) => number;
    readonly neofoldsession_setup_timings_ms: (a: number) => [number, number, number];
    readonly neofoldsession_params_summary: (a: number) => [number, number, number];
    readonly neofoldsession_circuit_summary: (a: number) => [number, number, number];
    readonly neofoldsession_add_step_witness_json: (a: number, b: number, c: number) => [number, number];
    readonly neofoldsession_add_step_io_json: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly neofoldsession_add_steps_from_test_export_json: (a: number, b: number, c: number) => [number, number];
    readonly neofoldsession_set_step_linking_pairs_json: (a: number, b: number, c: number) => [number, number];
    readonly neofoldsession_fold_and_prove: (a: number) => [number, number, number];
    readonly neofoldsession_verify: (a: number, b: number) => [number, number, number];
    readonly neofoldsession_spartan_prove: (a: number, b: number) => [number, number, number];
    readonly neofoldsession_spartan_verify: (a: number, b: number) => [number, number, number];
    readonly __wbg_neofoldproof_free: (a: number, b: number) => void;
    readonly neofoldproof_step_count: (a: number) => number;
    readonly neofoldproof_fold_step_ms: (a: number) => [number, number, number];
    readonly neofoldproof_proof_estimate: (a: number) => [number, number, number];
    readonly neofoldproof_folding_summary: (a: number) => [number, number, number];
    readonly __wbg_spartancompressedproof_free: (a: number, b: number) => void;
    readonly spartancompressedproof_bytes_len: (a: number) => [number, number, number];
    readonly spartancompressedproof_bytes: (a: number) => [number, number, number, number];
    readonly spartancompressedproof_vk_bytes_len: (a: number) => [number, number, number];
    readonly spartancompressedproof_vk_and_snark_bytes_len: (a: number) => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
