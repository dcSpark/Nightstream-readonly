use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use neo_fold::shard::{ShardProof, StepProof};
use neo_math::K;
use neo_memory::riscv::ccs::TraceShoutBusSpec;
use neo_memory::riscv::lookups::{
    encode_program, BranchCondition, RiscvInstruction, RiscvMemOp, RiscvOpcode, RiscvShoutTables, PROG_ID, RAM_ID,
    REG_ID,
};
use neo_memory::riscv::trace::{
    rv32_decode_lookup_backed_row_from_instr_word, rv32_decode_lookup_table_id_for_col,
    rv32_decode_lookup_transport_cols, rv32_decode_lookup_val_slot_for_col, rv32_is_decode_lookup_table_id,
    rv32_trace_lookup_n_vals_for_table_id, rv32_width_lookup_backed_cols, rv32_width_lookup_table_id_for_col,
    rv32_width_lookup_val_slot_for_col, rv32_width_sidecar_witness_from_exec_table, Rv32DecodeSidecarLayout,
    Rv32WidthSidecarLayout,
};
use p3_field::PrimeCharacteristicRing;

fn materialized_steps<'a>(proof: &'a ShardProof) -> Vec<&'a StepProof> {
    let mut out = Vec::new();
    for step in &proof.steps {
        if let Some(sub) = step.compressed_substeps.as_ref() {
            out.extend(sub.iter());
        }
        out.push(step);
    }
    out
}

fn first_materialized_step(proof: &ShardProof) -> &StepProof {
    materialized_steps(proof)
        .into_iter()
        .next()
        .expect("expected at least one materialized proof step")
}

fn first_materialized_step_mut(proof: &mut ShardProof) -> &mut StepProof {
    let step0 = proof
        .steps
        .first_mut()
        .expect("expected at least one proof step");
    if step0
        .compressed_substeps
        .as_ref()
        .is_some_and(|sub| !sub.is_empty())
    {
        return step0
            .compressed_substeps
            .as_mut()
            .and_then(|sub| sub.first_mut())
            .expect("expected at least one compressed materialized proof step");
    }
    step0
}

#[test]
fn rv32_trace_wiring_runner_prove_verify() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    run.verify().expect("trace wiring verify");

    assert_eq!(run.fold_count(), 1, "trace runner should produce one folding step");
    assert_eq!(run.trace_len(), 3, "active trace length mismatch");
    assert_eq!(
        run.exec_table().rows.len(),
        3,
        "exec rows remain unpadded; power-of-two padding applies to proving/layout length"
    );
    assert!(
        run.layout().t >= run.exec_table().rows.len(),
        "layout.t should cover exec rows (layout.t={}, exec_rows={})",
        run.layout().t,
        run.exec_table().rows.len()
    );
    assert_eq!(run.layout().t, 3, "layout.t should reflect unpadded proving length");

    let steps_public = run.steps_public();
    assert_eq!(steps_public.len(), 1, "trace runner should expose one step instance");
    let mut mem_ids: Vec<u32> = steps_public[0]
        .mem_insts
        .iter()
        .map(|inst| inst.mem_id)
        .collect();
    mem_ids.sort_unstable();
    let mut expected_mem_ids = vec![PROG_ID.0, RAM_ID.0, REG_ID.0];
    expected_mem_ids.retain(|&id| id != RAM_ID.0);
    expected_mem_ids.sort_unstable();
    assert_eq!(
        mem_ids, expected_mem_ids,
        "trace runner should default to used-sidecar instantiation (no RAM sidecar when unused)"
    );
    assert_eq!(
        run.used_memory_ids(),
        expected_mem_ids.as_slice(),
        "run artifact should record auto-derived S_memory"
    );
    let add_table_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0;
    let used_lookup_ids = run.used_shout_table_ids();
    assert!(
        used_lookup_ids.contains(&add_table_id),
        "run artifact should include opcode-backed S_lookup tables"
    );
    let decode_lookup_count = used_lookup_ids
        .iter()
        .copied()
        .filter(|table_id| rv32_is_decode_lookup_table_id(*table_id))
        .count();
    let expected_decode_lookup_tables = rv32_decode_lookup_transport_cols(&Rv32DecodeSidecarLayout::new())
        .into_iter()
        .map(rv32_decode_lookup_table_id_for_col)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(
        decode_lookup_count, expected_decode_lookup_tables,
        "run artifact should include decode lookup table family IDs in S_lookup"
    );
}

#[test]
fn rv32_trace_wiring_runner_reg_output_binding_prove_verify() {
    // Program: ADDI x2, x0, 3; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 3,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 2, /*expected=*/ neo_math::F::from_u64(3))
        .prove()
        .expect("trace wiring prove with reg output binding");

    run.verify()
        .expect("trace wiring verify with reg output binding");
}

#[test]
fn rv32_trace_wiring_runner_allows_without_insecure_ack() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring should no longer require insecure benchmark-only ack");
    run.verify()
        .expect("trace wiring proof should verify without insecure benchmark-only ack");
}

#[test]
fn rv32_trace_wiring_runner_prove_verify_without_insecure_ack() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring should prove without insecure benchmark-only ack");

    run.verify()
        .expect("trace wiring proof should verify without insecure benchmark-only ack");
}

#[test]
fn rv32_trace_wiring_runner_shared_bus_default_has_expected_layout() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let run_shared = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    assert_eq!(
        run_shared.ccs_num_variables(),
        run_shared.layout().m,
        "shared-bus trace layout width must match CCS width"
    );
}

#[test]
fn rv32_trace_wiring_runner_decode_lookup_table_content_matches_slots() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    let steps_public = run.steps_public();
    let step = steps_public
        .first()
        .expect("trace runner should expose one step instance");
    let exec_row = run
        .exec_table()
        .rows
        .iter()
        .find(|r| r.active)
        .expect("expected one active exec row");
    let key_idx = usize::try_from(exec_row.pc_before).expect("pc_before should fit usize");

    let decode_layout = Rv32DecodeSidecarLayout::new();
    let decode_cols = rv32_decode_lookup_transport_cols(&decode_layout);
    let decode_row =
        rv32_decode_lookup_backed_row_from_instr_word(&decode_layout, exec_row.instr_word, /*active=*/ true);
    for &col_id in decode_cols.iter() {
        let table_id = rv32_decode_lookup_table_id_for_col(col_id);
        let inst = step
            .lut_insts
            .iter()
            .find(|inst| inst.table_id == table_id)
            .unwrap_or_else(|| panic!("missing decode lookup table_id={table_id} in step instance"));
        let n_vals = rv32_trace_lookup_n_vals_for_table_id(table_id).max(1);
        assert_eq!(
            inst.table.len(),
            inst.k * n_vals,
            "decode lookup table content length must be k*n_vals (table_id={table_id}, k={}, n_vals={n_vals})",
            inst.k
        );
        assert!(
            key_idx < inst.k,
            "decode key index out of range: key_idx={key_idx}, k={}, table_id={table_id}",
            inst.k
        );
        let val_slot = if n_vals == 1 {
            0usize
        } else {
            rv32_decode_lookup_val_slot_for_col(col_id)
                .unwrap_or_else(|| panic!("missing decode val slot for col_id={col_id}"))
        };
        let idx = key_idx
            .checked_mul(n_vals)
            .and_then(|base| base.checked_add(val_slot))
            .expect("decode lookup content index overflow");
        assert!(
            idx < inst.table.len(),
            "decode lookup content index out of range: idx={idx}, len={}, table_id={table_id}, n_vals={n_vals}, val_slot={val_slot}",
            inst.table.len()
        );
        assert_eq!(
            inst.table[idx], decode_row[col_id],
            "decode lookup content mismatch at table_id={table_id}, key_idx={key_idx}, val_slot={val_slot}, col_id={col_id}"
        );
    }
}

#[test]
fn rv32_trace_wiring_runner_width_lookup_table_content_matches_slots() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x1234,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 0,
            rs2: 1,
            imm: 0x100,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 2,
            rs1: 0,
            imm: 0x100,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(1)
        .prove()
        .expect("trace wiring prove");

    let steps_public = run.steps_public();
    let step = steps_public
        .first()
        .expect("trace runner should expose one step instance");
    let width_layout = Rv32WidthSidecarLayout::new();
    let width_cols = rv32_width_lookup_backed_cols(&width_layout);
    let width_wit = rv32_width_sidecar_witness_from_exec_table(&width_layout, run.exec_table());

    for (row_idx, row) in run
        .exec_table()
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.active)
    {
        let key_idx = usize::try_from(row.cycle).expect("cycle should fit usize");
        for &col_id in width_cols.iter() {
            let table_id = rv32_width_lookup_table_id_for_col(col_id);
            let inst = step
                .lut_insts
                .iter()
                .find(|inst| inst.table_id == table_id)
                .unwrap_or_else(|| panic!("missing width lookup table_id={table_id} in step instance"));
            let n_vals = rv32_trace_lookup_n_vals_for_table_id(table_id).max(1);
            assert_eq!(
                inst.table.len(),
                inst.k * n_vals,
                "width lookup table content length must be k*n_vals (table_id={table_id}, k={}, n_vals={n_vals})",
                inst.k
            );
            assert!(
                key_idx < inst.k,
                "width key index out of range: key_idx={key_idx}, k={}, table_id={table_id}",
                inst.k
            );
            let val_slot = if n_vals == 1 {
                0usize
            } else {
                rv32_width_lookup_val_slot_for_col(col_id)
                    .unwrap_or_else(|| panic!("missing width val slot for col_id={col_id}"))
            };
            let idx = key_idx
                .checked_mul(n_vals)
                .and_then(|base| base.checked_add(val_slot))
                .expect("width lookup content index overflow");
            assert!(
                idx < inst.table.len(),
                "width lookup content index out of range: idx={idx}, len={}, table_id={table_id}, n_vals={n_vals}, val_slot={val_slot}",
                inst.table.len()
            );
            assert_eq!(
                inst.table[idx], width_wit.cols[col_id][row_idx],
                "width lookup content mismatch at table_id={table_id}, cycle={key_idx}, val_slot={val_slot}, col_id={col_id}, row_idx={row_idx}"
            );
        }
    }
}

#[test]
fn rv32_trace_wiring_runner_shout_override_must_superset_inferred_set() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .shout_ops([RiscvOpcode::Xor])
        .prove()
    {
        Ok(_) => panic!("shout override that misses required tables must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("superset") && msg.contains("Add"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_extra_shout_spec_without_table_spec() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .extra_shout_bus_specs([TraceShoutBusSpec {
            table_id: 1000,
            ell_addr: 13,
            n_vals: 1usize,
        }])
        .prove()
    {
        Ok(_) => panic!("extra shout geometry without table spec must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("extra_shout_bus_specs includes table_id=1000 without a table spec"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_accepts_extra_shout_spec_with_matching_table_spec() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .extra_lut_table_spec(1000, neo_memory::witness::LutTableSpec::IdentityU32)
        .extra_shout_bus_specs([TraceShoutBusSpec {
            table_id: 1000,
            ell_addr: 32,
            n_vals: 1usize,
        }])
        .prove()
        .expect("trace wiring prove with extra table/spec");
    run.verify()
        .expect("trace wiring verify with extra table/spec");

    assert!(
        run.used_shout_table_ids().contains(&1000),
        "run should record extra table_id in used shout set"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_extra_table_spec_colliding_with_opcode_table() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let add_table_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Add).0;
    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .extra_lut_table_spec(add_table_id, neo_memory::witness::LutTableSpec::IdentityU32)
        .prove()
    {
        Ok(_) => panic!("extra_lut_table_spec collision with inferred opcode table must fail"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("extra_lut_table_spec collides with existing table_id"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_max_steps_above_trace_cap() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .max_steps((1usize << 20) + 1)
        .prove()
    {
        Ok(_) => panic!("max_steps above trace cap must be rejected"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("max_steps=") && msg.contains("trace-mode hard cap"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_min_trace_len_above_trace_cap() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len((1usize << 20) + 1)
        .prove()
    {
        Ok(_) => panic!("min_trace_len above trace cap must be rejected"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(
        msg.contains("min_trace_len=") && msg.contains("trace-mode hard cap"),
        "unexpected error message: {msg}"
    );
}

#[test]
fn rv32_trace_wiring_runner_chunked_ivc_step_linking() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with chunked ivc");

    run.verify().expect("trace wiring verify with chunked ivc");

    assert_eq!(
        run.fold_count(),
        2,
        "chunk_rows=2 over 3 rows should produce two fold steps"
    );
    let steps = run.steps_public();
    assert_eq!(steps.len(), 2, "expected two public steps");

    let layout = run.layout();
    let prev = &steps[0].mcs_inst.x;
    let cur = &steps[1].mcs_inst.x;
    assert_eq!(
        prev[layout.pc_final], cur[layout.pc0],
        "trace step linking must enforce pc_final -> pc0 across steps"
    );
}

#[test]
fn rv32_trace_wiring_runner_chunking_avoids_virtual_split_boundaries() {
    // Program: ADDI x1, x0, 1; MULH x3, x1, x2; HALT
    // With decomposition enabled, MULH expands to a virtual run.
    // chunk_rows=2 would normally map to step_rows=8, which can split at row boundary 8
    // (inside the MULH virtual run) unless the runner expands chunk size.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Mulh,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with decomposition-aware chunk sizing");

    run.verify()
        .expect("trace wiring verify with decomposition-aware chunk sizing");

    assert_eq!(
        run.fold_count(),
        1,
        "chunk sizing should auto-expand to avoid cutting through virtual decomposition transitions"
    );
}

#[test]
fn rv32_trace_wiring_runner_chunked_ivc_batches_no_shared_val_lanes_per_mem() {
    // Program: ADDI x1, x0, 1; ADDI x2, x1, 2; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 1,
            imm: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(2)
        .prove()
        .expect("trace wiring prove with chunked ivc");
    run.verify().expect("trace wiring verify with chunked ivc");

    let steps_public = run.steps_public();
    let shard_proof = run.proof();
    assert_eq!(steps_public.len(), 2, "expected two public steps");
    let materialized = materialized_steps(shard_proof);
    assert_eq!(materialized.len(), 2, "expected two materialized proof steps");

    // Step 0 (shared-bus): one current CPU val claim.
    let proof_step0 = materialized[0];
    assert_eq!(
        proof_step0.mem.val_me_claims.len(),
        1,
        "step0(shared) must emit one current CPU val claim"
    );
    assert_eq!(
        proof_step0.val_fold.len(),
        1,
        "step0(shared) must emit one val-fold proof"
    );

    // Step 1 (shared-bus): val claims are [current_cpu, previous_cpu], each with its own fold proof.
    let proof_step1 = materialized[1];
    assert_eq!(
        proof_step1.mem.val_me_claims.len(),
        2,
        "step1(shared) must emit current+previous CPU val claims"
    );
    assert_eq!(
        proof_step1.val_fold.len(),
        2,
        "step1(shared) must emit one val-fold proof per claim"
    );
}

#[test]
fn rv32_trace_wiring_runner_wb_wp_folds_are_emitted_and_required() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    let step0 = first_materialized_step(&proof);
    assert!(
        !step0.mem.wb_me_claims.is_empty(),
        "expected WB ME claims for RV32 trace route-A"
    );
    assert!(
        !step0.mem.wp_me_claims.is_empty(),
        "expected WP ME claims for RV32 trace route-A"
    );
    assert!(
        !step0.wb_fold.is_empty(),
        "expected wb_fold proofs for RV32 trace route-A"
    );
    assert!(
        !step0.wp_fold.is_empty(),
        "expected wp_fold proofs for RV32 trace route-A"
    );

    let mut proof_missing_wb = proof.clone();
    first_materialized_step_mut(&mut proof_missing_wb)
        .wb_fold
        .clear();
    assert!(
        run.verify_proof(&proof_missing_wb).is_err(),
        "missing wb_fold must fail verification"
    );

    let mut proof_missing_wp = proof.clone();
    first_materialized_step_mut(&mut proof_missing_wp)
        .wp_fold
        .clear();
    assert!(
        run.verify_proof(&proof_missing_wp).is_err(),
        "missing wp_fold must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_decode_openings_are_embedded_in_wp_and_required() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    let step0 = first_materialized_step(&proof);
    assert_eq!(step0.mem.wp_me_claims.len(), 1, "expected one WP ME claim");
    let mut proof_missing_decode_me = proof.clone();
    let decode_layout = Rv32DecodeSidecarLayout::new();
    let target_col = decode_layout.op_alu_imm;
    let wp_point = step0.mem.wp_me_claims[0].r.clone();
    let step_mut = first_materialized_step_mut(&mut proof_missing_decode_me);
    let wp_open_idx = step_mut
        .fold
        .openings
        .iter()
        .find(|opening| opening.point == wp_point && opening.col_ids.iter().any(|&c| c == target_col))
        .or_else(|| {
            step_mut
                .fold
                .openings
                .iter()
                .find(|opening| opening.point == wp_point)
        })
        .and_then(|opening| {
            step_mut
                .fold
                .openings
                .iter()
                .position(|cand| cand.point == opening.point && cand.col_ids == opening.col_ids)
        })
        .expect("decode openings must be present in WP named openings");
    let wp_open = &mut step_mut.fold.openings[wp_open_idx];
    assert!(!wp_open.evals.is_empty(), "WP named opening evals must be non-empty");
    let decode_idx = wp_open
        .col_ids
        .iter()
        .position(|&c| c == target_col)
        .unwrap_or(0);
    wp_open.evals[decode_idx] += K::ONE;
    assert!(
        run.verify_proof(&proof_missing_decode_me).is_err(),
        "tampered decode lookup opening embedded in WP ME must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_width_openings_on_wp_are_required() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    let step0 = first_materialized_step(&proof);
    assert_eq!(step0.mem.wp_me_claims.len(), 1, "expected one WP ME claim");

    let mut proof_tampered_width_open = proof.clone();
    let width_layout = Rv32WidthSidecarLayout::new();
    let target_col = width_layout.rs2_low_bit[0];
    let wp_point = step0.mem.wp_me_claims[0].r.clone();
    let step_mut = first_materialized_step_mut(&mut proof_tampered_width_open);
    let wp_open_idx = step_mut
        .fold
        .openings
        .iter()
        .find(|opening| opening.point == wp_point && opening.col_ids.iter().any(|&c| c == target_col))
        .or_else(|| {
            step_mut
                .fold
                .openings
                .iter()
                .find(|opening| opening.point == wp_point)
        })
        .and_then(|opening| {
            step_mut
                .fold
                .openings
                .iter()
                .position(|cand| cand.point == opening.point && cand.col_ids == opening.col_ids)
        })
        .expect("width openings must be present in WP named openings");
    let wp_open = &mut step_mut.fold.openings[wp_open_idx];
    assert!(!wp_open.evals.is_empty(), "WP named opening evals must be non-empty");
    let width_idx = wp_open
        .col_ids
        .iter()
        .position(|&c| c == target_col)
        .unwrap_or(0);
    wp_open.evals[width_idx] += K::ONE;
    assert!(
        run.verify_proof(&proof_tampered_width_open).is_err(),
        "tampered width lookup opening embedded in WP ME must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_control_claims_are_emitted_and_required() {
    // Program: ADDI x1, x0, 1; HALT
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    let step0 = first_materialized_step(&proof);

    let labels = &step0.batched_time.labels;
    let find_w4 = |label: &'static [u8]| -> usize {
        labels
            .iter()
            .position(|l| *l == label)
            .expect("missing required control stage claim label in batched_time")
    };
    let control_linear_idx = find_w4(b"control/next_pc_linear");
    let control_control_idx = find_w4(b"control/next_pc_control");
    let control_branch_idx = find_w4(b"control/branch_semantics");
    let _control_writeback_idx = find_w4(b"control/writeback");
    assert!(
        control_linear_idx < labels.len() && control_control_idx < labels.len() && control_branch_idx < labels.len(),
        "control stage labels must be present in batched_time"
    );

    let mut proof_missing_control_claim = proof.clone();
    let step0_missing_control = first_materialized_step_mut(&mut proof_missing_control_claim);
    let _ = step0_missing_control
        .batched_time
        .claimed_sums
        .remove(control_control_idx);
    let _ = step0_missing_control
        .batched_time
        .degree_bounds
        .remove(control_control_idx);
    let _ = step0_missing_control
        .batched_time
        .labels
        .remove(control_control_idx);
    let _ = step0_missing_control
        .batched_time
        .round_polys
        .remove(control_control_idx);
    assert!(
        run.verify_proof(&proof_missing_control_claim).is_err(),
        "missing control/next_pc_control claim artifact must fail verification"
    );

    let mut proof_tampered_control_round = proof.clone();
    let coeff = first_materialized_step_mut(&mut proof_tampered_control_round)
        .batched_time
        .round_polys[control_control_idx]
        .get_mut(0)
        .and_then(|round| round.get_mut(0))
        .expect("control/next_pc_control first-round coeff must exist");
    *coeff += K::ONE;
    assert!(
        run.verify_proof(&proof_tampered_control_round).is_err(),
        "tampered control/next_pc_control round polynomial must fail verification"
    );
}

#[test]
fn rv32_trace_wiring_runner_rejects_zero_chunk_rows() {
    let program = vec![RiscvInstruction::Halt];
    let program_bytes = encode_program(&program);

    let err = match Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .chunk_rows(0)
        .prove()
    {
        Ok(_) => panic!("chunk_rows=0 must be rejected"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(msg.contains("chunk_rows"), "unexpected error message: {msg}");
}

#[test]
fn rv32_trace_wiring_runner_rejects_amo_via_wb_decode_scope_lock() {
    // Program includes one AMO row. In Tier 2.1 trace mode this is rejected by WB/decode stage
    // decode residuals (scope lock), not by the N0 main-trace CCS.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::Amo {
            op: RiscvMemOp::AmoaddW,
            rd: 2,
            rs1: 0,
            rs2: 1,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    assert!(
        Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
            .prove()
            .is_err(),
        "AMO must be rejected in Tier 2.1 trace mode via WB/decode stage scope lock"
    );
}

fn prove_verify_trace_program(program: Vec<RiscvInstruction>) {
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");
}

#[test]
fn rv32_trace_wiring_runner_accepts_wrapped_load_store_addressing() {
    // Build base near 2^32 so effective address wraps:
    // rs1 = 0xffff_ff10, imm = 0x140 => addr = 0x50 (mod 2^32).
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: -240,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 13,
            rs1: 0,
            imm: 42,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sw,
            rs1: 2,
            rs2: 13,
            imm: 320,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lw,
            rd: 14,
            rs1: 2,
            imm: 320,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 14, /*expected=*/ neo_math::F::from_u64(42))
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove with wrapped load/store address");
    run.verify()
        .expect("trace wiring verify with wrapped load/store address");
}

#[test]
fn rv32_trace_wiring_runner_accepts_load_to_x0_without_writeback() {
    // Regression for W3 load-semantics gating: loads to x0 are legal and must not
    // require rd writeback while still enforcing RAM-read semantics.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 0x100,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 0x7f,
        },
        RiscvInstruction::Store {
            op: RiscvMemOp::Sb,
            rs1: 1,
            rs2: 2,
            imm: 0,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lbu,
            rd: 0,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Load {
            op: RiscvMemOp::Lbu,
            rd: 3,
            rs1: 1,
            imm: 0,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 3, /*expected=*/ neo_math::F::from_u64(0x7f))
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove with load-to-x0");
    run.verify().expect("trace wiring verify with load-to-x0");
}

#[test]
fn rv32_trace_wiring_runner_accepts_auipc_wraparound_writeback() {
    // Regression for RV32 modular writeback semantics:
    // place AUIPC at pc=0x1000, use imm_u=0xffff_f000 (-1 << 12), so
    // rd = (0x1000 + 0xffff_f000) mod 2^32 = 0.
    let mut program = vec![RiscvInstruction::Nop; 1024];
    program.push(RiscvInstruction::Auipc { rd: 1, imm: -1 });
    program.push(RiscvInstruction::Halt);
    let program_bytes = encode_program(&program);
    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .reg_output_claim(/*reg=*/ 1, /*expected=*/ neo_math::F::from_u64(0))
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove with AUIPC wraparound");
    run.verify()
        .expect("trace wiring verify with AUIPC wraparound");
}

#[test]
fn rv32_trace_wiring_runner_accepts_mixed_addi_andi_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    prove_verify_trace_program(program);
}

#[test]
fn rv32_trace_wiring_runner_accepts_mixed_addi_ori_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    prove_verify_trace_program(program);
}

#[test]
fn rv32_trace_wiring_runner_accepts_mixed_with_srai_halt() {
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 2,
            rs1: 1,
            imm: 1,
        },
        RiscvInstruction::Halt,
    ];
    prove_verify_trace_program(program);
}

#[test]
fn rv32_trace_wiring_runner_accepts_full_mixed_sequence_halt() {
    let mut program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::And,
            rd: 2,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Or,
            rd: 3,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Xor,
            rd: 4,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Slt,
            rd: 6,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sltu,
            rd: 7,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sll,
            rd: 8,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Srl,
            rd: 9,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Sra,
            rd: 10,
            rs1: 0,
            imm: 1,
        },
        RiscvInstruction::Branch {
            cond: BranchCondition::Ne,
            rs1: 0,
            rs2: 0,
            imm: 8,
        },
    ];
    program.push(RiscvInstruction::Halt);
    prove_verify_trace_program(program.clone());
}
