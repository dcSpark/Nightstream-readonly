use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::{Rv32TraceWiring, Rv32TraceWiringRun};
use neo_fold::shard::ShardProof;
use neo_math::K;
use neo_memory::cpu::bus_layout::{
    build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes, ShoutInstanceShape,
};
use neo_memory::riscv::lookups::{encode_program, RiscvInstruction, RiscvOpcode, RiscvShoutTables};
use neo_memory::witness::LutTableSpec;
use p3_field::PrimeCharacteristicRing;

fn prove_packed_divu_trace_program() -> (Rv32TraceWiringRun, ShardProof) {
    // Program includes DIVU so shared-bus route uses packed RV32M lane columns.
    let program = vec![
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 1,
            rs1: 0,
            imm: 13,
        },
        RiscvInstruction::IAlu {
            op: RiscvOpcode::Add,
            rd: 2,
            rs1: 0,
            imm: 5,
        },
        RiscvInstruction::RAlu {
            op: RiscvOpcode::Divu,
            rd: 3,
            rs1: 1,
            rs2: 2,
        },
        RiscvInstruction::Halt,
    ];
    let program_bytes = encode_program(&program);

    let mut run = Rv32TraceWiring::from_rom(/*program_base=*/ 0, &program_bytes)
        .mode(FoldingMode::Optimized)
        .chunk_rows(program.len())
        .min_trace_len(program.len())
        .max_steps(program.len())
        .prove()
        .expect("trace wiring prove");
    run.verify().expect("trace wiring verify");

    let proof = run.proof().clone();
    (run, proof)
}

fn tamper_divu_packed_diff_bit_opening(run: &Rv32TraceWiringRun, proof: &mut ShardProof) {
    let steps = run.steps_public();
    assert_eq!(steps.len(), 1, "expected one trace step");
    let step0 = &steps[0];

    let shout_shapes = step0.lut_insts.iter().map(|inst| ShoutInstanceShape {
        ell_addr: inst.d * inst.ell,
        lanes: inst.lanes.max(1),
        n_vals: 1usize,
        addr_group: inst.addr_group,
        selector_group: inst.selector_group,
    });
    let twist_shapes = step0.mem_insts.iter().map(|inst| {
        assert!(inst.n_side.is_power_of_two(), "twist n_side must be power-of-two");
        let ell = inst.n_side.trailing_zeros() as usize;
        (inst.d * ell, inst.lanes.max(1))
    });

    let bus = build_bus_layout_for_instances_with_shout_shapes_and_twist_lanes(
        run.layout().m,
        run.layout().m_in,
        run.layout().t,
        shout_shapes,
        twist_shapes,
    )
    .expect("shared-bus layout");

    let divu_table_id = RiscvShoutTables::new(32).opcode_to_id(RiscvOpcode::Divu).0;
    let divu_inst_idx = step0
        .lut_insts
        .iter()
        .position(|inst| {
            inst.table_id == divu_table_id
                && matches!(
                    inst.table_spec,
                    Some(LutTableSpec::RiscvOpcodePacked {
                        opcode: RiscvOpcode::Divu,
                        xlen: 32
                    })
                )
        })
        .expect("expected packed DIVU lookup instance");

    let lane0 = bus.shout_cols[divu_inst_idx]
        .lanes
        .first()
        .expect("expected lane 0 for DIVU");

    // DIVU packed layout: [lhs, rhs, rem, rhs_inv, rhs_is_zero, diff, diff_bits(32)].
    // Flip the first diff bit (packed-only aux column).
    let diff_bit0_col = lane0.addr_bits.start + 6;

    let ccs_out0 = &mut proof.steps[0].fold.ccs_out[0];
    let bus_y_base = ccs_out0
        .y_scalars
        .len()
        .checked_sub(bus.bus_cols)
        .expect("CPU y_scalars should include shared-bus tail openings");
    let open_idx = bus.y_scalar_index(bus_y_base, diff_bit0_col);
    ccs_out0.y_scalars[open_idx] += K::ONE;
}

#[test]
fn packed_divu_diff_bit_tamper_is_rejected() {
    let (run, mut proof) = prove_packed_divu_trace_program();
    tamper_divu_packed_diff_bit_opening(&run, &mut proof);
    assert!(
        run.verify_proof(&proof).is_err(),
        "tampered packed DIVU diff-bit opening must fail verification"
    );
}
