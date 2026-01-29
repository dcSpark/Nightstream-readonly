use midnight_circuits::instructions::{AssignmentInstructions, PublicInputInstructions};
use midnight_circuits::types::AssignedNative;
use midnight_proofs::circuit::{Layouter, Value};
use midnight_proofs::plonk::Error;
use midnight_zk_stdlib::{Relation, ZkStdLib, ZkStdLibArch};
use neo_midnight_bridge::goldilocks::{alloc_gl_public, gl_add_mod_var, GlVar, OuterScalar, GOLDILOCKS_P_U64};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlAllocPublicRelation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlAllocPublicInstance {
    pub x: u64,
}

impl Relation for GlAllocPublicRelation {
    type Instance = GlAllocPublicInstance;
    type Witness = ();

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![OuterScalar::from(instance.x)])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writer.write_all(&bytes)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        bincode::deserialize(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        _witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        let _x = alloc_gl_public(std_lib, layouter, instance.as_ref().map(|i| i.x))?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlAddAmbiguousCarryRelation;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlAddAmbiguousCarryInstance {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlAddAmbiguousCarryWitness {
    /// Unconstrained metadata used only to compute the carry bit witness.
    pub x_val: u64,
    /// Unconstrained metadata used only to compute the carry bit witness.
    pub y_val: u64,
}

impl Relation for GlAddAmbiguousCarryRelation {
    type Instance = GlAddAmbiguousCarryInstance;
    type Witness = GlAddAmbiguousCarryWitness;

    fn used_chips(&self) -> ZkStdLibArch {
        let mut arch = ZkStdLibArch::default();
        arch.nr_pow2range_cols = 4;
        arch
    }

    fn format_instance(instance: &Self::Instance) -> Result<Vec<OuterScalar>, Error> {
        Ok(vec![
            OuterScalar::from(instance.x),
            OuterScalar::from(instance.y),
            OuterScalar::from(instance.z),
        ])
    }

    fn write_relation<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let bytes = bincode::serialize(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writer.write_all(&bytes)
    }

    fn read_relation<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        bincode::deserialize(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    fn circuit(
        &self,
        std_lib: &ZkStdLib,
        layouter: &mut impl Layouter<OuterScalar>,
        instance: Value<Self::Instance>,
        witness: Value<Self::Witness>,
    ) -> Result<(), Error> {
        // Allocate x and y as public inputs, but compute the carry bit using unconstrained
        // witness metadata (`x_val`, `y_val`) to simulate a malicious prover.
        let x_u64 = instance.as_ref().map(|i| i.x);
        let x_assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, x_u64.map(OuterScalar::from))?;
        std_lib.constrain_as_public_input(layouter, &x_assigned)?;
        let y_u64 = instance.as_ref().map(|i| i.y);
        let y_assigned: AssignedNative<OuterScalar> = std_lib.assign(layouter, y_u64.map(OuterScalar::from))?;
        std_lib.constrain_as_public_input(layouter, &y_assigned)?;

        let x = GlVar {
            assigned: x_assigned,
            value: witness.as_ref().map(|w| w.x_val),
        };
        let y = GlVar {
            assigned: y_assigned,
            value: witness.as_ref().map(|w| w.y_val),
        };

        let z = gl_add_mod_var(std_lib, layouter, &x, &y)?;
        std_lib.constrain_as_public_input(layouter, &z.assigned)?;
        Ok(())
    }
}

pub fn gl_modulus_u64() -> u64 {
    GOLDILOCKS_P_U64
}
