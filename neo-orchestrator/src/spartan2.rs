pub mod spartan2_ivc {
    #[allow(unused_imports)]
    use neo_fields::spartan2_engine::GoldilocksEngine;
    #[allow(unused_imports)]
    use neo_ccs::to_r1cs;
    // Bring these in when you actually wire:
    // use spartan2::r1cs::{R1CSGens, R1CSSNARK};        // or NeutronNova types if you prefer folding
    // use spartan2::provider::keccak::Keccak256Transcript;

    #[allow(unused_imports)]
    use neo_fold::{Proof /* your folded CCS types */};

    /// Compress a folded CCS instance into a succinct proof.
    pub fn compress_fold(/* folded CCS state here */) /* -> YourProof */ {
        // 1) Convert folded CCS to R1CS shape/inst/wit via to_r1cs::<GoldilocksEngine>()
        // 2) Run Spartan2 keygen once (cache R1CS gens/keys for that shape)
        // 3) Prove with Spartan2 (SNARK or NeutronNova)
        // 4) Return succinct proof bytes + any public outputs you expose
        unimplemented!("Wire Spartan2 at the R1CS layer here");
    }

    /// Helper module for converting dense matrices to Spartan2 format
    pub mod helpers {
        use neo_fields::F;

        #[allow(unused_imports)]
        use spartan2::traits::Engine;
        // TODO: Import r1cs when Spartan2 API is stable
        // use spartan2::r1cs;

        #[allow(unused_variables)]
        pub fn dense_abc_to_shape_instance<E: Engine>(
            _ab1: (&[Vec<F>], &[Vec<F>], &[Vec<F>]),
            _ab2: (&[Vec<F>], &[Vec<F>], &[Vec<F>]),
            _xw1: (&[F], &[F]),
            _xw2: (&[F], &[F]),
        ) -> Result<(String, String, String), String> { // Placeholder types
            // TODO: Return actual Spartan2 types when API is stable
            // (r1cs::Shape<E>, r1cs::Instance<E>, r1cs::Witness<E>)
            // Example: build a block-diagonal shape out of (A1,B1,C1) and (A2,B2,C2),
            // concatenating rows, sharing the same variable ordering (1|z).
            // Translate F -> E::Fr as needed (your pallas/goldilocks conversion utilities).
            // This block is implementation-specific to your pinned Spartan2 rev.
            unimplemented!("Map dense matrices and (x,w) into Spartan2 r1cs::Shape/Instance/Witness")
        }
    }
}
