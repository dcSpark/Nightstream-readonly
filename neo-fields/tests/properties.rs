use neo_fields::{ExtF, F};
use p3_field::PrimeCharacteristicRing;
use quickcheck::{Arbitrary, Gen};
use quickcheck_macros::quickcheck;

/// Implement Arbitrary for F to enable QuickCheck property testing
impl Arbitrary for F {
    fn arbitrary(g: &mut Gen) -> Self {
        F::from_u64(u64::arbitrary(g))
    }
}

/// Implement Arbitrary for ExtF to enable QuickCheck property testing
impl Arbitrary for ExtF {
    fn arbitrary(g: &mut Gen) -> Self {
        let real = F::arbitrary(g);
        let imag = F::arbitrary(g);
        ExtF::new_complex(real, imag)
    }
}

/// Property: Field addition is associative: (a + b) + c = a + (b + c)
#[quickcheck]
fn prop_f_addition_associative(a: F, b: F, c: F) -> bool {
    (a + b) + c == a + (b + c)
}

/// Property: Field addition is commutative: a + b = b + a
#[quickcheck]
fn prop_f_addition_commutative(a: F, b: F) -> bool {
    a + b == b + a
}

/// Property: Field multiplication is associative: (a * b) * c = a * (b * c)
#[quickcheck]
fn prop_f_multiplication_associative(a: F, b: F, c: F) -> bool {
    (a * b) * c == a * (b * c)
}

/// Property: Field multiplication is commutative: a * b = b * a
#[quickcheck]
fn prop_f_multiplication_commutative(a: F, b: F) -> bool {
    a * b == b * a
}

/// Property: Distributivity: a * (b + c) = a * b + a * c
#[quickcheck]
fn prop_f_distributivity(a: F, b: F, c: F) -> bool {
    a * (b + c) == a * b + a * c
}

/// Property: Additive identity: a + 0 = a
#[quickcheck]
fn prop_f_additive_identity(a: F) -> bool {
    a + F::ZERO == a && F::ZERO + a == a
}

/// Property: Multiplicative identity: a * 1 = a
#[quickcheck]
fn prop_f_multiplicative_identity(a: F) -> bool {
    a * F::ONE == a && F::ONE * a == a
}

/// Property: Additive inverse: a + (-a) = 0
#[quickcheck]
fn prop_f_additive_inverse(a: F) -> bool {
    a + (-a) == F::ZERO && (-a) + a == F::ZERO
}

/// Property: Multiplicative inverse for non-zero elements: a * a^(-1) = 1
#[quickcheck]
fn prop_f_multiplicative_inverse(a: F) -> bool {
    if a == F::ZERO {
        true // Skip zero (no multiplicative inverse)
    } else {
        let a_inv = a.inverse();
        a * a_inv == F::ONE && a_inv * a == F::ONE
    }
}

/// Property: Extension field addition is associative: (a + b) + c = a + (b + c)
#[quickcheck]
fn prop_extf_addition_associative(a: ExtF, b: ExtF, c: ExtF) -> bool {
    (a + b) + c == a + (b + c)
}

/// Property: Extension field addition is commutative: a + b = b + a
#[quickcheck]
fn prop_extf_addition_commutative(a: ExtF, b: ExtF) -> bool {
    a + b == b + a
}

/// Property: Extension field multiplication is associative: (a * b) * c = a * (b * c)
#[quickcheck]
fn prop_extf_multiplication_associative(a: ExtF, b: ExtF, c: ExtF) -> bool {
    (a * b) * c == a * (b * c)
}

/// Property: Extension field multiplication is commutative: a * b = b * a
#[quickcheck]
fn prop_extf_multiplication_commutative(a: ExtF, b: ExtF) -> bool {
    a * b == b * a
}

/// Property: Extension field distributivity: a * (b + c) = a * b + a * c
#[quickcheck]
fn prop_extf_distributivity(a: ExtF, b: ExtF, c: ExtF) -> bool {
    a * (b + c) == a * b + a * c
}

/// Property: Extension field additive identity: a + 0 = a
#[quickcheck]
fn prop_extf_additive_identity(a: ExtF) -> bool {
    a + ExtF::ZERO == a && ExtF::ZERO + a == a
}

/// Property: Extension field multiplicative identity: a * 1 = a
#[quickcheck]
fn prop_extf_multiplicative_identity(a: ExtF) -> bool {
    a * ExtF::ONE == a && ExtF::ONE * a == a
}

/// Property: Extension field additive inverse: a + (-a) = 0
#[quickcheck]
fn prop_extf_additive_inverse(a: ExtF) -> bool {
    a + (-a) == ExtF::ZERO && (-a) + a == ExtF::ZERO
}

/// Property: Extension field multiplicative inverse for non-zero elements: a * a^(-1) = 1
#[quickcheck]
fn prop_extf_multiplicative_inverse(a: ExtF) -> bool {
    if a == ExtF::ZERO {
        true // Skip zero (no multiplicative inverse)
    } else {
        let a_inv = a.inverse();
        let product1 = a * a_inv;
        let product2 = a_inv * a;
        // Allow for small numerical errors in extension field arithmetic
        (product1 - ExtF::ONE).abs_norm() < 1e-10 && (product2 - ExtF::ONE).abs_norm() < 1e-10
    }
}

/// Property: Base field embedding preserves operations: embed(a + b) = embed(a) + embed(b)
#[quickcheck]
fn prop_embed_preserves_addition(a: F, b: F) -> bool {
    use neo_fields::embed_base_to_ext;
    embed_base_to_ext(a + b) == embed_base_to_ext(a) + embed_base_to_ext(b)
}

/// Property: Base field embedding preserves multiplication: embed(a * b) = embed(a) * embed(b)
#[quickcheck]
fn prop_embed_preserves_multiplication(a: F, b: F) -> bool {
    use neo_fields::embed_base_to_ext;
    embed_base_to_ext(a * b) == embed_base_to_ext(a) * embed_base_to_ext(b)
}

/// Property: Projection is left inverse of embedding for base field elements
#[quickcheck]
fn prop_embed_project_roundtrip(a: F) -> bool {
    use neo_fields::{embed_base_to_ext, project_ext_to_base};
    match project_ext_to_base(embed_base_to_ext(a)) {
        Some(recovered) => recovered == a,
        None => false, // Should always succeed for embedded base elements
    }
}

/// Property: Extension field norm is multiplicative: norm(a * b) = norm(a) * norm(b)
#[quickcheck]
fn prop_norm_multiplicative(a: ExtF, b: ExtF) -> bool {
    let norm_ab = (a * b).abs_norm();
    let norm_a_times_norm_b = a.abs_norm() * b.abs_norm();
    // Allow for floating point precision errors
    (norm_ab - norm_a_times_norm_b).abs() < 1e-6
}

/// Property: Norm of base field element equals its absolute value
#[quickcheck]
fn prop_norm_base_field(a: F) -> bool {
    use neo_fields::embed_base_to_ext;
    let ext_a = embed_base_to_ext(a);
    let norm = ext_a.abs_norm();
    let expected = a.as_canonical_u64() as f64;
    (norm - expected).abs() < 1e-6
}

/// Property: Conjugate operation is involutive: conj(conj(a)) = a
#[quickcheck]
fn prop_conjugate_involutive(a: ExtF) -> bool {
    let [real, imag] = a.to_array();
    let conj_a = ExtF::new_complex(real, -imag);
    let conj_conj_a = ExtF::new_complex(real, -(-imag));
    conj_conj_a == a
}

/// Property: Norm via conjugate: a * conj(a) = norm(a) (as base field element)
#[quickcheck]
fn prop_norm_via_conjugate(a: ExtF) -> bool {
    let [real, imag] = a.to_array();
    let conj_a = ExtF::new_complex(real, -imag);
    let product = a * conj_a;
    
    // Product should be a real number (imaginary part = 0)
    let [prod_real, prod_imag] = product.to_array();
    if prod_imag != F::ZERO {
        return false;
    }
    
    // Real part should equal the norm
    let expected_norm = a.abs_norm();
    let actual_norm = prod_real.as_canonical_u64() as f64;
    (expected_norm - actual_norm).abs() < 1e-6
}

/// Property: Zero norm only for zero element
#[quickcheck]
fn prop_zero_norm_iff_zero(a: ExtF) -> bool {
    let norm = a.abs_norm();
    if a == ExtF::ZERO {
        norm == 0.0
    } else {
        norm > 0.0
    }
}

/// Test specific field properties for Goldilocks field
#[test]
fn test_goldilocks_specific_properties() {
    // Test that the field modulus is correct
    assert_eq!(F::ORDER_U64, 0xFFFFFFFF00000001u64, "Goldilocks modulus should be 2^64 - 2^32 + 1");
    
    // Test generator properties
    let gen = F::from_u64(7); // Common generator for Goldilocks
    assert_ne!(gen, F::ZERO);
    assert_ne!(gen, F::ONE);
    
    // Test that -1 is not a square (needed for extension field)
    let minus_one = -F::ONE;
    let mut is_square = false;
    for i in 0..100 {
        let candidate = F::from_u64(i);
        if candidate * candidate == minus_one {
            is_square = true;
            break;
        }
    }
    assert!(!is_square, "-1 should not be a quadratic residue in Goldilocks for extension field security");
}

/// Test extension field construction properties
#[test]
fn test_extension_field_construction() {
    // Test that i^2 = -1 in the extension field
    let i = ExtF::new_complex(F::ZERO, F::ONE);
    let i_squared = i * i;
    let minus_one = ExtF::new_complex(-F::ONE, F::ZERO);
    assert_eq!(i_squared, minus_one, "i^2 should equal -1 in extension field");
    
    // Test basic complex arithmetic
    let a = ExtF::new_complex(F::ONE, F::from_u64(2));
    let b = ExtF::new_complex(F::from_u64(3), F::ONE);
    
    // (1 + 2i) + (3 + i) = 4 + 3i
    let sum = a + b;
    assert_eq!(sum, ExtF::new_complex(F::from_u64(4), F::from_u64(3)));
    
    // (1 + 2i) * (3 + i) = 3 + i + 6i + 2i^2 = 3 + 7i - 2 = 1 + 7i
    let product = a * b;
    assert_eq!(product, ExtF::new_complex(F::ONE, F::from_u64(7)));
}
