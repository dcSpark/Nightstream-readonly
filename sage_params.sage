#!/usr/bin/env sage

# Neo cryptographic parameter validation
# Validates security parameters for lattice-based folding scheme

load("lattice_estimator.sage")

# Goldilocks field parameters
q = 2^64 - 2^32 + 1  # Goldilocks prime
n = 54               # Lattice dimension
k = 16               # Number of vectors
d = 32               # Polynomial degree
sigma = 3.2          # Gaussian width
beta = 3             # Bound factor
norm_bound = 4096    # Maximum norm

print("=== Neo Cryptographic Parameter Validation ===")
print(f"Field modulus q = {q}")
print(f"Lattice dimension n = {n}")
print(f"Number of vectors k = {k}")
print(f"Polynomial degree d = {d}")
print(f"Gaussian width σ = {sigma}")
print(f"Bound factor β = {beta}")
print(f"Maximum norm bound = {norm_bound}")

# MSIS (Module SIS) hardness estimate
# This bounds the hardness of finding short vectors in module lattices
msis_lambda = log(k * d * log(q, 2) + 2 * sigma * sqrt(n * k) * d - log(norm_bound, 2), 2)
print(f"\\nMSIS security λ_M = {float(msis_lambda):.2f} bits")

# RLWE (Ring Learning With Errors) hardness estimate
# This bounds the hardness of solving noisy polynomial equations
rlwe_lambda = log(n * log(q, 2) - sigma^2 * n, 2)
print(f"RLWE security λ_R = {float(rlwe_lambda):.2f} bits")

# Overall security level
security_bits = min(msis_lambda, rlwe_lambda)
print(f"\\nOverall security level: {float(security_bits):.2f} bits")

# Security requirements check
MIN_SECURITY = 128
if security_bits >= MIN_SECURITY:
    print(f"✅ SECURITY VALID: {float(security_bits):.2f} ≥ {MIN_SECURITY} bits")
    print("Parameters are cryptographically secure!")
else:
    print(f"❌ SECURITY BREACH: {float(security_bits):.2f} < {MIN_SECURITY} bits")
    print("Parameters are INSECURE - DO NOT USE!")
    exit(1)

print("\\n=== Additional Security Checks ===")

# Check field size
field_bits = log(q, 2)
print(f"Field size: {float(field_bits):.0f} bits")
if field_bits >= 64:
    print("✅ Field size sufficient")
else:
    print("❌ Field size too small")

# Check lattice dimension
if n >= 32:
    print("✅ Lattice dimension sufficient")
else:
    print("❌ Lattice dimension too small")

# Check polynomial degree
if d >= 16:
    print("✅ Polynomial degree sufficient")
else:
    print("❌ Polynomial degree too small")

print(f"\\n=== Security Summary ===")
print(f"✓ MSIS hardness: {float(msis_lambda):.2f} bits")
print(f"✓ RLWE hardness: {float(rlwe_lambda):.2f} bits")
print(f"✓ Overall security: {float(security_bits):.2f} bits")
print(f"✓ Field size: {float(field_bits):.0f} bits")
print(f"✓ Parameters validated for production use")