from sympy import nextprime, mod_inverse
from random import getrandbits
import math
def extended_gcd(a, b):
    """Extended Euclidean Algorithm to find the GCD and coefficients."""
    if a == 0:
        return b, 0, 1
    else:
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

def modinv(a, m):
    """Compute the modular inverse of a modulo m."""
    gcd, x, y = extended_gcd(a % m, m)
    if gcd != 1:
        raise Exception('Modular inverse does not exist')
    else:
        return x % m

# Given RSA parameters
# Modulus = 17460671
# Exponent = 65537
# PrimeP = 4931
# PrimeQ = 3541

# # Compute modular inverses
# qInv = modinv(PrimeQ, PrimeP)
# pInv = modinv(PrimeP, PrimeQ)

# # Compute CoefficientP and CoefficientQ
# CoefficientP = PrimeQ * qInv
# CoefficientQ = PrimeP * pInv

# print("CoefficientP =", CoefficientP)
# print("CoefficientQ =", CoefficientQ)

def generate_small_rsa_key(key_size_bits=64):
    if key_size_bits < 16:
        raise ValueError("Key size must be at least 16 bits for demonstration purposes.")

    # Generate two distinct primes of half the key size
    prime_size_bits = key_size_bits // 2
    p = nextprime(getrandbits(prime_size_bits))
    q = nextprime(getrandbits(prime_size_bits))
    while p == q:  # Ensure p and q are distinct
        q = nextprime(getrandbits(prime_size_bits))

    # For consistency with your example, we set p and q manually
    # p = 4931
    # q = 3541

    # Calculate modulus (n) and totient (φ(n))
    n = p * q
    phi_n = (p - 1) * (q - 1)

    # Choose a public exponent (commonly 65537)
    e = 65537
    if math.gcd(e, phi_n) != 1:  # Ensure e is coprime with φ(n)
        raise ValueError("Public exponent e is not coprime with φ(n).")

    # Calculate the private exponent (d)
    d = pow(e, -1, phi_n)  # Modular multiplicative inverse of e mod φ(n)

    # Precompute CRT parameters
    dp = d % (p - 1)
    dq = d % (q - 1)

    # # Compute modular inverses
    qInv = modinv(p, q)
    pInv = modinv(q, p)

    # Compute CoefficientP and CoefficientQ
    CoefficientQ = p * qInv
    CoefficientP = q * pInv

    # Return RSA key components in string format
    return {
        "modulusString": str(n),
        "exponentString": str(e),
        "primePString": str(p),
        "primeQString": str(q),
        "dpString": str(dp),
        "dqString": str(dq),
        "coefficientPString": str(CoefficientP),
        "coefficientQString": str(CoefficientQ),
    }

# Generate an example RSA key
# rsa_key = generate_small_rsa_key()
# print(rsa_key)
