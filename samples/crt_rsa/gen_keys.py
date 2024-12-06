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
        "Modulus": n,
        "Exponent": e,
        "PrimeP": p,
        "PrimeQ": q,
        "DP": dp,
        "DQ": dq,
        "CoefficientP": CoefficientP,
        "CoefficientQ": CoefficientQ,
    }

# Function to convert integers to 64-bit binary strings
def to_bit_binary(value, bits=64):
    return format(value, f'0{bits}b')

import base64

# Function to encode a message in base64 and convert it to a binary string
def encode_message_to_2048_binary(message):
    # Encode the message as base64
    base64_encoded = base64.b64encode(message.encode())
    
    # Convert the base64-encoded message to an integer
    base64_integer = int.from_bytes(base64_encoded, 'big')
    
    # Convert the integer to a 2048-bit binary string
    return to_bit_binary(base64_integer,2048)

# Function to revert a 2048-bit binary string back to the original message
def revert_2048_binary_to_message(binary_2048):
    # Convert the binary string back to an integer
    original_integer = int(binary_2048, 2)
    
    # Convert the integer back to bytes
    original_bytes = original_integer.to_bytes((original_integer.bit_length() + 7) // 8, 'big')
    
    # Decode the base64 bytes back to the original message
    decoded_message = base64.b64decode(original_bytes).decode()
    
    return decoded_message


# Example usage
message = '''
Technology serves as a bridge between imagination and reality,
empowering us to solve complex problems, connect across boundaries,
and build a future where innovation thrives.'''

binary_output = encode_message_to_2048_binary(message)
print(binary_output)
print(revert_2048_binary_to_message(binary_output))
# Generate an example RSA key
# key_size_bits=2048
# rsa_key = generate_small_rsa_key(key_size_bits)
# print(rsa_key)
# print({key: to_bit_binary(value,key_size_bits) for key, value in rsa_key.items()})
