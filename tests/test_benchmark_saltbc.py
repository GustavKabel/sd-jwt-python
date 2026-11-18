import pytest
import json
from jwcrypto import jwk, jws
from sd_jwt.issuer import SDJWTIssuer
from sd_jwt.holder import SDJWTHolder
from sd_jwt.verifier import SDJWTVerifier
from sd_jwt.common import SDObj

""" 
To run benchmark run bash command:
    pytest tests/test_benchmark_saltbc.py --benchmark-columns=mean,stddev,ops --benchmark-sort=mean
"""
# 1. Setup baseline user attributes

USER_CLAIMS = {
    SDObj("given_name"): "Julian",
    SDObj("family_name"): "Doe",
    SDObj("birthdate"): "1990-01-01",
    SDObj("address"): {
        "street": "Aarhus Way 1",
        "city": "Aarhus",
        "country": "Denmark"
    },
    SDObj("age_over_18"): True
}

# Generate an EC JWK key pair
issuer_key = jwk.JWK.generate(kty="EC", crv="P-256")
issuer_public_jwk = issuer_key.export(private_key=False, as_dict=True)

# Create a proper JWK object for verification
issuer_public_jwk_obj = jwk.JWK(**issuer_public_jwk)

def test_benchmark_kgen(benchmark):
    """Phase 1: Key Generation (KGen) - EC P-256 JWK"""
    def generate_key():
        return jwk.JWK.generate(kty="EC", crv="P-256")
    benchmark(generate_key)


def test_benchmark_salt_and_hash(benchmark):
    """Phase 1.5a: Salting and Hashing (SaltedHash Phase)"""
    base_issuer = SDJWTIssuer(
        user_claims=USER_CLAIMS,
        issuer_keys=[issuer_key],
        sign_alg="ES256"
    )

    def salt_and_hash():
        base_issuer.ii_disclosures = []
        return base_issuer._create_sd_claims(USER_CLAIMS)

    benchmark(salt_and_hash)


def test_benchmark_sign(benchmark):
    """Phase 1.5b: Raw Signing (ECDSA ES256)"""
    payload = b"dummy payload for pure signature benchmark"
    protected_header = json.dumps({"alg": "ES256"})

    def raw_sign():
        token = jws.JWS(payload)
        token.add_signature(issuer_key, alg="ES256", protected=protected_header)
        return token.serialize()

    benchmark(raw_sign)

def test_benchmark_issue(benchmark):
    """Phase 2: Issuance (SaltedHash + Signing)"""
    def issue_credential():
        issuer = SDJWTIssuer(
            user_claims=USER_CLAIMS,
            issuer_keys=[issuer_key],
            sign_alg="ES256"
        )
        return issuer.sd_jwt_issuance
    benchmark(issue_credential)

# Pre-compute an issued credential for presentation/verification benchmarks
issuer = SDJWTIssuer(
    user_claims=USER_CLAIMS,
    issuer_keys=[issuer_key],
    sign_alg="ES256"
)
issued_credential = issuer.sd_jwt_issuance

def test_benchmark_present(benchmark):
    """Phase 3: Presentation (Selective Disclosure)"""
    def create_presentation():
        holder = SDJWTHolder(issued_credential)
        holder.create_presentation({"given_name": True})
        return holder.sd_jwt_presentation
    benchmark(create_presentation)

holder = SDJWTHolder(issued_credential)
holder.create_presentation({"given_name": True})
presentation = holder.sd_jwt_presentation

def test_benchmark_verify(benchmark):
    """Phase 4: Verification (VfPres)"""
    def verify_presentation():
        verifier = SDJWTVerifier(
            presentation,
            cb_get_issuer_key=lambda issuer, header: issuer_public_jwk_obj,  # Return the JWK object directly
            expected_aud=None,
            expected_nonce=None
        )
        return verifier.get_verified_payload()
    benchmark(verify_presentation)