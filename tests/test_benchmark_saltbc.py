import pytest
from jwcrypto import jwk
from sd_jwt.issuer import SDJWTIssuer
from sd_jwt.holder import SDJWTHolder
from sd_jwt.verifier import SDJWTVerifier

# 1. Setup baseline user attributes
USER_CLAIMS = {
    "given_name": "Julian",
    "family_name": "Doe",
    "birthdate": "1990-01-01",
    "address": {
        "street": "Aarhus Way 1",
        "city": "Aarhus",
        "country": "Denmark"
    },
    "age_over_18": True
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

def test_benchmark_issue(benchmark):
    """Phase 2: Issuance (SaltedHash + Signing)"""
    def issue_credential():
        issuer = SDJWTIssuer(
            user_claims=USER_CLAIMS,
            issuer_keys=[issuer_key],
            sign_alg="ES256"
        )
        return issuer.sd_jwt_issuance
    benchmark(issue_issuer_credential := issue_credential)

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