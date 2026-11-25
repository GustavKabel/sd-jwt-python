import json
from jwcrypto import jwk, jws
from sd_jwt.issuer import SDJWTIssuer
from sd_jwt.holder import SDJWTHolder
from sd_jwt.verifier import SDJWTVerifier
from sd_jwt.common import SDObj

"""
To run benchmark:
    pytest tests/test_benchmark_sdjwtbc.py \
      --benchmark-warmup=on \
      --benchmark-disable-gc \
      --benchmark-columns=mean,median,stddev,ops \
      --benchmark-sort=mean \
      --benchmark-save=baseline_sdjwtbc
"""

# Paper parameter n: maximum number of attributes per credential (real + decoy digests)
MAX_ATTRIBUTES = 10


class SDJWTBCIssuer(SDJWTIssuer):
    # SDJWTIssuer padded with (n - |A|) decoy digests up to a fixed max attribute count n.

    def __init__(self, user_claims: dict, *args, max_attributes: int = MAX_ATTRIBUTES, **kwargs):
        num_real = sum(1 for k in user_claims if isinstance(k, SDObj))
        if max_attributes < num_real:
            raise ValueError(
                f"max_attributes ({max_attributes}) must be >= number of SD attributes ({num_real})"
            )
        num_decoys = max_attributes - num_real
        self.DECOY_MIN_ELEMENTS = num_decoys
        self.DECOY_MAX_ELEMENTS = num_decoys
        kwargs["add_decoy_claims"] = num_decoys > 0
        super().__init__(user_claims, *args, **kwargs)



USER_CLAIMS = {
    SDObj("given_name"): "Julian",
    SDObj("family_name"): "Doe",
    SDObj("nationality"): "Danish",
    SDObj("birthdate"): "1990-01-01",
    SDObj("age_over_18"): True,
}

# Generate an EC JWK key pair
issuer_key = jwk.JWK.generate(kty="EC", crv="P-256")
issuer_public_jwk = issuer_key.export(private_key=False, as_dict=True)
issuer_public_jwk_obj = jwk.JWK(**issuer_public_jwk)

# Pre-compute an issued sdjwtBC credential
issuer = SDJWTBCIssuer(
    user_claims=USER_CLAIMS,
    issuer_keys=[issuer_key],
    sign_alg="ES256",
    max_attributes=MAX_ATTRIBUTES,
)
issued_credential = issuer.sd_jwt_issuance
assert len(issuer.sd_jwt_payload["_sd"]) == MAX_ATTRIBUTES
assert len(issuer.decoy_digests) == MAX_ATTRIBUTES - len(USER_CLAIMS)


def test_benchmark_kgen(benchmark):
    def generate_key():
        return jwk.JWK.generate(kty="EC", crv="P-256")

    benchmark(generate_key)


def test_benchmark_salt_and_hash(benchmark):
    base_issuer = SDJWTBCIssuer(
        user_claims=USER_CLAIMS,
        issuer_keys=[issuer_key],
        sign_alg="ES256",
        max_attributes=MAX_ATTRIBUTES,
    )

    def salt_and_hash():
        base_issuer.ii_disclosures = []
        base_issuer.decoy_digests = []
        return base_issuer._create_sd_claims(USER_CLAIMS)

    res = benchmark(salt_and_hash)
    assert len(res["_sd"]) == MAX_ATTRIBUTES
    assert len(base_issuer.ii_disclosures) == len(USER_CLAIMS)
    assert len(base_issuer.decoy_digests) == MAX_ATTRIBUTES - len(USER_CLAIMS)


def test_benchmark_sign(benchmark):
    payload = json.dumps(issuer.sd_jwt_payload).encode("utf-8")
    protected_header = json.dumps({"alg": "ES256"})

    def raw_sign():
        token = jws.JWS(payload)
        token.add_signature(issuer_key, alg="ES256", protected=protected_header)
        return token.serialize(compact=True)

    benchmark(raw_sign)


def test_benchmark_issue(benchmark):
    def issue_credential():
        iss = SDJWTBCIssuer(
            user_claims=USER_CLAIMS,
            issuer_keys=[issuer_key],
            sign_alg="ES256",
            max_attributes=MAX_ATTRIBUTES,
        )
        return iss.sd_jwt_issuance

    benchmark(issue_credential)


def test_benchmark_present(benchmark):
    def create_presentation():
        h = SDJWTHolder(issued_credential)
        h.create_presentation({"given_name": True})
        return h.sd_jwt_presentation

    benchmark(create_presentation)


holder = SDJWTHolder(issued_credential)
holder.create_presentation({"given_name": True})
presentation = holder.sd_jwt_presentation


def test_benchmark_fin_creds(benchmark):
    all_claims = {
        "given_name": True,
        "family_name": True,
        "nationality": True,
        "birthdate": True,
        "age_over_18": True,
    }

    def fin_creds():
        h = SDJWTHolder(issued_credential)
        h.create_presentation(all_claims)
        v = SDJWTVerifier(
            h.sd_jwt_presentation,
            cb_get_issuer_key=lambda iss, hdr: issuer_public_jwk_obj,
            expected_aud=None,
            expected_nonce=None,
        )
        return v.get_verified_payload()

    verified = benchmark(fin_creds)
    assert verified["given_name"] == "Julian"
    assert verified["nationality"] == "Danish"
    assert verified["age_over_18"] is True


def test_benchmark_verify(benchmark):
    """Phase 4: Verification (VfPres)"""
    def verify_presentation():
        verifier = SDJWTVerifier(
            presentation,
            cb_get_issuer_key=lambda issuer, header: issuer_public_jwk_obj,
            expected_aud=None,
            expected_nonce=None,
        )
        return verifier.get_verified_payload()

    verified = benchmark(verify_presentation)
    assert verified["given_name"] == "Julian"
    assert "family_name" not in verified