from jwcrypto import jwk, jws

from sd_jwt.common import SDObj
from sd_jwt.holder import SDJWTHolder
from sd_jwt.saltbc_plus import SaltBCPlusIssuer, SaltBCPlusUser, _canonical_json
from sd_jwt.verifier import SDJWTVerifier

"""
To run and compare directly against baseline_sdjwtbc (0004):
    pytest tests/test_benchmark_saltbc_plus.py \
      --benchmark-warmup=on \
      --benchmark-disable-gc \
      --benchmark-columns=mean,median,stddev,ops \
      --benchmark-group-by=func \
      --benchmark-sort=mean \
      --benchmark-save=phase1_saltbc_plus_ecdsa \
      --benchmark-compare=0004
"""

# Paper parameter n: maximum number of attributes per credential (|A| = 5 real + 5 decoys)
MAX_ATTRIBUTES = 10

USER_CLAIMS = {
    SDObj("given_name"): "Julian",
    SDObj("family_name"): "Doe",
    SDObj("nationality"): "Danish",
    SDObj("birthdate"): "1990-01-01",
    SDObj("age_over_18"): True,
}

# Setup: KGen (ECDSA P-256)
issuer_key = jwk.JWK.generate(kty="EC", crv="P-256")
issuer_public_jwk = issuer_key.export(private_key=False, as_dict=True)
issuer_public_jwk_obj = jwk.JWK(**issuer_public_jwk)

# Pre-compute ReqCreds, Issue, FinCreds, and Present states for isolated benchmarks
user_wallet = SaltBCPlusUser(sign_alg="ES256")
user_state, req = user_wallet.req_creds(USER_CLAIMS, max_attributes=MAX_ATTRIBUTES)

issuer = SaltBCPlusIssuer(
    user_claims=USER_CLAIMS,
    req=req,
    issuer_keys=[issuer_key],
    sign_alg="ES256",
)
issuer_rsp = issuer.rsp

finalized_credential = user_wallet.fin_creds(user_state, issuer_rsp, issuer_public_jwk_obj)

holder = SDJWTHolder(finalized_credential)
holder.create_presentation({"given_name": True})
presentation = holder.sd_jwt_presentation


def test_benchmark_kgen(benchmark):
    """KGen: Issuer key generation (ECDSA P-256)."""
    benchmark(lambda: jwk.JWK.generate(kty="EC", crv="P-256"))


def test_benchmark_req_creds(benchmark):
    """ReqCreds: User samples n=10 fresh salts (|A| attribute + n-|A| decoy salts)."""
    _, r = benchmark(user_wallet.req_creds, USER_CLAIMS, MAX_ATTRIBUTES)
    assert len(r["attr_salts"]) == len(USER_CLAIMS)
    assert len(r["decoy_salts"]) == MAX_ATTRIBUTES - len(USER_CLAIMS)


def test_benchmark_salt_and_hash(benchmark):
    """SaltedHash + Order: Deterministic hashing on Issuer using user-supplied salts."""
    base_issuer = SaltBCPlusIssuer(
        user_claims=USER_CLAIMS,
        req=req,
        issuer_keys=[issuer_key],
        sign_alg="ES256",
    )

    def deterministic_hash():
        base_issuer.ii_disclosures = []
        base_issuer.decoy_digests = []
        base_issuer._reset_salt_iter()
        return base_issuer._create_sd_claims(USER_CLAIMS)

    res = benchmark(deterministic_hash)
    assert len(res["_sd"]) == MAX_ATTRIBUTES
    assert len(base_issuer.ii_disclosures) == len(USER_CLAIMS)
    assert len(base_issuer.decoy_digests) == MAX_ATTRIBUTES - len(USER_CLAIMS)


def test_benchmark_sign(benchmark):
    """SIG.Sign: Raw ECDSA ES256 signing over canonical n-digest payload pl."""
    payload = _canonical_json(issuer.sd_jwt_payload).encode("utf-8")
    protected_header = _canonical_json({"alg": "ES256", "typ": SaltBCPlusIssuer.SD_JWT_HEADER})

    def raw_sign():
        token = jws.JWS(payload)
        token.add_signature(issuer_key, alg="ES256", protected=protected_header)
        return token.serialize(compact=True)

    benchmark(raw_sign)


def test_benchmark_issue(benchmark):
    """Issue: Deterministic SaltedHash + alphanumeric Order + SIG.Sign -> rsp = σ."""
    def issue_credential():
        iss = SaltBCPlusIssuer(
            user_claims=USER_CLAIMS,
            req=req,
            issuer_keys=[issuer_key],
            sign_alg="ES256",
        )
        return iss.rsp

    sig = benchmark(issue_credential)
    assert isinstance(sig, str) and "." not in sig


def test_benchmark_fin_creds(benchmark):
    """FinCreds: Local reconstruction of pl & S from st + SIG.Vf(ipk, pl, σ)."""
    cred = benchmark(user_wallet.fin_creds, user_state, issuer_rsp, issuer_public_jwk_obj)
    assert cred == finalized_credential


def test_benchmark_present(benchmark):
    """Present: Selective disclosure of 1 of |A| attributes."""
    def create_presentation():
        h = SDJWTHolder(finalized_credential)
        h.create_presentation({"given_name": True})
        return h.sd_jwt_presentation

    benchmark(create_presentation)


def test_benchmark_verify(benchmark):
    """VfPres: Standard verifier checks SIG.Vf and disclosed attribute digest."""
    def verify_presentation():
        verifier = SDJWTVerifier(
            presentation,
            cb_get_issuer_key=lambda iss, hdr: issuer_public_jwk_obj,
            expected_aud=None,
            expected_nonce=None,
        )
        return verifier.get_verified_payload()

    verified = benchmark(verify_presentation)
    assert verified["given_name"] == "Julian"
    assert "family_name" not in verified