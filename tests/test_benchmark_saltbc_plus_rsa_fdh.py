import pytest

from sd_jwt.common import SDObj
from sd_jwt.holder import SDJWTHolder
from sd_jwt.rsa_fdh import RSAFDHPrivateKey, b64url_encode, rsa_fdh_sign
from sd_jwt.saltbc_plus import _canonical_json
from sd_jwt.saltbc_plus_rsa_fdh import (
    RSA_FDH_SIGN_ALG,
    SaltBCPlusRSAFDHIssuer,
    SaltBCPlusRSAFDHUser,
    SaltBCPlusRSAFDHVerifier,
)

"""
To run and save Phase 2 benchmarks (RSA-2048 legacy baseline & RSA-3072 128-bit security):
    pytest tests/test_benchmark_saltbc_plus_rsa_fdh.py \
      --benchmark-warmup=on \
      --benchmark-disable-gc \
      --benchmark-columns=mean,median,stddev,ops \
      --benchmark-group-by=func \
      --benchmark-sort=mean \
      --benchmark-save=phase2_saltbc_plus_rsa_fdh

    Benchmark note:
        KGen and MGF1-SHA256 execute in OpenSSL C
        modular exponentiation in SIG.Sign and SIG.Vf executes in CPython's built-in pow().
"""

MAX_ATTRIBUTES = 10
KEY_SIZES = [2048, 3072]

USER_CLAIMS = {
    SDObj("given_name"): "Julian",
    SDObj("family_name"): "Doe",
    SDObj("nationality"): "Danish",
    SDObj("birthdate"): "1990-01-01",
    SDObj("age_over_18"): True,
}


def _build_fixture(key_size: int) -> dict:
    # Pre-compute KGen, ReqCreds, Issue, FinCreds, and Present for isolated benchmarks
    issuer_key = RSAFDHPrivateKey.generate(key_size=key_size)
    issuer_public_key = issuer_key.public_key()

    user_wallet = SaltBCPlusRSAFDHUser(sign_alg=RSA_FDH_SIGN_ALG)
    user_state, req = user_wallet.req_creds(USER_CLAIMS, max_attributes=MAX_ATTRIBUTES)

    issuer = SaltBCPlusRSAFDHIssuer(
        user_claims=USER_CLAIMS,
        req=req,
        issuer_keys=[issuer_key],
        sign_alg=RSA_FDH_SIGN_ALG,
    )
    issuer_rsp = issuer.rsp

    finalized_credential = user_wallet.fin_creds(user_state, issuer_rsp, issuer_public_key)

    holder = SDJWTHolder(finalized_credential)
    holder.create_presentation({"given_name": True})
    presentation = holder.sd_jwt_presentation

    protected_b64 = b64url_encode(
        _canonical_json(
            {"alg": RSA_FDH_SIGN_ALG, "typ": SaltBCPlusRSAFDHIssuer.SD_JWT_HEADER}
        ).encode("utf-8")
    )
    payload_b64 = b64url_encode(_canonical_json(issuer.sd_jwt_payload).encode("utf-8"))
    signing_input = f"{protected_b64}.{payload_b64}".encode("ascii")

    return {
        "issuer_key": issuer_key,
        "issuer_public_key": issuer_public_key,
        "user_wallet": user_wallet,
        "user_state": user_state,
        "req": req,
        "issuer": issuer,
        "issuer_rsp": issuer_rsp,
        "finalized_credential": finalized_credential,
        "presentation": presentation,
        "signing_input": signing_input,
    }


_FIXTURES = {size: _build_fixture(size) for size in KEY_SIZES}


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_kgen(benchmark, key_size):
    # KGen: Issuer RSA key generation via cryptography (OpenSSL C).
    sk = benchmark.pedantic(
        RSAFDHPrivateKey.generate,
        args=(key_size,),
        rounds=10,
        iterations=1,
        warmup_rounds=1,
    )
    assert sk.key_size() == key_size


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_req_creds(benchmark, key_size):
    # ReqCreds: User samples n=10 fresh salts (|A| attribute + n-|A| decoy salts).
    fx = _FIXTURES[key_size]
    _, r = benchmark(fx["user_wallet"].req_creds, USER_CLAIMS, MAX_ATTRIBUTES)
    assert len(r["attr_salts"]) == len(USER_CLAIMS)
    assert len(r["decoy_salts"]) == MAX_ATTRIBUTES - len(USER_CLAIMS)


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_salt_and_hash(benchmark, key_size):
    # SaltedHash + Order: deterministic hashing on Issuer using user-supplied salts.
    fx = _FIXTURES[key_size]
    base_issuer = SaltBCPlusRSAFDHIssuer(
        user_claims=USER_CLAIMS,
        req=fx["req"],
        issuer_keys=[fx["issuer_key"]],
        sign_alg=RSA_FDH_SIGN_ALG,
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


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_sign(benchmark, key_size):
    # SIG.Sign: Raw RFC 9381 RSA-FDH signing over canonical n-digest payload pl.
    fx = _FIXTURES[key_size]
    sig = benchmark(rsa_fdh_sign, fx["issuer_key"], fx["signing_input"])
    assert len(sig) == fx["issuer_key"].modulus_len()


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_issue(benchmark, key_size):
    # Issue: Deterministic SaltedHash + alphanumeric Order + RSA-FDH SIG.Sign -> rsp.
    fx = _FIXTURES[key_size]

    def issue_credential():
        iss = SaltBCPlusRSAFDHIssuer(
            user_claims=USER_CLAIMS,
            req=fx["req"],
            issuer_keys=[fx["issuer_key"]],
            sign_alg=RSA_FDH_SIGN_ALG,
        )
        return iss.rsp

    sig = benchmark(issue_credential)
    assert sig == fx["issuer_rsp"]


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_fin_creds(benchmark, key_size):
    # FinCreds: Local reconstruction of pl & S from st + RSA-FDH SIG.Vf(ipk, pl, sigma).
    fx = _FIXTURES[key_size]
    cred = benchmark(
        fx["user_wallet"].fin_creds,
        fx["user_state"],
        fx["issuer_rsp"],
        fx["issuer_public_key"],
    )
    assert cred == fx["finalized_credential"]


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_present(benchmark, key_size):
    # Present: Selective disclosure of 1 of |A| attributes via stock SDJWTHolder.
    fx = _FIXTURES[key_size]

    def create_presentation():
        h = SDJWTHolder(fx["finalized_credential"])
        h.create_presentation({"given_name": True})
        return h.sd_jwt_presentation

    pres = benchmark(create_presentation)
    assert pres == fx["presentation"]


@pytest.mark.parametrize("key_size", KEY_SIZES, ids=["rsa2048", "rsa3072"])
def test_benchmark_verify(benchmark, key_size):
    # VfPres: SaltBCPlusRSAFDHVerifier checks RSA-FDH SIG.Vf and disclosed attribute digest.
    fx = _FIXTURES[key_size]
    pk = fx["issuer_public_key"]
    presentation = fx["presentation"]

    def verify_presentation():
        verifier = SaltBCPlusRSAFDHVerifier(
            presentation,
            cb_get_issuer_key=lambda iss, hdr: pk,
            expected_aud=None,
            expected_nonce=None,
        )
        return verifier.get_verified_payload()

    verified = benchmark(verify_presentation)
    assert verified["given_name"] == "Julian"
    assert "family_name" not in verified
