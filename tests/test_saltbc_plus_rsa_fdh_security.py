import copy

import pytest

from sd_jwt.common import SDObj
from sd_jwt.holder import SDJWTHolder
from sd_jwt.rsa_fdh import (
    RSAFDHPrivateKey,
    b64url_decode,
    b64url_encode,
    i2osp,
    os2ip,
    rsa_fdh_sign,
    rsa_fdh_verify,
)
from sd_jwt.saltbc_plus import SaltBCPlusVerificationError, _canonical_json
from sd_jwt.saltbc_plus_rsa_fdh import (
    RSA_FDH_SIGN_ALG,
    SaltBCPlusRSAFDHIssuer,
    SaltBCPlusRSAFDHUser,
    SaltBCPlusRSAFDHVerifier,
)

MAX_ATTRIBUTES = 10

USER_CLAIMS = {
    SDObj("given_name"): "Julian",
    SDObj("family_name"): "Doe",
    SDObj("nationality"): "Danish",
    SDObj("birthdate"): "1990-01-01",
    SDObj("age_over_18"): True,
}


@pytest.fixture(scope="module")
def rsa2048_keypair():
    # KGen for RSA-2048
    sk = RSAFDHPrivateKey.generate(key_size=2048)
    return sk, sk.public_key()


@pytest.fixture(scope="module")
def rsa3072_keypair():
    # KGen for RSA-3072 (128-bit security matching ECDSA P-256)
    sk = RSAFDHPrivateKey.generate(key_size=3072)
    return sk, sk.public_key()


@pytest.fixture
def setup_parties_2048(rsa2048_keypair):
    sk, pk = rsa2048_keypair
    user = SaltBCPlusRSAFDHUser()
    state, req = user.req_creds(USER_CLAIMS, max_attributes=MAX_ATTRIBUTES)
    return sk, pk, user, state, req


def _replace_presentation_sig(presentation: str, new_sig_b64: str) -> str:
    parts = presentation.split("~")
    jws_parts = parts[0].split(".")
    jws_parts[2] = new_sig_b64
    parts[0] = ".".join(jws_parts)
    return "~".join(parts)


@pytest.mark.parametrize("key_size", [2048, 3072], ids=["rsa2048", "rsa3072"])
def test_honest_partial_and_full_disclosure(key_size, rsa2048_keypair, rsa3072_keypair):
    # Honest KGen -> ReqCreds -> Issue -> FinCreds -> Present -> VfPres
    sk, pk = rsa2048_keypair if key_size == 2048 else rsa3072_keypair
    user = SaltBCPlusRSAFDHUser()
    state, req = user.req_creds(USER_CLAIMS, max_attributes=MAX_ATTRIBUTES)

    issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    assert len(issuer.sd_jwt_payload["_sd"]) == MAX_ATTRIBUTES
    assert issuer.sd_jwt_issuance is None
    assert len(b64url_decode(issuer.rsp)) == sk.modulus_len()

    finalized_cred = user.fin_creds(state, issuer.rsp, pk)

    # Partial disclosure (1 of |A| attributes)
    holder = SDJWTHolder(finalized_cred)
    holder.create_presentation({"given_name": True})
    verifier = SaltBCPlusRSAFDHVerifier(
        holder.sd_jwt_presentation,
        cb_get_issuer_key=lambda iss, hdr: pk,
        expected_aud=None,
        expected_nonce=None,
    )
    partial = verifier.get_verified_payload()
    assert partial["given_name"] == "Julian"
    assert "family_name" not in partial

    # Full disclosure
    holder_full = SDJWTHolder(finalized_cred)
    holder_full.create_presentation({k.value: True for k in USER_CLAIMS})
    verifier_full = SaltBCPlusRSAFDHVerifier(
        holder_full.sd_jwt_presentation,
        cb_get_issuer_key=lambda iss, hdr: pk,
        expected_aud=None,
        expected_nonce=None,
    )
    full = verifier_full.get_verified_payload()
    assert full["given_name"] == "Julian"
    assert full["family_name"] == "Doe"
    assert full["nationality"] == "Danish"
    assert full["birthdate"] == "1990-01-01"
    assert full["age_over_18"] is True


def test_honest_roundtrip_with_kid():
    # Optional kid propagates through Issue, FinCreds, and VfPres key callback
    sk = RSAFDHPrivateKey.generate(key_size=2048, kid="issuer-rsa-fdh-1")
    pk = sk.public_key()
    assert "kid" in sk and sk["kid"] == "issuer-rsa-fdh-1"
    assert "kid" in pk and pk["kid"] == "issuer-rsa-fdh-1"

    user = SaltBCPlusRSAFDHUser()
    state, req = user.req_creds(USER_CLAIMS, max_attributes=MAX_ATTRIBUTES)
    issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    cred = user.fin_creds(state, issuer.rsp, pk)

    holder = SDJWTHolder(cred)
    holder.create_presentation({"given_name": True})

    seen_headers = {}

    def get_key(iss, hdr):
        seen_headers.update(hdr)
        return pk

    verifier = SaltBCPlusRSAFDHVerifier(
        holder.sd_jwt_presentation,
        cb_get_issuer_key=get_key,
    )
    assert seen_headers["kid"] == "issuer-rsa-fdh-1"
    assert seen_headers["alg"] == RSA_FDH_SIGN_ALG
    assert verifier.get_verified_payload()["given_name"] == "Julian"


def test_deterministic_unique_signature_and_key_binding(
    setup_parties_2048, rsa3072_keypair
):
    # Deterministic uniqueness: identical (isk, A, req) produces byte-identical rsp
    sk, pk, user, state, req = setup_parties_2048
    _, other_pk = rsa3072_keypair

    issuer_1 = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    issuer_2 = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    assert issuer_1.rsp == issuer_2.rsp

    msg = b"saltBC+ deterministic signing input"
    sig_1 = rsa_fdh_sign(sk, msg)
    sig_2 = rsa_fdh_sign(sk, msg)
    assert sig_1 == sig_2
    assert rsa_fdh_verify(pk, msg, sig_1) is True
    assert rsa_fdh_verify(other_pk, msg, sig_1) is False

    # Cross-key verification fails in FinCreds
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, issuer_1.rsp, other_pk)


def test_rejects_wrong_signature_byte_length(setup_parties_2048):
    sk, pk, user, state, req = setup_parties_2048
    issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    cred = user.fin_creds(state, issuer.rsp, pk)

    sig_bytes = b64url_decode(issuer.rsp)
    jws_parts = issuer.serialized_sd_jwt.split(".")
    signing_input = f"{jws_parts[0]}.{jws_parts[1]}".encode("ascii")

    # Prepending a leading zero byte preserves OS2IP(sig) == s, but violates fixed length k
    too_long = b"\x00" + sig_bytes
    too_short = sig_bytes[1:]
    assert rsa_fdh_verify(pk, signing_input, too_long) is False
    assert rsa_fdh_verify(pk, signing_input, too_short) is False

    holder = SDJWTHolder(cred)
    holder.create_presentation({"given_name": True})

    for bad_sig in (too_long, too_short):
        bad_rsp = b64url_encode(bad_sig)
        with pytest.raises(SaltBCPlusVerificationError):
            user.fin_creds(state, bad_rsp, pk)
        tampered_pres = _replace_presentation_sig(holder.sd_jwt_presentation, bad_rsp)
        with pytest.raises(SaltBCPlusVerificationError):
            SaltBCPlusRSAFDHVerifier(tampered_pres, cb_get_issuer_key=lambda i, h: pk)


def test_rejects_out_of_range_signature_s_ge_n(setup_parties_2048):
    sk, pk, user, state, req = setup_parties_2048
    k = pk.modulus_len
    upper_bound = 1 << (8 * k)

    # 1. Exact k-byte encoding of s = n must be rejected in FinCreds and VfPres
    issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    cred = user.fin_creds(state, issuer.rsp, pk)
    s_eq_n_bytes = i2osp(pk.n, k)
    s_eq_n_b64 = b64url_encode(s_eq_n_bytes)

    jws_parts = issuer.serialized_sd_jwt.split(".")
    signing_input = f"{jws_parts[0]}.{jws_parts[1]}".encode("ascii")
    assert rsa_fdh_verify(pk, signing_input, s_eq_n_bytes) is False

    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, s_eq_n_b64, pk)

    holder = SDJWTHolder(cred)
    holder.create_presentation({"given_name": True})
    tampered_pres = _replace_presentation_sig(holder.sd_jwt_presentation, s_eq_n_b64)
    with pytest.raises(SaltBCPlusVerificationError):
        SaltBCPlusRSAFDHVerifier(tampered_pres, cb_get_issuer_key=lambda i, h: pk)

    # 2. Genuine k-byte modular equivalent s' = s + n < 2^(8k) satisfies (s')^e == s^e (mod n),
    #    proving rsa_fdh_verify actively enforces 0 <= s < n rather than relying on mod n.
    found_equiv = False
    for counter in range(512):
        candidate_msg = f"range-check-{counter}".encode("ascii")
        sig = rsa_fdh_sign(sk, candidate_msg)
        s = os2ip(sig)
        if s + pk.n < upper_bound:
            s_equiv = s + pk.n
            assert pow(s_equiv, pk.e, pk.n) == pow(s, pk.e, pk.n)
            s_equiv_bytes = i2osp(s_equiv, k)
            assert len(s_equiv_bytes) == k
            assert rsa_fdh_verify(pk, candidate_msg, sig) is True
            assert rsa_fdh_verify(pk, candidate_msg, s_equiv_bytes) is False
            found_equiv = True
            break
    assert found_equiv


def test_rejects_malformed_or_corrupted_signature_and_tampered_alg(setup_parties_2048):
    sk, pk, user, state, req = setup_parties_2048
    issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    cred = user.fin_creds(state, issuer.rsp, pk)

    # Malformed rsp rejected by FinCreds
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, "not.a.bare_sig", pk)
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, "sig~with~tilde", pk)

    # Corrupted signature character rejected by FinCreds and VfPres
    bad_char = "A" if issuer.rsp[0] != "A" else "B"
    corrupted_rsp = bad_char + issuer.rsp[1:]
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, corrupted_rsp, pk)

    holder = SDJWTHolder(cred)
    holder.create_presentation({"given_name": True})
    tampered_pres = _replace_presentation_sig(holder.sd_jwt_presentation, corrupted_rsp)
    with pytest.raises(SaltBCPlusVerificationError):
        SaltBCPlusRSAFDHVerifier(tampered_pres, cb_get_issuer_key=lambda i, h: pk)

    # Tampered alg in JWS header rejected by VfPres even if validly signed under isk
    bad_hdr_b64 = b64url_encode(
        _canonical_json({"alg": "ES256", "typ": SaltBCPlusRSAFDHIssuer.SD_JWT_HEADER}).encode("utf-8")
    )
    payload_b64 = issuer.serialized_sd_jwt.split(".")[1]
    bad_alg_sig = b64url_encode(
        rsa_fdh_sign(sk, f"{bad_hdr_b64}.{payload_b64}".encode("ascii"))
    )
    parts = holder.sd_jwt_presentation.split("~")
    parts[0] = f"{bad_hdr_b64}.{payload_b64}.{bad_alg_sig}"
    with pytest.raises(SaltBCPlusVerificationError):
        SaltBCPlusRSAFDHVerifier("~".join(parts), cb_get_issuer_key=lambda i, h: pk)


def test_rejects_subverted_attribute_salt(setup_parties_2048):
    # Phase 1 regression: Salt Attack
    sk, pk, user, state, req = setup_parties_2048
    tampered_req = copy.deepcopy(req)
    tampered_req["attr_salts"]["given_name"] = "subverted_salt_123456"

    subverted_issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, tampered_req, issuer_keys=[sk])
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, subverted_issuer.rsp, pk)


def test_rejects_subverted_decoy_digest(setup_parties_2048):
    # Phase 1 regression: Decoy-Digest Attack
    sk, pk, user, state, req = setup_parties_2048
    tampered_req = copy.deepcopy(req)
    tampered_req["decoy_salts"][0] = "subverted_decoy_salt_999"

    subverted_issuer = SaltBCPlusRSAFDHIssuer(USER_CLAIMS, tampered_req, issuer_keys=[sk])
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, subverted_issuer.rsp, pk)


def test_rejects_subverted_digest_order(setup_parties_2048):
    # Phase 1 regression: Digest-Order Attack
    sk, pk, user, state, req = setup_parties_2048

    class PermutingIssuer(SaltBCPlusRSAFDHIssuer):
        def _create_signed_jws(self):
            sorted_sd = list(self.sd_jwt_payload["_sd"])
            permuted_sd = list(reversed(sorted_sd))
            assert permuted_sd != sorted_sd
            self.sd_jwt_payload["_sd"] = permuted_sd
            super()._create_signed_jws()

    subverted_issuer = PermutingIssuer(USER_CLAIMS, req, issuer_keys=[sk])
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, subverted_issuer.rsp, pk)
