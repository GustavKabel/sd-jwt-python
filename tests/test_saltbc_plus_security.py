import copy
import json

import pytest
from jwcrypto import jwk, jws

from sd_jwt.common import SDObj
from sd_jwt.holder import SDJWTHolder
from sd_jwt.saltbc_plus import (
    SaltBCPlusIssuer,
    SaltBCPlusUser,
    SaltBCPlusVerificationError,
    _canonical_json,
)
from sd_jwt.verifier import SDJWTVerifier

MAX_ATTRIBUTES = 10

USER_CLAIMS = {
    SDObj("given_name"): "Julian",
    SDObj("family_name"): "Doe",
    SDObj("nationality"): "Danish",
    SDObj("birthdate"): "1990-01-01",
    SDObj("age_over_18"): True,
}

@pytest.fixture
def setup_parties():
    # Run KGen, ReqCreds to prep (isk, ipk, user, st, req)
    sk = jwk.JWK.generate(kty="EC", crv="P-256")
    pk = jwk.JWK(**sk.export(private_key=False, as_dict=True))
    user = SaltBCPlusUser()
    state, req = user.req_creds(USER_CLAIMS, max_attributes=MAX_ATTRIBUTES)
    return sk, pk, user, state, req

def test_honest_partial_and_full_disclosure(setup_parties):
    # Honest issue -> FinCreds -> Present -> VfPres for parital and full disclosure
    sk, pk, user, state, req = setup_parties
    issuer = SaltBCPlusIssuer(USER_CLAIMS, req, issuer_keys=[sk], sign_alg="ES256")

    assert len(issuer.sd_jwt_payload["_sd"]) == MAX_ATTRIBUTES
    assert issuer.sd_jwt_issuance is None

    finalized_cred = user.fin_creds(state, issuer.rsp, pk)

    # Partial Disclosure
    holder = SDJWTHolder(finalized_cred)
    holder.create_presentation({"given_name": True})
    verifier = SDJWTVerifier(
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
    verifier_full = SDJWTVerifier(
        holder_full.sd_jwt_presentation,
        cb_get_issuer_key=lambda iss, hdr: pk,
        expected_aud=None,
        expected_nonce=None,
    )
    full = verifier_full.get_verified_payload()
    assert full["given_name"] == "Julian"
    assert full["nationality"] == "Danish"
    assert full["age_over_18"] is True

def test_rejects_subverted_attribute_salt(setup_parties):
    # Salt Attack
    sk, pk, user, state, req = setup_parties
    tampered_req = copy.deepcopy(req)
    tampered_req["attr_salts"]["given_name"] = "subverted_salt_123456"

    subverted_issuer = SaltBCPlusIssuer(USER_CLAIMS, tampered_req, issuer_keys=[sk], sign_alg="ES256")
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, subverted_issuer.rsp, pk)

def test_rejects_subverted_decoy_digest(setup_parties):
    # Decoy-Digest Attack
    sk, pk, user, state, req = setup_parties
    tampered_req = copy.deepcopy(req)
    tampered_req["decoy_salts"][0] = "subverted_decoy_salt_999"

    subverted_issuer = SaltBCPlusIssuer(USER_CLAIMS, tampered_req, issuer_keys=[sk], sign_alg="ES256")
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, subverted_issuer.rsp, pk)

def test_rejects_subverted_digest_order(setup_parties):
    # Digest-Order Attack
    sk, pk, user, state, req = setup_parties

    class PermutingIssuer(SaltBCPlusIssuer):
        def _create_signed_jws(self):
            sorted_sd = list(self.sd_jwt_payload["_sd"])
            permuted_sd = list(reversed(sorted_sd))
            assert permuted_sd != sorted_sd
            self.sd_jwt_payload["_sd"] = permuted_sd
            super()._create_signed_jws()

    subverted_issuer = PermutingIssuer(USER_CLAIMS, req, issuer_keys=[sk], sign_alg="ES256")
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, subverted_issuer.rsp, pk)

def test_rejects_non_canonical_json_formatting(setup_parties):
    # Encoding side-channel
    sk, pk, user, state, req = setup_parties
    honest_issuer = SaltBCPlusIssuer(USER_CLAIMS, req, issuer_keys=[sk], sign_alg="ES256")

    protected_str = _canonical_json({"alg": "ES256", "typ": SaltBCPlusIssuer.SD_JWT_HEADER})

    non_canonical_payload = json.dumps(honest_issuer.sd_jwt_payload, sort_keys=True)
    token = jws.JWS(payload=non_canonical_payload.encode("utf-8"))
    token.add_signature(sk, alg="ES256", protected=protected_str)
    tampered_sig = token.serialize(compact=True).split(".")[2]

    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, tampered_sig, pk)

def test_rejects_malformed_or_corrupted_signature(setup_parties):
    # FinCreds aborts on malformed rsp or invalid signature
    sk, pk, user, state, req = setup_parties
    issuer = SaltBCPlusIssuer(USER_CLAIMS, req, issuer_keys=[sk], sign_alg="ES256")

    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, "not.a.bare_sig", pk)

    bad_char = "A" if issuer.rsp[0] != "A" else "B"
    corrupted_sig = bad_char + issuer.rsp[1:]
    with pytest.raises(SaltBCPlusVerificationError):
        user.fin_creds(state, corrupted_sig, pk)