import json
from typing import Any, Dict, Iterator, List, Optional, Tuple,Union

from jwcrypto import jwk, jws
from jwcrypto.common import JWException

from sd_jwt.common import SD_DIGESTS_KEY, DIGEST_ALG_KEY, SDJWTCommon, SDObj
from sd_jwt.disclosure import SDJWTDisclosure
from sd_jwt.issuer import SDJWTIssuer

class SaltBCPlusVerificationError(ValueError):
    """
    Error raised by FinCreds if FinCreds aborts, this could be because
    Issuer response does not match credential determined by the user 's state.
    e.g. substituted attribute, decoy salt , permuted digest order , tampered payload/header or invalid signature
    """

def _canonical_json(obj: Dict[str, Any]) -> str:
    # Deterministic JSON encoding with sorted keys, and no extra whitespace
    return json.dumps(obj, sort_keys = True , separators=(',', ':'))

def _validate_flat_claims(user_claims: Dict[Any, Any]) -> None:
    # Like in the paper, we do not consider nested attributes, so all claims must be flat (thus this models more accurately saltBC)
    if not isinstance(user_claims, dict):
        raise TypeError("user_claims must be a dictionary")
    for value in user_claims.values():
        if isinstance(value, SDObj):
            raise ValueError("SDObj must wrap claim names (keys) not values.")
        if isinstance(value, dict) and any(isinstance(k, SDObj) for k in value):
            raise ValueError("Nested SDObj claims are out of scope in flat saltBC+")
        if isinstance(value, list) and any(isinstance(item, SDObj) for item in value):
            raise ValueError("Selectively disclosable array elements out of scope in flat saltBC+")

class SaltBCPlusIssuer(SDJWTIssuer):
    """
    Implementation of saltBC+ Issuer, as seen in Fig. 7 + SD-JWT safeguards from Section 8.2.

    Runs SaltedHash on attributes sent by the User, using the salts received in "req", thus no Issuer randomness.
    Applies alphanumeric ordering to H to form pl
    Sign pl, under isk to get signatures and outputs the signatures
    """

    def __init__(
        self,
        user_claims: Dict[Any, Any],
        req: Dict[str, Any],
        issuer_keys: Union[jwk.JWK, List[jwk.JWK]],
        holder_key=None,
        sign_alg: str = "ES256",
        serialization_format: str = "compact",
        extra_header_parameters: Optional[Dict[str, Any]] = None,
    ):
        if serialization_format != "compact":
            raise ValueError("SaltBCPlusIssuer only supports compact serialization.")
        if holder_key is not None:
            raise ValueError("Holder key binding is out of scope for saltBC+")
        _validate_flat_claims(user_claims)

        keys_list = issuer_keys if isinstance(issuer_keys, list) else [issuer_keys]
        if len(keys_list) != 1:
            raise ValueError("SaltBCPlusIssuer requires a single issuer signing key isk")

        attr_salts: Dict[str, str] = req["attr_salts"]
        decoy_salts: List[str] = req["decoy_salts"]

        sd_keys = [k.value for k in user_claims if isinstance(k, SDObj)]
        if set(attr_salts.keys()) != set(sd_keys):
            raise ValueError("req['attr_salts'] must match the attribute set A")

        # Order user salts to match SaltedHash: |A| attributes salts followed by (n - |A|) decoy salts
        ordered_salts = [attr_salts[name] for name in sd_keys] + list(decoy_salts)
        if not all(isinstance(s, str) and s for s in ordered_salts):
            raise ValueError("All user-supplied salts in req must be non-empty strings.")

        self._ordered_salts: List[str] = ordered_salts
        self._salt_iter: Iterator[str] = iter(self._ordered_salts)
        self._in_top_level_object: bool = False

        # Pad credential digest to maximum length n using decoy salts

        num_decoys = len(decoy_salts)
        self.DECOY_MIN_ELEMENTS = num_decoys
        self.DECOY_MAX_ELEMENTS = num_decoys

        super().__init__(
            user_claims = user_claims,
            issuer_keys = keys_list,
            holder_key = None,
            sign_alg = sign_alg,
            add_decoy_claims = (num_decoys > 0),
            serialization_format = "compact",
            extra_header_parameters = extra_header_parameters or {},
        )

    def _reset_salt_iter(self) -> None:
        # Reset the salt stream, used in benchmarking
        self._salt_iter = iter(self._ordered_salts)

    def _generate_salt(self) -> str:
        # Supply next salt from req
        try:
            return next(self._salt_iter)
        except StopIteration as exc:
            raise ValueError("Insufficient user-supplied salts in req") from exc

    def _create_sd_claims_object(self, user_claims: Dict[Any, Any]) -> Dict[str, Any]:
        # Compute H = {H(s_j || a_j)} U {H(s_{|A|+j})} and apply alphanumeric Order(H).
        is_top_level = not self._in_top_level_object
        self._in_top_level_object = True
        orig_flag = self._add_decoy_claims
        self._add_decoy_claims = orig_flag and is_top_level
        try:
            return super()._create_sd_claims_object(user_claims)
        finally:
            self._add_decoy_claims = orig_flag
            if is_top_level:
                self._in_top_level_object = False

    def _sign_compact_jws(self, protected_str: str, payload_str: str) -> str:
        # Compute SIG.Sign(isk, pl) with ECDSA
        token = jws.JWS(payload = payload_str.encode("utf-8"))
        token.add_signature(
            self._issuer_keys[0],
            alg = self._sign_alg,
            protected = protected_str,
        )
        self.sd_jwt = token
        return token.serialize(compact = True)

    def _create_signed_jws(self) -> None:
        # Sign payload pl and expose rsp
        protected_headers: Dict[str, Any] ={
            "alg": self._sign_alg,
            "typ": self.SD_JWT_HEADER,
        }
        if "kid" in self._issuer_keys[0]:
            protected_headers["kid"] = self._issuer_keys[0]["kid"]
        protected_headers.update(self._extra_header_parameters)

        protected_str = _canonical_json(protected_headers)
        payload_str = _canonical_json(self.sd_jwt_payload)

        self.serialized_sd_jwt = self._sign_compact_jws(protected_str, payload_str)
        self.signature_b64 = self.serialized_sd_jwt.split(".")[2]
        self.rsp = self.signature_b64

    def _create_combined(self) -> None :
        # saltBC+ issuer outputs rsp only, user builds cred in FinCreds.
        self.sd_jwt_issuance = None

class SaltBCPlusUser(SDJWTCommon):
    """
    Implementation of saltBC+ User algorithms
    ReqCreds(ipk, A) -> (st, req) and FinCreds(st, rsp) -> cred
    """

    def __init__(self, sign_alg: str = "ES256"):
        super().__init__(serialization_format="compact")
        self._sign_alg = sign_alg
        self._salt_iter: Optional[Iterator[str]] = None

    def _generate_salt(self) -> str:
        # return stored salt for reconstruction in FinCreds
        if self._salt_iter is not None:
            return next(self._salt_iter)
        return super()._generate_salt()

    def req_creds(
            self,
            user_claims: Dict[Any, Any],
            max_attributes: int,
    ) -> Tuple[Dict[str, Any ], Dict[str, Any]]:
        """
        Samples n fresh lambda bit salts
        saves state st = (A, req) locally and outputs "req" to be sent to the issuer
        """
        _validate_flat_claims(user_claims)
        sd_keys = [k.value for k in user_claims if isinstance(k, SDObj)]
        if max_attributes < len(sd_keys):
            raise ValueError(f"max_attributes n ({max_attributes}) must be >= |A| ({len(sd_keys)})")

        sample_salt = super()._generate_salt
        attr_salts = {name: sample_salt() for name in sd_keys}
        num_decoys = max_attributes - len(sd_keys)
        decoy_salts = [sample_salt() for _ in range(num_decoys)]

        req ={
            "attr_salts": attr_salts,
            "decoy_salts": decoy_salts,
        }
        state = {
            "user_claims": user_claims,
            "attr_salts": attr_salts,
            "decoy_salts": decoy_salts,
            "max_attributes": max_attributes,
        }
        return state, req

    def _verify_compact_jws(self, compact_jws: str, issuer_public_key: jwk.JWK) -> None:
        # Compute SIG.Vf(ipk, pl, sigma)
        try:
            token = jws.JWS()
            token.deserialize(compact_jws)
            token.verify(issuer_public_key, alg = self._sign_alg)
        except (JWException, ValueError) as exc:
            raise SaltBCPlusVerificationError("FinCreds aborted: signature invalid for the expected payload") from exc

    def fin_creds(
            self,
            state: Dict[str, Any],
            rsp: str ,
            issuer_public_key: jwk.JWK,
            extra_header_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Locally computes SaltedHash(A, req) and alphanumeric order from state
        reconstructs exact payload "pl" and openings "S"
        Verifies SIG.Vf(ipk, pl, sigma) = 1 against rsp, aborts if any check fails
        """
        if not isinstance(rsp, str) or not rsp or "." in rsp or "~" in rsp:
            raise SaltBCPlusVerificationError("FinCreds aborted: malformed signature response rsp")

        user_claims = state["user_claims"]
        attr_salts: Dict[str, str] = state["attr_salts"]
        decoy_salts: List[str] = state["decoy_salts"]

        sd_item = [(k.value, v) for k, v in user_claims.items() if isinstance(k, SDObj)]
        expected_payload: Dict[str, Any] = {k: v for k, v in user_claims.items() if isinstance(k, str)}

        # Reconstruct openings S = {(s_j, a_j)} and digest H(s_j || a_j)
        disclosure_b64: List[str] = []
        expected_hashes: List[str] = []
        self._salt_iter = iter(attr_salts[name] for name, _ in sd_item)
        try:
            for name, value in sd_item:
                disclosure = SDJWTDisclosure(self, key=name, value=value)
                disclosure_b64.append(disclosure.b64)
                expected_hashes.append(disclosure.hash)
        finally:
            self._salt_iter = None

        for decoy_salt in decoy_salts:
            expected_hashes.append(self._b64hash(decoy_salt.encode("ascii")))

        # Enforce alphanumeric ordering
        if expected_hashes:
            expected_hashes.sort()
            expected_payload[SD_DIGESTS_KEY] = expected_hashes

        expected_payload[DIGEST_ALG_KEY] = self.HASH_ALG["name"]

        expected_header: Dict[str, Any] = {
            "alg": self._sign_alg,
            "typ": SDJWTIssuer.SD_JWT_HEADER,
        }
        if "kid" in issuer_public_key:
            expected_header["kid"] = issuer_public_key["kid"]
        if extra_header_parameters:
            expected_header.update(extra_header_parameters)

        protected_b64 = self._base64url_encode(_canonical_json(expected_header).encode("utf-8"))
        payload_b64 = self._base64url_encode(_canonical_json(expected_payload).encode("utf-8"))
        reconstructed_jws = f"{protected_b64}.{payload_b64}.{rsp}"

        self._verify_compact_jws(reconstructed_jws, issuer_public_key)

        return self._combine(reconstructed_jws, *disclosure_b64, "")