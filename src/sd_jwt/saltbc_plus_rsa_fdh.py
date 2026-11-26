"""
Phase 2 saltBC+ (sdjwtBC+) instantiation with RFC 9381 RSA-FDH unique signatures.

Thin subclasses over Phase 1's SaltBCPlusIssuer / SaltBCPlusUser and stock
SDJWTVerifier that swap ECDSA (ES256) for deterministic, unique RSA-FDH
signatures while reusing 100% of Phase 1's salt, decoy, alphanumeric digest
ordering, and canonical JSON defenses.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Union

from sd_jwt.rsa_fdh import (
    RSAFDHPrivateKey,
    RSAFDHPublicKey,
    b64url_decode,
    b64url_encode,
    rsa_fdh_sign,
    rsa_fdh_verify,
)
from sd_jwt.saltbc_plus import (
    SaltBCPlusIssuer,
    SaltBCPlusUser,
    SaltBCPlusVerificationError,
)
from sd_jwt.verifier import SDJWTVerifier

RSA_FDH_SIGN_ALG: str = "RSA-FDH-SHA256"


class SaltBCPlusRSAFDHIssuer(SaltBCPlusIssuer):
    """
    saltBC+ Issuer (Issue) instantiated with RFC 9381 RSA-FDH unique signatures.

    Reuses Phase 1's deterministic SaltedHash(A, req), alphanumeric Order(H),
    canonical JSON serialization, and rsp-only output, replacing SIG.Sign(isk, pl)
    with deterministic RSA-FDH signing.
    """

    def __init__(
        self,
        user_claims: Dict[Any, Any],
        req: Dict[str, Any],
        issuer_keys: Union[RSAFDHPrivateKey, List[RSAFDHPrivateKey]],
        holder_key=None,
        sign_alg: str = RSA_FDH_SIGN_ALG,
        serialization_format: str = "compact",
        extra_header_parameters: Optional[Dict[str, Any]] = None,
    ):
        keys_list = issuer_keys if isinstance(issuer_keys, list) else [issuer_keys]
        if len(keys_list) != 1 or not isinstance(keys_list[0], RSAFDHPrivateKey):
            raise ValueError(
                "SaltBCPlusRSAFDHIssuer requires a single RSAFDHPrivateKey signing key isk"
            )

        super().__init__(
            user_claims=user_claims,
            req=req,
            issuer_keys=keys_list,
            holder_key=holder_key,
            sign_alg=sign_alg,
            serialization_format=serialization_format,
            extra_header_parameters=extra_header_parameters,
        )

    def _sign_compact_jws(self, protected_str: str, payload_str: str) -> str:
        """Compute deterministic unique signature sigma = SIG.Sign(isk, pl) over JWS signing input."""
        protected_b64 = b64url_encode(protected_str.encode("utf-8"))
        payload_b64 = b64url_encode(payload_str.encode("utf-8"))
        signing_input = f"{protected_b64}.{payload_b64}".encode("ascii")

        sig_bytes = rsa_fdh_sign(self._issuer_keys[0], signing_input)
        sig_b64 = b64url_encode(sig_bytes)

        self.sd_jwt = None
        return f"{protected_b64}.{payload_b64}.{sig_b64}"


class SaltBCPlusRSAFDHUser(SaltBCPlusUser):
    """
    saltBC+ User algorithms ReqCreds(ipk, A) -> (st, req) and FinCreds(st, rsp) -> cred
    instantiated with RFC 9381 RSA-FDH unique signature verification.
    """

    def __init__(self, sign_alg: str = RSA_FDH_SIGN_ALG):
        super().__init__(sign_alg=sign_alg)

    def _verify_compact_jws(
        self,
        compact_jws: str,
        issuer_public_key: RSAFDHPublicKey,
    ) -> None:
        """Compute SIG.Vf(ipk, pl, sigma) using RFC 9381 RSA-FDH verification."""
        if not isinstance(issuer_public_key, RSAFDHPublicKey):
            raise SaltBCPlusVerificationError(
                "FinCreds aborted: issuer_public_key must be an RSAFDHPublicKey"
            )

        parts = compact_jws.split(".")
        if len(parts) != 3 or not all(parts):
            raise SaltBCPlusVerificationError(
                "FinCreds aborted: malformed compact JWS"
            )
        protected_b64, payload_b64, sig_b64 = parts

        try:
            sig_bytes = b64url_decode(sig_b64)
        except ValueError as exc:
            raise SaltBCPlusVerificationError(
                f"FinCreds aborted: malformed signature encoding ({exc})"
            ) from exc

        signing_input = f"{protected_b64}.{payload_b64}".encode("ascii")
        if not rsa_fdh_verify(issuer_public_key, signing_input, sig_bytes):
            raise SaltBCPlusVerificationError(
                "FinCreds aborted: RSA-FDH signature invalid for the expected payload"
            )


class SaltBCPlusRSAFDHVerifier(SDJWTVerifier):
    """
    saltBC+ Verifier (VfPres) for presentations carrying RFC 9381 RSA-FDH signatures.

    Verifies JWS header alg, RSA-FDH signature SIG.Vf(ipk, pl, sigma), and delegates
    selective-disclosure unpacking to SDJWTVerifier.
    """

    def __init__(
        self,
        sd_jwt_presentation: str,
        cb_get_issuer_key: Callable[[Optional[str], Dict[str, Any]], RSAFDHPublicKey],
        expected_aud: Optional[str] = None,
        expected_nonce: Optional[str] = None,
        serialization_format: str = "compact",
        sign_alg: str = RSA_FDH_SIGN_ALG,
    ):
        if serialization_format != "compact":
            raise ValueError("SaltBCPlusRSAFDHVerifier only supports compact serialization.")
        if expected_aud is not None or expected_nonce is not None:
            raise ValueError("Holder key binding is out of scope for saltBC+")

        self._expected_sign_alg = sign_alg
        super().__init__(
            sd_jwt_presentation=sd_jwt_presentation,
            cb_get_issuer_key=cb_get_issuer_key,
            expected_aud=None,
            expected_nonce=None,
            serialization_format="compact",
        )

    def _verify_sd_jwt(
        self,
        cb_get_issuer_key: Callable[[Optional[str], Dict[str, Any]], RSAFDHPublicKey],
        sign_alg: Optional[str] = None,
    ) -> None:
        expected_alg = sign_alg or self._expected_sign_alg

        parts = self._unverified_input_sd_jwt.split(".")
        if len(parts) != 3 or not all(parts):
            raise SaltBCPlusVerificationError(
                "VfPres aborted: malformed compact JWS in presentation"
            )
        protected_b64, payload_b64, sig_b64 = parts

        try:
            protected_bytes = b64url_decode(protected_b64)
            payload_bytes = b64url_decode(payload_b64)
            sig_bytes = b64url_decode(sig_b64)
        except ValueError as exc:
            raise SaltBCPlusVerificationError(
                f"VfPres aborted: malformed Base64url encoding ({exc})"
            ) from exc

        try:
            unverified_header_parameters = json.loads(protected_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SaltBCPlusVerificationError(
                "VfPres aborted: malformed protected header JSON"
            ) from exc

        if not isinstance(unverified_header_parameters, dict):
            raise SaltBCPlusVerificationError(
                "VfPres aborted: protected header must be a JSON object"
            )
        if unverified_header_parameters.get("alg") != expected_alg:
            raise SaltBCPlusVerificationError(
                f"VfPres aborted: unexpected header alg {unverified_header_parameters.get('alg')!r}"
            )

        unverified_issuer = self._unverified_input_sd_jwt_payload.get("iss", None)
        issuer_public_key = cb_get_issuer_key(
            unverified_issuer,
            unverified_header_parameters,
        )
        if not isinstance(issuer_public_key, RSAFDHPublicKey):
            raise SaltBCPlusVerificationError(
                "VfPres aborted: cb_get_issuer_key must return an RSAFDHPublicKey"
            )

        signing_input = f"{protected_b64}.{payload_b64}".encode("ascii")
        if not rsa_fdh_verify(issuer_public_key, signing_input, sig_bytes):
            raise SaltBCPlusVerificationError(
                "VfPres aborted: invalid RSA-FDH signature"
            )

        try:
            self._sd_jwt_payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SaltBCPlusVerificationError(
                "VfPres aborted: malformed payload JSON"
            ) from exc

        self._holder_public_key_payload = self._sd_jwt_payload.get("cnf", None)
