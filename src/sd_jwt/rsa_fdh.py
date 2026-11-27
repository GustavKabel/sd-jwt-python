"""
    RSA-FDH implementation based on RFC 9381,
    provides a unique signature scheme primitive for saltBC+ phase 2

    Implements KGen, SIG.Sign, SIG.Vf, where rsa_fdh_sign corresponds to VRF_prove and rsa_fdh_verify corresponds to VRF_verify

    From RFC 9381 we use fixed k-byte length, 0 <= s < n, and deterministic MGF1-SHA256 hashing.

    Benchmark note:
        KGen and MGF1-SHA256 execute in OpenSSL C
        modular exponentiation in SIG.Sign and SIG.Vf executes in CPython's built-in pow().
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
import binascii
from dataclasses import dataclass, field
from hashlib import sha256
import hmac
from typing import Optional

from cryptography.hazmat.primitives.asymmetric import rsa

# RFC 9381 Section 4 constants for RSA-FDH-VRF-SHA256
SUITE_STRING: bytes = b"\x01"
MGF_DOMAIN_SEPARATOR: bytes = b"\x01"
DEFAULT_RSA_KEY_SIZE: int = 3072
DEFAULT_PUBLIC_EXPONENT: int = 65537

_SHA256_DIGEST_SIZE: int = 32


def i2osp(x: int, x_len: int) -> bytes:
    # RFC 8017 I2OSP convert non-negative integer x to x_len big-endian bytes.
    if not isinstance(x, int) or isinstance(x, bool):
        raise TypeError("x must be an integer")
    if not isinstance(x_len, int) or isinstance(x_len, bool) or x_len < 0:
        raise ValueError("x_len must be a non-negative integer")
    if x < 0 or x >= (1 << (8 * x_len)):
        raise ValueError(f"Integer out of range for I2OSP({x_len})")
    return x.to_bytes(x_len, "big")


def os2ip(octet_string: bytes) -> int:
    # RFC 8017 OS2IP convert big-endian octet string to non-negative integer.
    if not isinstance(octet_string, (bytes, bytearray)):
        raise TypeError("octet_string must be bytes")
    return int.from_bytes(octet_string, "big")



# Format functions to bride RSA-FDH with SD-JWT implementation
def b64url_encode(data: bytes) -> str:
    # Unpadded Base64url encoding
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    return urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(data: str) -> bytes:
    # Standard unpadded Base64url decoding
    if not isinstance(data, str):
        raise ValueError("Base64url input must be a string")
    rem = len(data) % 4
    if rem == 1:
        raise ValueError("Invalid Base64url length")
    padded = data + ("=" * ((4 - rem) % 4))
    try:
        return urlsafe_b64decode(padded.encode("ascii"))
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("Invalid Base64url encoding") from exc


#Masking function from RFC 8017
def mgf1_sha256(mgf_seed: bytes, mask_len: int) -> bytes:
    # RFC 8017 MGF1 instantiated with SHA-256.
    if not isinstance(mgf_seed, (bytes, bytearray)):
        raise TypeError("mgf_seed must be bytes")
    if mask_len < 0 or mask_len > (1 << 32) * _SHA256_DIGEST_SIZE:
        raise ValueError("mask too long")
    if mask_len == 0:
        return b""

    base_hash = sha256(mgf_seed)
    num_blocks = (mask_len + _SHA256_DIGEST_SIZE - 1) // _SHA256_DIGEST_SIZE
    chunks = []
    for counter in range(num_blocks):
        h = base_hash.copy()
        h.update(counter.to_bytes(4, "big"))
        chunks.append(h.digest())
    return b"".join(chunks)[:mask_len]


@dataclass(frozen=True)
class RSAFDHPublicKey:
    """
    Issuer public verification key ipk = (n, e).

    Precomputes the modulus byte length k and domain-separation
    prefix: 0x01 || 0x01 || I2OSP(k, 4) || I2OSP(n, k).
    """
    n: int
    e: int
    kid: Optional[str] = None
    modulus_len: int = field(init=False, repr=False, compare=False)
    mgf_prefix: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError("RSA modulus n must be a positive integer")
        if not isinstance(self.e, int) or self.e < 3 or self.e % 2 == 0:
            raise ValueError("RSA public exponent e must be an odd integer >= 3")
        k = (self.n.bit_length() + 7) // 8
        if k < 2:
            raise ValueError("RSA modulus byte length k must be >= 2")
        mgf_salt = i2osp(k, 4) + i2osp(self.n, k)
        mgf_prefix = SUITE_STRING + MGF_DOMAIN_SEPARATOR + mgf_salt
        object.__setattr__(self, "modulus_len", k)
        object.__setattr__(self, "mgf_prefix", mgf_prefix)

    def key_size(self) -> int:
        return self.n.bit_length()

    def __contains__(self, item: object) -> bool:
        return item == "kid" and self.kid is not None

    def __getitem__(self, item: str) -> str:
        if item == "kid" and self.kid is not None:
            return self.kid
        raise KeyError(item)


@dataclass(frozen=True)
class RSAFDHPrivateKey:
    """
    Issuer secret signing key isk = (n, e, d, p, q, dmp1, dmq1, iqmp) for RSA-FDH-VRF-SHA256.
    dmp1, dmq1, iqmp are chinese remainder theorem parameter such that
        dmp = d mod p
        dmq = d mod q
        iqmp = inverse q mod p

    Used to improve sign performance
    """

    n: int
    e: int
    d: int
    p: int
    q: int
    dmp1: int
    dmq1: int
    iqmp: int
    kid: Optional[str] = None
    _public_key: RSAFDHPublicKey = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.p * self.q != self.n:
            raise ValueError("Invalid RSA private key: p * q != n")
        object.__setattr__(
            self,
            "_public_key",
            RSAFDHPublicKey(n=self.n, e=self.e, kid=self.kid),
        )

    def public_key(self) -> RSAFDHPublicKey:
        return self._public_key

    def modulus_len(self) -> int:
        return self._public_key.modulus_len

    def key_size(self) -> int:
        return self._public_key.key_size()

    def __contains__(self, item: object) -> bool:
        return item == "kid" and self.kid is not None

    def __getitem__(self, item: str) -> str:
        if item == "kid" and self.kid is not None:
            return self.kid
        raise KeyError(item)

    @classmethod
    def from_cryptography_key(
        cls,
        key: rsa.RSAPrivateKey,
        kid: Optional[str] = None,
    ) -> "RSAFDHPrivateKey":
        priv_nums = key.private_numbers()
        pub_nums = priv_nums.public_numbers
        return cls(
            n=pub_nums.n,
            e=pub_nums.e,
            d=priv_nums.d,
            p=priv_nums.p,
            q=priv_nums.q,
            dmp1=priv_nums.dmp1,
            dmq1=priv_nums.dmq1,
            iqmp=priv_nums.iqmp,
            kid=kid,
        )

    @classmethod
    def generate(
        cls,
        key_size: int = DEFAULT_RSA_KEY_SIZE,
        public_exponent: int = DEFAULT_PUBLIC_EXPONENT,
        kid: Optional[str] = None,
    ) -> "RSAFDHPrivateKey":
        key = rsa.generate_private_key(
            public_exponent=public_exponent,
            key_size=key_size,
        )
        return cls.from_cryptography_key(key, kid=kid)


def rsa_fdh_sign(private_key: RSAFDHPrivateKey, message: bytes) -> bytes:

    #Computes EM and uses CRT accelerated sign according to RFC 9381
    if not isinstance(private_key, RSAFDHPrivateKey):
        raise TypeError("private_key must be an RSAFDHPrivateKey")
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")

    k = private_key.modulus_len()
    em = mgf1_sha256(private_key.public_key().mgf_prefix + bytes(message), k - 1)
    m = os2ip(em)

    # CRT-accelerated RSASP1: s = m^d mod n with 0 <= s < n
    s1 = pow(m % private_key.p, private_key.dmp1, private_key.p)
    s2 = pow(m % private_key.q, private_key.dmq1, private_key.q)
    h = (private_key.iqmp * (s1 - s2)) % private_key.p
    s = s2 + h * private_key.q

    return i2osp(s, k)


def rsa_fdh_verify(
    public_key: RSAFDHPublicKey,
    message: bytes,
    signature: bytes,
) -> bool:

    # verifies a signature according to RFC 9381
    if not isinstance(public_key, RSAFDHPublicKey):
        raise TypeError("public_key must be an RSAFDHPublicKey")
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    if not isinstance(signature, (bytes, bytearray)):
        return False

    k = public_key.modulus_len
    if len(signature) != k:
        return False

    s = os2ip(signature)
    if not (0 <= s < public_key.n):
        return False

    m = pow(s, public_key.e, public_key.n)
    try:
        em = i2osp(m, k - 1)
    except ValueError:
        return False

    expected_em = mgf1_sha256(public_key.mgf_prefix + bytes(message), k - 1)
    return hmac.compare_digest(em, expected_em)
