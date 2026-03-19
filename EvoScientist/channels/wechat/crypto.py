"""WeChat / WeCom crypto helpers.

Implements the message encryption/decryption protocol used by both
WeCom (企业微信) and WeChat Official Account (公众号) callback APIs.

The protocol uses AES-256-CBC with a key derived from the EncodingAESKey
(base64-encoded 43-char string → 32-byte AES key).

References:
  - WeCom: https://developer.work.weixin.qq.com/document/path/90930
  - MP: https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Message_Encryption_and_Decryption_Instructions.html
"""

import base64
import hashlib
import struct
import time
import xml.etree.ElementTree as ET

# Crypto imports — all from the Python standard library + pycryptodome
# (but we'll use a pure-Python fallback if not available)
try:
    from Crypto.Cipher import AES

    _HAS_PYCRYPTO = True
except ImportError:
    _HAS_PYCRYPTO = False


def _pkcs7_pad(data: bytes, block_size: int = 32) -> bytes:
    """PKCS#7 padding."""
    pad_len = block_size - (len(data) % block_size)
    return data + bytes([pad_len]) * pad_len


def _pkcs7_unpad(data: bytes) -> bytes:
    """PKCS#7 unpadding."""
    pad_len = data[-1]
    if pad_len < 1 or pad_len > 32:
        return data
    return data[:-pad_len]


def _aes_decrypt(key: bytes, iv: bytes, ciphertext: bytes) -> bytes:
    """AES-256-CBC decryption."""
    if _HAS_PYCRYPTO:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.decrypt(ciphertext)
    else:
        # Pure-Python AES fallback (slower but no C deps)
        # We'll try pyaes as a fallback
        try:
            import pyaes

            decrypter = pyaes.Decrypter(pyaes.AESModeOfOperationCBC(key, iv=iv))
            decrypted = decrypter.feed(ciphertext)
            decrypted += decrypter.feed()
            return decrypted
        except ImportError:
            raise ImportError(
                "WeChat message decryption requires pycryptodome or pyaes. "
                "Install with: pip install pycryptodome"
            ) from None


def _aes_encrypt(key: bytes, iv: bytes, plaintext: bytes) -> bytes:
    """AES-256-CBC encryption."""
    if _HAS_PYCRYPTO:
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(plaintext)
    else:
        try:
            import pyaes

            encrypter = pyaes.Encrypter(pyaes.AESModeOfOperationCBC(key, iv=iv))
            encrypted = encrypter.feed(plaintext)
            encrypted += encrypter.feed()
            return encrypted
        except ImportError:
            raise ImportError(
                "WeChat message encryption requires pycryptodome or pyaes. "
                "Install with: pip install pycryptodome"
            ) from None


class WeChatCrypto:
    """Handles WeChat/WeCom message encryption and decryption.

    Parameters
    ----------
    token:
        The Token configured in the WeChat/WeCom callback URL settings.
    encoding_aes_key:
        The 43-character EncodingAESKey (base64-encoded).
    app_id:
        The AppID (for MP) or CorpID (for WeCom).
    """

    def __init__(self, token: str, encoding_aes_key: str, app_id: str):
        self.token = token
        self.app_id = app_id
        # Decode the AES key: EncodingAESKey + "=" → base64 decode → 32 bytes
        self.aes_key = base64.b64decode(encoding_aes_key + "=")
        # IV is the first 16 bytes of the key
        self.iv = self.aes_key[:16]

    def verify_signature(
        self,
        signature: str,
        timestamp: str,
        nonce: str,
        encrypt: str = "",
    ) -> bool:
        """Verify the callback signature.

        For plain-mode verification (no encryption), *encrypt* can be empty.
        """
        parts = sorted([self.token, timestamp, nonce] + ([encrypt] if encrypt else []))
        sha1 = hashlib.sha1("".join(parts).encode()).hexdigest()
        return sha1 == signature

    def decrypt(self, encrypt: str) -> tuple[str, str]:
        """Decrypt an encrypted message.

        Returns ``(xml_content, from_app_id)`` tuple.
        """
        ciphertext = base64.b64decode(encrypt)
        plaintext = _aes_decrypt(self.aes_key, self.iv, ciphertext)
        plaintext = _pkcs7_unpad(plaintext)

        # plaintext layout:
        # 16 bytes random + 4 bytes msg_len (big-endian) + msg + app_id
        msg_len = struct.unpack("!I", plaintext[16:20])[0]
        msg = plaintext[20 : 20 + msg_len].decode("utf-8")
        from_app_id = plaintext[20 + msg_len :].decode("utf-8")
        return msg, from_app_id

    def encrypt(self, reply_msg: str) -> str:
        """Encrypt a reply message.

        Returns the base64-encoded ciphertext.
        """
        msg_bytes = reply_msg.encode("utf-8")
        app_id_bytes = self.app_id.encode("utf-8")

        # Random 16 bytes + msg_len (4 bytes big-endian) + msg + app_id
        import os

        random_bytes = os.urandom(16)
        msg_len = struct.pack("!I", len(msg_bytes))
        plaintext = random_bytes + msg_len + msg_bytes + app_id_bytes
        plaintext = _pkcs7_pad(plaintext)

        ciphertext = _aes_encrypt(self.aes_key, self.iv, plaintext)
        return base64.b64encode(ciphertext).decode("utf-8")

    def generate_signature(
        self,
        encrypt: str,
        timestamp: str,
        nonce: str,
    ) -> str:
        """Generate the msg_signature for an encrypted reply."""
        parts = sorted([self.token, timestamp, nonce, encrypt])
        return hashlib.sha1("".join(parts).encode()).hexdigest()

    def wrap_encrypted_reply(self, reply_msg: str) -> str:
        """Encrypt a reply and wrap it in the XML envelope.

        Returns the full XML string to return in the HTTP response.
        """
        encrypt = self.encrypt(reply_msg)
        timestamp = str(int(time.time()))
        nonce = hashlib.md5(str(time.time()).encode()).hexdigest()[:10]
        signature = self.generate_signature(encrypt, timestamp, nonce)

        return (
            f"<xml>"
            f"<Encrypt><![CDATA[{encrypt}]]></Encrypt>"
            f"<MsgSignature><![CDATA[{signature}]]></MsgSignature>"
            f"<TimeStamp>{timestamp}</TimeStamp>"
            f"<Nonce><![CDATA[{nonce}]]></Nonce>"
            f"</xml>"
        )


def parse_xml(xml_str: str) -> dict[str, str]:
    """Parse a WeChat callback XML into a flat dict."""
    root = ET.fromstring(xml_str)
    result = {}
    for child in root:
        result[child.tag] = child.text or ""
    return result
