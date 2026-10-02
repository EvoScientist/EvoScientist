"""AES-256-GCM utilities for QQ Bot scan-to-configure credential decryption.

Ported from hermes-agent/gateway/platforms/qqbot/crypto.py — the q.qq.com
``create_bind_task`` / ``poll_bind_result`` flow uses AES-256-GCM to keep
the bot's *client_secret* off the wire in plaintext.
"""

from __future__ import annotations

import base64
import os


def generate_bind_key() -> str:
    """Generate a 256-bit random AES key, base64-encoded.

    The key is sent to ``create_bind_task`` so the server can encrypt
    the bot's *client_secret* before returning it.  Only this client
    holds the key, so the secret never travels in plaintext.
    """
    return base64.b64encode(os.urandom(32)).decode()


def decrypt_secret(encrypted_base64: str, key_base64: str) -> str:
    """Decrypt a base64-encoded AES-256-GCM ciphertext.

    Ciphertext layout (after base64-decoding)::

        IV (12 bytes) ‖ ciphertext (N bytes) ‖ AuthTag (16 bytes)

    Args:
        encrypted_base64: The ``bot_encrypt_secret`` value returned by
            ``poll_bind_result``.
        key_base64: The base64 AES key produced by :func:`generate_bind_key`.

    Returns:
        The decrypted *client_secret* as a UTF-8 string.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = base64.b64decode(key_base64)
    raw = base64.b64decode(encrypted_base64)

    iv = raw[:12]
    ciphertext_with_tag = raw[12:]  # AESGCM expects ciphertext + tag concatenated

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext_with_tag, None)
    return plaintext.decode("utf-8")
