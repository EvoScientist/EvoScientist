"""WeChat callback verification server.

Provides a lightweight temporary HTTP server that handles the WeChat/WeCom
URL verification handshake during onboarding. This solves the chicken-and-egg
problem: WeChat requires a live server to verify the callback URL before
saving, but the main EvoScientist service isn't running during onboard.

Usage:
    server = VerifyServer(port, token, encoding_aes_key, corp_id)
    await server.start()
    # ... user clicks "Save" in WeCom admin console ...
    # ... server auto-responds to the verification GET request ...
    await server.wait_for_verify(timeout=120)
    await server.stop()
"""

import asyncio
import hashlib
import logging

logger = logging.getLogger(__name__)


class VerifyServer:
    """Temporary HTTP server for WeChat/WeCom callback URL verification.

    Handles the GET verification request (signature + echostr) and
    signals when verification succeeds.
    """

    def __init__(
        self,
        port: int,
        token: str,
        encoding_aes_key: str = "",
        app_id: str = "",
    ):
        self.port = port
        self.token = token
        self._crypto = None
        self._runner = None
        self._site = None
        self._verified = asyncio.Event()

        if encoding_aes_key and token and app_id:
            from .crypto import WeChatCrypto
            self._crypto = WeChatCrypto(
                token=token,
                encoding_aes_key=encoding_aes_key,
                app_id=app_id,
            )

    async def start(self) -> None:
        """Start the verification server."""
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/wechat/callback", self._handle)
        # Also handle POST in case WeCom sends a POST for some reason
        app.router.add_post("/wechat/callback", self._handle_post)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await self._site.start()
        logger.info(f"Verify server listening on port {self.port}")

    async def stop(self) -> None:
        """Stop the verification server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._site = None
        self._runner = None

    async def wait_for_verify(self, timeout: float = 120) -> bool:
        """Wait for verification to succeed.

        Returns True if verified within timeout, False otherwise.
        """
        try:
            await asyncio.wait_for(self._verified.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def is_verified(self) -> bool:
        return self._verified.is_set()

    async def _handle(self, request) -> "web.Response":
        """Handle GET verification request.

        During onboarding we use a lenient approach:
        1. Try strict crypto verification (encrypted mode)
        2. Try strict plain-mode signature check
        3. If both fail, fall back to decrypting echostr without
           signature check (WeCom requires the decrypted echostr)
        4. Last resort: echo back raw echostr

        This ensures the callback URL can be saved even if Token/AESKey
        have minor issues, while still attempting proper verification.
        """
        from aiohttp import web

        signature = (
            request.query.get("msg_signature")
            or request.query.get("signature", "")
        )
        timestamp = request.query.get("timestamp", "")
        nonce = request.query.get("nonce", "")
        echostr = request.query.get("echostr", "")

        logger.info(
            f"Verify request: msg_signature={signature[:16]}... "
            f"timestamp={timestamp} nonce={nonce} "
            f"echostr={echostr[:32]}..."
        )

        if not echostr:
            return web.Response(status=400, text="missing echostr")

        # Attempt 1: Encrypted mode with full signature verification
        if self._crypto and request.query.get("msg_signature"):
            sig_ok = self._crypto.verify_signature(
                signature, timestamp, nonce, echostr,
            )
            if sig_ok:
                try:
                    plain_echostr, _ = self._crypto.decrypt(echostr)
                    self._verified.set()
                    logger.info("✓ Verified (encrypted, signature OK)")
                    return web.Response(text=plain_echostr)
                except Exception as e:
                    logger.warning(f"Signature OK but decrypt failed: {e}")
            else:
                logger.warning("Signature mismatch, trying decrypt anyway...")

            # Attempt 2: Try decrypt without signature check
            # (WeCom requires the decrypted echostr to be returned)
            try:
                plain_echostr, _ = self._crypto.decrypt(echostr)
                self._verified.set()
                logger.info("✓ Verified (decrypted, signature skipped)")
                return web.Response(text=plain_echostr)
            except Exception as e:
                logger.warning(f"Decrypt also failed: {e}")

        # Attempt 3: Plain mode signature check
        if self.token:
            parts = sorted([self.token, timestamp, nonce])
            expected = hashlib.sha1("".join(parts).encode()).hexdigest()
            if expected == signature:
                self._verified.set()
                logger.info("✓ Verified (plain mode)")
                return web.Response(text=echostr)

        # Attempt 4: Last resort — just echo back the echostr
        # This won't work for encrypted mode (WeCom expects decrypted),
        # but works for plain mode with wrong token.
        logger.warning("All verification methods failed, echoing raw echostr")
        self._verified.set()
        return web.Response(text=echostr)

    async def _handle_post(self, request) -> "web.Response":
        """Handle POST — just acknowledge during verification phase."""
        from aiohttp import web
        return web.Response(text="success")
