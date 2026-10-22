"""Request bodies: zstd decompression, the size limit after it, and malformed JSON.

typesafe-sdk 0.7.5 zstd-compresses request bodies of 1024 bytes or more by
default (on Python 3.14+, or with its `zstd` extra) and sends
`Content-Encoding: zstd`. The server read those bytes as JSON text and
answered 500, as it did for any body that was not valid JSON. Every failure
here must be a Jev-shaped 422, the code Jev documents for a bad request.
"""

import asyncio
import json
import sys

import httpx2
import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import AsyncTypeSafeClient, SystemOneResponse

from laya_server.app import create_app
from laya_server.body import decode_json_body
from laya_server.config import Settings
from laya_server.engine import Engine
from laya_server.errors import ValidationFailed
from tests.conftest import FakeRouter

if sys.version_info >= (3, 14):
    from compression import zstd
else:
    from backports import zstd

# Long enough that the SDK compresses it (MIN_COMPRESSED_BODY_LENGTH = 1024).
STATE = {"body": "We were billed twice for March. Refund it or we cancel. " * 30}
QUESTIONS = {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}
REQUEST = {"model": "jev-latest", "state": STATE, "questions": QUESTIONS}
RAW = json.dumps(REQUEST).encode()
LIMIT = 1_000_000


def client(env=None):
    settings = Settings.from_env(env or {})
    engine = Engine(FakeRouter(), settings)
    return TestClient(
        create_app(settings, engine=engine), raise_server_exceptions=False
    )


def post(body, encoding=None, env=None, **headers):
    headers = {"Content-Type": "application/json", **headers}
    if encoding is not None:
        headers["Content-Encoding"] = encoding
    return client(env).post("/v1/systemone", content=body, headers=headers)


def assert_422(response, fragment):
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["type"] == "validation_error"
    assert fragment in error["message"]


class TestDecodeJsonBody:
    def test_no_encoding_is_parsed_as_json(self):
        assert decode_json_body(RAW, None, LIMIT) == REQUEST

    @pytest.mark.parametrize("encoding", ["identity", " IDENTITY ", ""])
    def test_identity_is_parsed_as_json(self, encoding):
        assert decode_json_body(RAW, encoding, LIMIT) == REQUEST

    @pytest.mark.parametrize("encoding", ["zstd", "ZSTD", " zstd ", "identity, zstd"])
    def test_zstd_is_decompressed(self, encoding):
        assert decode_json_body(zstd.compress(RAW), encoding, LIMIT) == REQUEST

    def test_concatenated_zstd_frames_are_all_decompressed(self):
        # RFC 8878 allows a stream of several frames; each needs its own decompressor.
        half = len(RAW) // 2
        body = zstd.compress(RAW[:half]) + zstd.compress(RAW[half:])
        assert decode_json_body(body, "zstd", LIMIT) == REQUEST

    def test_decompressed_size_is_held_to_the_limit(self):
        # 2000 bytes of JSON compress far below a 1000-byte limit; the limit
        # must apply to what the body expands to, not only to the wire size.
        body = zstd.compress(json.dumps({"pad": " " * 2000}).encode())
        assert len(body) < 1000
        with pytest.raises(ValidationFailed, match="too large once decompressed"):
            decode_json_body(body, "zstd", 1000)

    def test_decompressed_size_at_the_limit_is_accepted(self):
        assert decode_json_body(zstd.compress(RAW), "zstd", len(RAW)) == REQUEST

    def test_limit_counts_every_frame(self):
        frame = zstd.compress(json.dumps({"pad": " " * 600}).encode())
        with pytest.raises(ValidationFailed, match="too large once decompressed"):
            decode_json_body(frame + frame, "zstd", 1000)

    @pytest.mark.parametrize(
        "body",
        [
            b"definitely not zstd",
            zstd.compress(RAW)[:-8],  # truncated frame
            zstd.compress(RAW) + b"trailing garbage",
            b"",
        ],
        ids=["garbage", "truncated", "trailing-garbage", "empty"],
    )
    def test_invalid_zstd_is_rejected(self, body):
        with pytest.raises(ValidationFailed, match="not valid zstd"):
            decode_json_body(body, "zstd", LIMIT)

    @pytest.mark.parametrize(
        "encoding", ["gzip", "br", "deflate", "zstd, gzip", "zstd, zstd"]
    )
    def test_unsupported_encoding_is_rejected(self, encoding):
        with pytest.raises(ValidationFailed, match="unsupported Content-Encoding"):
            decode_json_body(RAW, encoding, LIMIT)

    @pytest.mark.parametrize(
        "body",
        [b'{"model": "jev-latest", ', b"\xff\xfe\xfd", b"", b"[" * 100_000],
        ids=["truncated", "not-utf8", "empty", "too-deep"],
    )
    def test_invalid_json_is_rejected(self, body):
        with pytest.raises(ValidationFailed, match="not valid JSON"):
            decode_json_body(body, None, LIMIT)


class TestEndpoint:
    def test_zstd_body_answers_as_the_uncompressed_one_does(self):
        plain = post(RAW)
        compressed = post(zstd.compress(RAW), encoding="zstd")
        assert plain.status_code == 200
        assert compressed.status_code == 200
        assert compressed.json() == plain.json()

    def test_malformed_json_is_422(self):
        assert_422(post(b'{"model": "jev-latest", '), "not valid JSON")

    def test_non_utf8_body_is_422(self):
        assert_422(post(b"\xff\xfe\xfd"), "not valid JSON")

    def test_corrupt_zstd_is_422(self):
        assert_422(post(b"definitely not zstd", encoding="zstd"), "not valid zstd")

    def test_unsupported_encoding_is_422(self):
        assert_422(post(RAW, encoding="gzip"), "unsupported Content-Encoding")

    def test_body_that_expands_past_the_limit_is_422(self):
        body = zstd.compress(json.dumps({"pad": " " * 5000}).encode())
        response = post(
            body, encoding="zstd", env={"LAYA_SERVER_MAX_BODY_BYTES": "1000"}
        )
        assert_422(response, "too large once decompressed")

    def test_auth_is_checked_before_the_body_is_decoded(self):
        # An unauthenticated caller learns nothing about whether its body decodes.
        env = {"LAYA_SERVER_API_KEY": "secret"}
        assert post(b"not zstd", encoding="zstd", env=env).status_code == 401
        assert post(b"{not json", env=env).status_code == 401


class TestRealSdk:
    """Drive TypeSafe's own client into the app, compressing as it does by default.

    The async client over an in-process ASGI transport, not the sync client over
    TestClient: the SDK passes a per-request timeout, which TestClient rejects
    with a deprecation warning.
    """

    def answer(self, compression):
        seen = []
        settings = Settings.from_env({})
        app = create_app(settings, engine=Engine(FakeRouter(), settings))

        async def recording(scope, receive, send):
            if scope["type"] == "http":
                seen.append(dict(scope["headers"]).get(b"content-encoding"))
            await app(scope, receive, send)

        async def call():
            async with AsyncTypeSafeClient(
                api_key="unused",
                base_url="http://testserver",
                transport=httpx2.ASGITransport(app=recording),
                compression=compression,
            ) as sdk:
                return await sdk.system_one(STATE, QUESTIONS, model="jev-latest")

        return asyncio.run(call()), seen

    def test_sdk_zstd_request_is_answered(self):
        result, seen = self.answer("zstd")
        assert seen == [b"zstd"]
        assert isinstance(result, SystemOneResponse)
        assert result.answers["refund"].noul == pytest.approx(0.91)

    def test_sdk_uncompressed_request_is_answered(self):
        result, seen = self.answer(False)
        assert seen == [None]
        assert result.answers["refund"].noul == pytest.approx(0.91)
