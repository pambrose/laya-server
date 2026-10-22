"""Turn a request body's bytes into JSON, undoing any Content-Encoding first.

typesafe-sdk 0.7.5 and later zstd-compresses request bodies of 1024 bytes or
more by default and says so with `Content-Encoding: zstd`. Starlette does not
decode request bodies, so without this the compressed bytes reached the JSON
parser and the request failed with a 500.

Every failure here is a `ValidationFailed`, a 422: Jev documents no other code
for a request it cannot accept, and none of them may surface as a 500.
"""

import json
import sys
from typing import Any

from .errors import ValidationFailed

if sys.version_info >= (3, 14):
    from compression import zstd
else:  # the same module, backported for 3.12 and 3.13
    from backports import zstd

SUPPORTED_ENCODINGS = ("zstd",)


def _codings(content_encoding: str | None) -> list[str]:
    """The codings applied to the body, in order, without `identity`.

    Content-Encoding is a comma-separated list (RFC 9110 section 8.4), and
    `identity` means no coding at all.
    """
    if not content_encoding:
        return []
    codings = (c.strip().lower() for c in content_encoding.split(","))
    return [c for c in codings if c and c != "identity"]


def _unzstd(body: bytes, limit: int) -> bytes:
    """Decompress every frame in `body`, refusing to produce more than `limit` bytes.

    The cap is applied while decompressing, not after: a few kilobytes of zstd
    can expand to gigabytes, so the output is never allowed past `limit + 1`
    bytes. One decompressor reads one frame, and RFC 8878 allows several frames
    back to back, so each frame's leftover input starts the next.
    """
    out = bytearray()
    remaining = body
    while True:
        decompressor = zstd.ZstdDecompressor()
        try:
            out += decompressor.decompress(remaining, max_length=limit - len(out) + 1)
        except zstd.ZstdError as exc:
            raise ValidationFailed(f"request body is not valid zstd: {exc}") from exc
        if len(out) > limit:
            # Decompression stopped at the cap, so the full size is never known.
            raise ValidationFailed(
                f"request body too large once decompressed: exceeds the limit of "
                f"{limit} bytes"
            )
        # Below the cap, a frame that has not ended has run out of input.
        if not decompressor.eof:
            raise ValidationFailed(
                "request body is not valid zstd: the frame is truncated"
            )
        remaining = decompressor.unused_data
        if not remaining:
            return bytes(out)


def decode_json_body(body: bytes, content_encoding: str | None, limit: int) -> Any:
    """Parse `body` as JSON after undoing `content_encoding`.

    `limit` bounds the decoded size; the caller has already held the encoded
    size to it. Raises `ValidationFailed` for an unsupported coding, a body that
    does not decode, one that decodes past `limit`, or one that is not JSON.
    """
    codings = _codings(content_encoding)
    if codings == ["zstd"]:
        body = _unzstd(body, limit)
    elif codings:
        raise ValidationFailed(
            f"unsupported Content-Encoding {content_encoding!r}; "
            f"send the body uncompressed or as one of: {', '.join(SUPPORTED_ENCODINGS)}"
        )

    try:
        return json.loads(body)
    except (ValueError, RecursionError) as exc:
        # ValueError covers JSONDecodeError and UnicodeDecodeError; RecursionError
        # is what the parser raises on nesting deeper than the interpreter allows.
        raise ValidationFailed(f"request body is not valid JSON: {exc}") from exc
