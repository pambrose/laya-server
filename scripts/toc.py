"""Regenerate the README's table of contents, or check that it is current.

    uv run python scripts/toc.py           # rewrite the TOC in place
    uv run python scripts/toc.py --check   # exit 1 if it is out of date

A hand-maintained TOC rots silently: a renamed heading leaves an anchor that
simply does not scroll, with nothing to catch it. `make toc-check` runs in CI
so a heading change fails the build instead.
"""

import argparse
import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

START = "<!-- toc -->"
END = "<!-- /toc -->"

# Headings shallower than this are the title; deeper ones are too granular.
MIN_LEVEL, MAX_LEVEL = 2, 3


def anchor(title: str) -> str:
    """GitHub's slug: lowercase, punctuation dropped, spaces to hyphens."""
    slug = title.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug)


def headings(markdown: str) -> list[tuple[int, str]]:
    body = markdown.split(END, 1)[-1] if END in markdown else markdown
    found = []
    for match in re.finditer(r"^(#{2,6})[ \t]+(.+?)[ \t]*$", body, re.M):
        level = len(match.group(1))
        if MIN_LEVEL <= level <= MAX_LEVEL:
            found.append((level, match.group(2)))
    return found


def render(markdown: str) -> str:
    lines = [
        f"{'  ' * (level - MIN_LEVEL)}- [{title}](#{anchor(title)})"
        for level, title in headings(markdown)
    ]
    return "\n".join([START, "", *lines, "", END])


def rewrite(markdown: str) -> str:
    toc = render(markdown)
    if START in markdown and END in markdown:
        return re.sub(
            re.escape(START) + r".*?" + re.escape(END), toc, markdown, flags=re.S
        )
    raise SystemExit(f"README.md has no {START} / {END} markers")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the TOC is out of date"
    )
    args = parser.parse_args()

    current = README.read_text()
    updated = rewrite(current)

    if args.check:
        if current != updated:
            print("README.md table of contents is out of date; run `make toc`")
            return 1
        print("README.md table of contents is current")
        return 0

    if current != updated:
        README.write_text(updated)
        print("README.md table of contents updated")
    else:
        print("README.md table of contents already current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
