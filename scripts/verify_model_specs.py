"""Verify every motion_mirror model download spec against the live Hugging Face API.

This is the credit-protection layer: before a GPU pod ever spends money pulling
tens of gigabytes, this script asserts that every repo in ``_MODEL_SPECS`` exists,
that pinned single-file weights resolve to the expected size, and that snapshot
specs (including ``allow_patterns`` filtered ones) sum to roughly the advertised
byte count. For ``allow_patterns`` specs it additionally proves the allowlist plus
known-excludable files covers *every* file in the repo tree - an uncovered file is
almost always a config the base cache silently needs, so we fail loudly on it.

Uses only the standard library (urllib) beyond ``motion_mirror`` itself. Run it
directly::

    python scripts/verify_model_specs.py            # all specs, live HF API
    python scripts/verify_model_specs.py --timeout 60

Exit code is 0 only if every spec passes.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# --- src/ layout bootstrap -------------------------------------------------
# Insert the repo's src/ at the FRONT of sys.path so we import the in-tree
# motion_mirror.cli (with the current specs) rather than any stale copy that
# may be pip-installed into site-packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from motion_mirror.cli import _MODEL_GROUPS, _MODEL_SPECS  # noqa: E402

_HF = "https://huggingface.co"
_USER_AGENT = "motion-mirror-verify-model-specs/1.0"

# Tolerance on advertised vs. actual byte counts.
_SIZE_TOLERANCE = 0.15

# Files that legitimately live in a repo tree but are excluded from an
# allow_patterns snapshot. The transformer/* weights are pulled from the GGUF
# repo instead; the rest are repo metadata / example assets that the base cache
# does not need. Any file NOT covered by allow_patterns + these is suspect.
_KNOWN_EXCLUDABLE = ["transformer/*", ".gitattributes", "README.md", "assets/*"]


# --- HTTP helpers ----------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that does not follow - lets us inspect the 3xx itself."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


def _request(url: str, *, method: str, timeout: float, follow: bool = True) -> Any:
    """Perform an HTTP request, returning the response object.

    Raises urllib.error.HTTPError on non-2xx when following redirects; when
    ``follow`` is False a 3xx is returned as an HTTPError-like response so the
    caller can read redirect headers (e.g. HF's ``x-linked-size``).
    """
    req = urllib.request.Request(url, method=method, headers={"User-Agent": _USER_AGENT})
    if follow:
        opener = urllib.request.build_opener()
    else:
        opener = urllib.request.build_opener(_NoRedirect)
    return opener.open(req, timeout=timeout)


def _get_json(url: str, timeout: float) -> tuple[Any, dict[str, str]]:
    """GET a URL and parse JSON, returning (payload, response_headers)."""
    resp = _request(url, method="GET", timeout=timeout)
    with resp:
        payload = json.loads(resp.read().decode("utf-8"))
        headers = {k.lower(): v for k, v in resp.headers.items()}
    return payload, headers


# --- individual checks -----------------------------------------------------
def check_repo_exists(repo_id: str, timeout: float) -> tuple[bool, str]:
    """GET /api/models/<repo_id> and expect HTTP 200."""
    url = f"{_HF}/api/models/{repo_id}"
    try:
        resp = _request(url, method="GET", timeout=timeout)
        with resp:
            code = resp.getcode()
        if code == 200:
            return True, "repo exists (200)"
        return False, f"unexpected status {code}"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} for {url}"
    except Exception as exc:  # pragma: no cover - network failure surface
        return False, f"error: {exc}"


def _resolve_size(repo_id: str, filename: str, timeout: float) -> tuple[bool, int | None, str]:
    """HEAD the resolve URL for a single file, returning (reachable, size, detail).

    HF's resolve endpoint 302-redirects LFS files to a CDN host but exposes the
    real size via the ``x-linked-size`` header on that FIRST response. We read
    that when present, otherwise follow the redirect and read Content-Length.
    """
    url = f"{_HF}/{repo_id}/resolve/main/{filename}"
    try:
        resp = _request(url, method="HEAD", timeout=timeout, follow=False)
    except urllib.error.HTTPError as exc:
        # A 3xx surfaces here (HTTPError) because we suppressed auto-follow.
        if 300 <= exc.code < 400:
            linked = exc.headers.get("x-linked-size")
            if linked is not None:
                return True, int(linked), f"resolve 302, x-linked-size={linked}"
            location = exc.headers.get("location")
            if location:
                return _follow_for_size(location, timeout)
            return False, None, f"redirect {exc.code} with no size header"
        return False, None, f"HTTP {exc.code} for {url}"
    except Exception as exc:  # pragma: no cover
        return False, None, f"error: {exc}"

    # Direct 200 (small, non-LFS files served inline).
    with resp:
        if resp.getcode() != 200:
            return False, None, f"unexpected status {resp.getcode()}"
        linked = resp.headers.get("x-linked-size")
        clen = resp.headers.get("content-length")
        if linked is not None:
            return True, int(linked), f"200, x-linked-size={linked}"
        if clen is not None:
            return True, int(clen), f"200, content-length={clen}"
    return False, None, "200 but no size header"


def _follow_for_size(url: str, timeout: float) -> tuple[bool, int | None, str]:
    try:
        resp = _request(url, method="HEAD", timeout=timeout, follow=True)
    except urllib.error.HTTPError as exc:
        return False, None, f"HTTP {exc.code} following redirect"
    except Exception as exc:  # pragma: no cover
        return False, None, f"error following redirect: {exc}"
    with resp:
        clen = resp.headers.get("content-length")
        if resp.getcode() == 200 and clen is not None:
            return True, int(clen), f"followed redirect, content-length={clen}"
    return False, None, "followed redirect but no content-length"


def _within_tolerance(actual: int, expected: int) -> bool:
    if expected <= 0:
        return False
    return abs(actual - expected) <= _SIZE_TOLERANCE * expected


def check_filename_spec(spec: dict, timeout: float) -> tuple[bool, str]:
    """Verify a single pinned-file spec: reachable + size within tolerance."""
    repo_id = spec["repo_id"]
    filename = spec["filename"]
    expected = spec["expected_bytes"]
    reachable, size, detail = _resolve_size(repo_id, filename, timeout)
    if not reachable or size is None:
        return False, f"{filename}: {detail}"
    ok = _within_tolerance(size, expected)
    pct = (size - expected) / expected * 100 if expected else 0.0
    status = "within" if ok else "OUTSIDE"
    return ok, (
        f"{filename}: {size:,}B vs expected {expected:,}B "
        f"({pct:+.1f}%, {status} +/-{int(_SIZE_TOLERANCE * 100)}%)"
    )


def _fetch_tree(repo_id: str, timeout: float) -> list[dict]:
    """Fetch the full recursive file tree, paginating via the Link header."""
    entries: list[dict] = []
    url: str | None = f"{_HF}/api/models/{repo_id}/tree/main?recursive=true"
    while url:
        resp = _request(url, method="GET", timeout=timeout)
        with resp:
            page = json.loads(resp.read().decode("utf-8"))
            link = resp.headers.get("Link", "")
        entries.extend(page)
        url = _parse_next_link(link)
    return entries


def _parse_next_link(link_header: str) -> str | None:
    """Extract the rel="next" URL from an RFC-5988 Link header, if any."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        target = segments[0].strip().lstrip("<").rstrip(">")
        for seg in segments[1:]:
            if seg.strip() == 'rel="next"':
                return target
    return None


def _entry_size(entry: dict) -> int:
    lfs = entry.get("lfs")
    if isinstance(lfs, dict) and lfs.get("size") is not None:
        return int(lfs["size"])
    return int(entry.get("size", 0) or 0)


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def check_snapshot_spec(spec: dict, timeout: float) -> tuple[bool, str]:
    """Verify a snapshot spec: summed size within tolerance, and (when
    allow_patterns is set) the allowlist + known-excludable covers every file."""
    repo_id = spec["repo_id"]
    expected = spec["expected_bytes"]
    allow_patterns = spec.get("allow_patterns")

    try:
        tree = _fetch_tree(repo_id, timeout)
    except urllib.error.HTTPError as exc:
        return False, f"tree fetch HTTP {exc.code}"
    except Exception as exc:  # pragma: no cover
        return False, f"tree fetch error: {exc}"

    files = [e for e in tree if e.get("type") == "file"]

    if allow_patterns:
        matched = [e for e in files if _matches_any(e["path"], allow_patterns)]
    else:
        matched = files
    total = sum(_entry_size(e) for e in matched)

    ok = _within_tolerance(total, expected)
    pct = (total - expected) / expected * 100 if expected else 0.0
    status = "within" if ok else "OUTSIDE"
    detail = (
        f"{len(matched)}/{len(files)} files, {total:,}B vs expected "
        f"{expected:,}B ({pct:+.1f}%, {status} +/-{int(_SIZE_TOLERANCE * 100)}%)"
    )

    if allow_patterns:
        covering = list(allow_patterns) + _KNOWN_EXCLUDABLE
        uncovered = [e["path"] for e in files if not _matches_any(e["path"], covering)]
        if uncovered:
            ok = False
            detail += f"; UNCOVERED files (base cache likely needs these): {uncovered}"

    return ok, detail


def verify_spec(key: str, spec: dict, timeout: float) -> tuple[bool, str]:
    """Verify one spec end to end: repo existence then size/coverage."""
    repo_ok, repo_detail = check_repo_exists(spec["repo_id"], timeout)
    if not repo_ok:
        return False, repo_detail
    if spec.get("filename") is not None:
        ok, detail = check_filename_spec(spec, timeout)
    else:
        ok, detail = check_snapshot_spec(spec, timeout)
    return ok, detail


# --- CLI -------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-request timeout in seconds (default 30)."
    )
    args = parser.parse_args(argv)

    print(f"Verifying {len(_MODEL_SPECS)} model spec(s) against {_HF} ...")
    print(f"(groups: {', '.join(_MODEL_GROUPS)})\n")

    results: list[tuple[str, bool, str]] = []
    for key, spec in _MODEL_SPECS.items():
        ok, detail = verify_spec(key, spec, args.timeout)
        results.append((key, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {key} ({spec['repo_id']})")
        print(f"         {detail}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{total} specs passed")
    for key, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL':4}  {key}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
