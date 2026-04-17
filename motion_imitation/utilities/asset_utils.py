"""Helpers for resolving bundled asset paths."""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_asset_path(relative_path):
  """Return a repo-local absolute asset path when bundled, else keep fallback."""
  candidate = _REPO_ROOT / relative_path
  if candidate.is_file():
    return str(candidate)
  return relative_path
