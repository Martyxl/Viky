"""Importing this module registers every tool and re-exports the registry API."""

from __future__ import annotations

# Importing the modules runs their register(...) calls.
from tools import clock, n8n, n8n_deploy, stats  # noqa: F401
from tools.base import all_tools, dispatch, get_tool, is_dry_run

__all__ = ["all_tools", "dispatch", "get_tool", "is_dry_run"]
