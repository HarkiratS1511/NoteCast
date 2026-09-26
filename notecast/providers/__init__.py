"""Alternative LLM provider adapters for NoteCast.

Each adapter presents an `anthropic.Anthropic`-shaped (duck-typed) surface so
the rest of the codebase can call it exactly like the real Anthropic client.
"""

from __future__ import annotations
