"""X-MDD: LLM-based mispronunciation detection and diagnosis with canonical references."""

from xmdd.data import load_manifest, select_demos
from xmdd.model import XMDD

__all__ = ["XMDD", "load_manifest", "select_demos"]
