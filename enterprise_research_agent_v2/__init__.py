"""
NVIDIA AIQ Enterprise Research Agent v2

Dataloop implementation of the NVIDIA AI-Q Blueprint v2.0.0.
Uses deepagents for deep research, intent classification for routing,
and citation verification for trusted outputs.
"""

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, Model

# ---------------------------------------------------------------------------
# SDK patches — applied once when the package is first imported.
# ---------------------------------------------------------------------------

# 1. Register models that aren't in langchain's static table so that
#    determine_model() resolves them immediately and supports_tools is set.
for _model_id in (
    "nvidia/nemotron-3-nano-30b-a3b",
    "openai/gpt-oss-120b",
):
    if _model_id not in MODEL_TABLE:
        MODEL_TABLE[_model_id] = Model(
            id=_model_id,
            model_type="chat",
            client="ChatNVIDIA",
            supports_tools=True,
            supports_structured_output=True,
        )

# 2. The NVIDIA API can return duplicate model entries, which triggers an
#    assertion in the SDK. Override the property to auto-deduplicate.
_orig_available_models = _NVIDIAClient.available_models.fget


def _deduped_available_models(self):
    models = _orig_available_models(self)
    seen, unique = set(), []
    for m in models:
        if m.id not in seen:
            seen.add(m.id)
            unique.append(m)
    if len(unique) != len(models):
        self._available_models = unique
    return unique


_NVIDIAClient.available_models = property(_deduped_available_models)
