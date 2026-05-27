"""``ignite_tools.core`` - shared infrastructure for all ignite-tools.

Public surface intentionally narrow. Tools import from here, not from the
internal modules. The submodules (``config``, ``sources``, ``format``,
``sampling``, ``sniffer``, ``cli``) are the implementation.
"""

from ignite_tools.core.config import (
    FiltersBlock,
    FormatBlock,
    FormatConfig,
    IdBlock,
    LabelsBlock,
    MaskRule,
    NormalizeBlock,
    RouteBlock,
    SamplingBlock,
    StorageBlock,
    TextBlock,
    TrimRule,
)
from ignite_tools.core.embed import (
    BACKEND_ENV_VAR,
    DEFAULT_BACKEND,
    clear_model_cache,
    embed,
    resolve_backend,
)
from ignite_tools.core.format import (
    CorpusReadError,
    Item,
    ReadSummary,
    read_corpus,
)
from ignite_tools.core.sniffer import (
    SniffResult,
    build_proposal,
    format_human_summary,
    sniff_path,
)
from ignite_tools.core.sources import cache_dir_for

__all__ = [
    # config
    "FormatConfig",
    "StorageBlock",
    "FormatBlock",
    "TextBlock",
    "RouteBlock",
    "IdBlock",
    "LabelsBlock",
    "NormalizeBlock",
    "MaskRule",
    "TrimRule",
    "FiltersBlock",
    "SamplingBlock",
    # read layer
    "Item",
    "read_corpus",
    "CorpusReadError",
    "ReadSummary",
    # sniffer
    "sniff_path",
    "build_proposal",
    "format_human_summary",
    "SniffResult",
    "cache_dir_for",
    # embedding
    "embed",
    "clear_model_cache",
    "resolve_backend",
    "DEFAULT_BACKEND",
    "BACKEND_ENV_VAR",
]
