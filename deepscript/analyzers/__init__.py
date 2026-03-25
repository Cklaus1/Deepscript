"""Analyzer auto-discovery registry.

Discovers all BaseAnalyzer subclasses and builds a call_type → class mapping.
Adding a new analyzer = create the file with a class that has `supported_types` — done.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Optional, TYPE_CHECKING

from deepscript.analyzers.base import BaseAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# Cache discovered classes
_discovered: dict[str, type[BaseAnalyzer]] | None = None


def discover_analyzer_classes() -> dict[str, type[BaseAnalyzer]]:
    """Import all analyzer modules and collect class → supported_types mapping.

    Returns: {call_type: AnalyzerClass}
    """
    global _discovered
    if _discovered is not None:
        return _discovered

    registry: dict[str, type[BaseAnalyzer]] = {}

    # Import all modules in this package
    package_path = __path__  # type: ignore[name-defined]
    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        if modname == "base":
            continue
        try:
            module = importlib.import_module(f"deepscript.analyzers.{modname}")
            # Find all BaseAnalyzer subclasses in this module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseAnalyzer)
                    and attr is not BaseAnalyzer
                    and hasattr(attr, "supported_types")
                ):
                    # Instantiate temporarily to get supported_types
                    try:
                        instance = attr.__new__(attr)
                        types = instance.supported_types
                        for ct in types:
                            if ct not in registry:
                                registry[ct] = attr
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Skipped analyzer module %s: %s", modname, e)

    _discovered = registry
    return registry


def collect_keywords() -> dict[str, list[str]]:
    """Collect CLASSIFICATION_KEYWORDS from all analyzer classes.

    Analyzers can define a class attribute `classification_keywords: dict[str, list[str]]`
    mapping call types to keyword lists for the classifier.
    """
    keywords: dict[str, list[str]] = {}
    classes = discover_analyzer_classes()

    seen_classes: set[type] = set()
    for ct, cls in classes.items():
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        cls_keywords = getattr(cls, "classification_keywords", None)
        if cls_keywords and isinstance(cls_keywords, dict):
            for ktype, kwords in cls_keywords.items():
                if ktype not in keywords:
                    keywords[ktype] = kwords

    return keywords


def build_analyzer_registry(
    llm: Optional["LLMProvider"] = None,
    settings: Any = None,
) -> dict[str, BaseAnalyzer]:
    """Build instantiated analyzer registry from auto-discovered classes.

    Falls back to SimpleAnalyzer for any init failures.
    """
    from deepscript.analyzers.specialized import SimpleAnalyzer

    fallback = SimpleAnalyzer(llm=llm)
    classes = discover_analyzer_classes()
    registry: dict[str, BaseAnalyzer] = {}
    instantiated: dict[type, BaseAnalyzer] = {}

    # Config-driven kwargs per class
    init_kwargs: dict[str, dict[str, Any]] = {}
    if settings:
        init_kwargs = {
            "SalesAnalyzer": {"methodology": getattr(settings.sales, "methodology", "meddic"),
                              "competitors": getattr(settings.sales, "competitors", [])},
            "DiscoveryAnalyzer": {"framework": getattr(settings.discovery, "framework", "mom_test")},
            "InterviewAnalyzer": {},  # interview_type varies per call_type
            "ManagementAnalyzer": {},
            "OperationsAnalyzer": {},
            "CustomerAnalyzer": {},
            "EducationAnalyzer": {},
        }

    for ct, cls in classes.items():
        try:
            if cls not in instantiated:
                kwargs: dict[str, Any] = {"llm": llm}
                # Apply config-driven kwargs
                cls_name = cls.__name__
                if cls_name in init_kwargs:
                    kwargs.update(init_kwargs[cls_name])

                # Some analyzers need type-specific params
                if cls_name == "InterviewAnalyzer":
                    kwargs["interview_type"] = "technical" if "technical" in ct else "behavioral"
                elif cls_name == "ManagementAnalyzer":
                    kwargs["meeting_type"] = ct
                elif cls_name == "OperationsAnalyzer":
                    kwargs["ops_type"] = ct
                elif cls_name == "CustomerAnalyzer":
                    kwargs["cs_type"] = ct
                elif cls_name == "EducationAnalyzer":
                    kwargs["edu_type"] = ct

                # For parameterized analyzers, don't cache across different call_types
                if cls_name in ("InterviewAnalyzer", "ManagementAnalyzer", "OperationsAnalyzer",
                                "CustomerAnalyzer", "EducationAnalyzer"):
                    registry[ct] = cls(**kwargs)
                    continue

                instantiated[cls] = cls(**kwargs)

            registry[ct] = instantiated[cls]
        except Exception as e:
            logger.warning("Failed to init %s for %s: %s", cls.__name__, ct, e)
            registry[ct] = fallback

    # Ensure "unknown" always exists
    if "unknown" not in registry:
        registry["unknown"] = fallback

    return registry
