"""Tests for defer-on-LLM-failure: a call whose LLM step fails must be recorded
'failed' (so --new-only retries it next pass), not 'completed' with degraded
rule-based output baked in. A provider:none run, by contrast, is a legitimate
rule-based result and must still record 'completed'."""

import json

from deepscript.analyzers import build_analyzer_registry
from deepscript.cli.commands.analyze import _analyze_single, _record_result
from deepscript.config.settings import get_settings
from deepscript.llm.provider import LLMProvider
from deepscript.utils.manifest import ProcessingManifest


class _FailingProvider(LLMProvider):
    """Provider whose every LLM call fails (simulates a NIM 401 storm)."""

    def complete(self, *a, **k):
        self._note_failure()
        return None

    def complete_json(self, *a, **k):
        self._note_failure()
        return None


def _write_transcript(path):
    path.write_text(json.dumps({
        "text": "We discussed the Q3 sales pipeline and next steps with the customer.",
        "segments": [{"text": "We discussed the Q3 sales pipeline.", "speaker": "A"}],
    }))


def test_llm_failure_is_deferred_not_completed(tmp_path):
    cfg = get_settings()
    prov = _FailingProvider(cfg.llm)
    analyzers = build_analyzer_registry(llm=prov, settings=cfg)

    fp = tmp_path / "call.json"
    _write_transcript(fp)
    out = tmp_path / "out"

    prov.reset_failures()
    ac = _analyze_single(fp, cfg, prov, analyzers, None, False, False, False)
    assert ac.llm_failed is True

    man = ProcessingManifest()
    ok = _record_result(ac, fp, man, str(out))

    assert ok is False
    assert man.entries[str(fp)].status == "failed"
    # Output not written, and the file is NOT considered processed -> retried.
    assert not (out / "call.analysis.json").exists()
    assert man.is_processed(fp) is False


def test_rule_based_only_run_still_completes(tmp_path):
    """provider:none is an intentional rule-based run, not a failure."""
    cfg = get_settings()
    analyzers = build_analyzer_registry(llm=None, settings=cfg)

    fp = tmp_path / "call.json"
    _write_transcript(fp)
    out = tmp_path / "out"

    ac = _analyze_single(fp, cfg, None, analyzers, None, False, False, False)
    assert ac.llm_failed is False

    man = ProcessingManifest()
    ok = _record_result(ac, fp, man, str(out))

    assert ok is True
    assert man.entries[str(fp)].status == "completed"
    assert (out / "call.analysis.json").exists()
    assert man.is_processed(fp) is True
