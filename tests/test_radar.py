"""Tests for EvoScientist.radar — pure logic that doesn't need LLM or server."""

from EvoScientist.radar import (
    IdeaSketch,
    PaperMatch,
    RadarUpdate,
    _extract_radar_update,
    _parse_arxiv_entries,
    _write_radar_update,
    list_radar_history,
    load_radar_update,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2406.12345v1</id>
    <title>Multi-line
Title  Test</title>
    <summary>This is a test abstract about LLMs and retrieval.</summary>
    <published>2026-06-10T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
    <link title="pdf" href="http://arxiv.org/pdf/2406.12345v1"/>
    <arxiv:comment>12 pages, 3 figures, NeurIPS 2026</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2406.99999v1</id>
    <title>No Comment Paper</title>
    <summary>Short.</summary>
    <published>2026-06-11T00:00:00Z</published>
    <author><name>Carol</name></author>
    <category term="cs.LG"/>
    <link href="http://arxiv.org/abs/2406.99999v1" rel="alternate"/>
  </entry>
</feed>
"""


def _sample_update(**overrides) -> RadarUpdate:
    defaults = {
        "papers": [
            PaperMatch(
                title="Test Paper",
                url="https://arxiv.org/abs/2406.00001",
                relevance="matches interest in LLMs",
                connection="relates to RAG pipeline work",
            )
        ],
        "trends": ["growing interest in agents"],
        "ideas": [
            IdeaSketch(
                title="Agent benchmarks",
                motivation="no standard benchmark exists",
                suggested_approach="collect existing evals",
            )
        ],
        "suggested_actions": ["survey agent papers"],
        "summary_text": "# [Research Radar]\n\nFound 1 paper.",
    }
    defaults.update(overrides)
    return RadarUpdate(**defaults)


# ---------------------------------------------------------------------------
# _parse_arxiv_entries
# ---------------------------------------------------------------------------


class TestParseArxivEntries:
    def test_parses_two_entries(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert len(papers) == 2

    def test_extracts_arxiv_id(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[0]["arxiv_id"] == "2406.12345v1"

    def test_cleans_multiline_title(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert "\n" not in papers[0]["title"]
        assert papers[0]["title"] == "Multi-line Title  Test"

    def test_extracts_authors(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[0]["authors"] == ["Alice Smith", "Bob Jones"]

    def test_extracts_categories(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[0]["categories"] == ["cs.AI", "cs.CL"]

    def test_extracts_pdf_url(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[0]["pdf_url"] == "http://arxiv.org/pdf/2406.12345v1"

    def test_missing_pdf_url(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[1]["pdf_url"] == ""

    def test_extracts_comment(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert "NeurIPS" in papers[0]["comment"]

    def test_missing_comment(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[1]["comment"] == ""

    def test_builds_url(self):
        papers = _parse_arxiv_entries(SAMPLE_ARXIV_XML)
        assert papers[0]["url"] == "https://arxiv.org/abs/2406.12345v1"

    def test_empty_feed(self):
        xml = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        assert _parse_arxiv_entries(xml) == []


# ---------------------------------------------------------------------------
# _extract_radar_update
# ---------------------------------------------------------------------------


class TestExtractRadarUpdate:
    def test_none_input(self):
        assert _extract_radar_update(None) is None

    def test_empty_dict(self):
        assert _extract_radar_update({}) is None

    def test_from_structured_response_object(self):
        update = _sample_update()
        result = {"structured_response": update}
        assert _extract_radar_update(result) is update

    def test_from_structured_response_dict(self):
        update = _sample_update()
        result = {"structured_response": update.model_dump()}
        extracted = _extract_radar_update(result)
        assert extracted is not None
        assert extracted.papers[0].title == "Test Paper"

    def test_from_message_content_json(self):
        update = _sample_update()
        result = {
            "messages": [{"role": "assistant", "content": update.model_dump_json()}]
        }
        extracted = _extract_radar_update(result)
        assert extracted is not None
        assert len(extracted.papers) == 1

    def test_skips_non_string_content(self):
        result = {"messages": [{"role": "assistant", "content": 12345}]}
        assert _extract_radar_update(result) is None

    def test_skips_invalid_json(self):
        result = {"messages": [{"role": "assistant", "content": "not json {"}]}
        assert _extract_radar_update(result) is None

    def test_prefers_structured_response_over_messages(self):
        update_sr = _sample_update(summary_text="from SR")
        update_msg = _sample_update(summary_text="from msg")
        result = {
            "structured_response": update_sr,
            "messages": [
                {"role": "assistant", "content": update_msg.model_dump_json()}
            ],
        }
        assert _extract_radar_update(result).summary_text == "from SR"


# ---------------------------------------------------------------------------
# File I/O round-trip
# ---------------------------------------------------------------------------


class TestFileRoundTrip:
    def test_write_list_load(self, tmp_path):
        update = _sample_update()
        path, ts = _write_radar_update(update, memory_dir=tmp_path)

        assert path.exists()
        assert (tmp_path / "radar" / f"{ts}.json").exists()

        history = list_radar_history(memory_dir=tmp_path)
        assert len(history) == 1
        assert history[0]["date"] == ts
        assert history[0]["paper_count"] == 1

        loaded = load_radar_update(ts, memory_dir=tmp_path)
        assert loaded is not None
        assert loaded.papers[0].title == "Test Paper"
        assert loaded.summary_text == update.summary_text

    def test_list_empty_dir(self, tmp_path):
        assert list_radar_history(memory_dir=tmp_path) == []

    def test_load_missing_date(self, tmp_path):
        assert load_radar_update("1999-01-01_000000", memory_dir=tmp_path) is None

    def test_multiple_reports_sorted(self, tmp_path):
        import time

        u1 = _sample_update(summary_text="first")
        _write_radar_update(u1, memory_dir=tmp_path)
        time.sleep(1.1)
        u2 = _sample_update(summary_text="second")
        _write_radar_update(u2, memory_dir=tmp_path)

        history = list_radar_history(memory_dir=tmp_path)
        assert len(history) == 2
        assert history[0]["date"] > history[1]["date"]  # newest first

    def test_non_matching_filename_skipped(self, tmp_path):
        radar_dir = tmp_path / "radar"
        radar_dir.mkdir()
        (radar_dir / "bad.json").write_text("not json", encoding="utf-8")
        assert list_radar_history(memory_dir=tmp_path) == []

    def test_malformed_json_skipped(self, tmp_path):
        radar_dir = tmp_path / "radar"
        radar_dir.mkdir()
        (radar_dir / "2026-06-10_120000.json").write_text("{invalid", encoding="utf-8")
        assert list_radar_history(memory_dir=tmp_path) == []


# ---------------------------------------------------------------------------
# _is_graph_not_registered
# ---------------------------------------------------------------------------
