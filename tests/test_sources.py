"""Tier 1 — watching the world (sources.py, relevance.py). Offline: a fixture feed
and fake judges, so nothing here touches the network or a model.
"""

import pytest

import memory
import relevance
import sources

pytestmark = pytest.mark.unit

FEED = """<?xml version="1.0"?><rss version="2.0"><channel>
  <item><title>Erin Milez&#8217;s Dense Paintings</title>
        <link>https://example.com/a</link>
        <description>A painter of crowded cities.</description>
        <pubDate>Mon, 01 Sep 2026 10:00:00 +0000</pubDate></item>
  <item><title>Audubon Photography Awards</title>
        <link>https://example.com/b</link><description>Birds.</description></item>
  <item><title>No link here</title><description>Body.</description></item>
</channel></rss>"""


def test_parses_titles_links_and_dates():
    items = sources.parse_feed(FEED, "test")
    assert [i.title for i in items][:2] == ["Erin Milez’s Dense Paintings",
                                            "Audubon Photography Awards"]
    assert items[0].url == "https://example.com/a"
    assert items[0].published is not None
    assert "painter" in items[0].body


def test_entities_are_decoded_not_left_raw():
    """Feeds escape punctuation; a Keeper reading "&#8217;" aloud is a tell."""
    assert "&#8217;" not in sources.parse_feed(FEED, "t")[0].title


def test_malformed_xml_yields_nothing_rather_than_raising():
    assert sources.parse_feed("<not xml", "t") == []
    assert sources.parse_feed("", "t") == []


def test_the_key_follows_the_url_not_the_title():
    """Publishers retitle stories after posting. Keyed on the title, the same item
    would arrive again as though it were new."""
    a = sources.SourceItem(source="s", title="One title", url="https://x/1")
    b = sources.SourceItem(source="s", title="Retitled later", url="https://x/1")
    assert a.key == b.key


def test_an_item_with_no_link_still_gets_a_stable_key():
    a = sources.SourceItem(source="s", title="No link here")
    b = sources.SourceItem(source="s", title="No link here")
    assert a.key == b.key and a.key


def test_seen_store_survives_a_restart(tmp_path):
    p = tmp_path / "seen.jsonl"
    item = sources.SourceItem(source="s", title="t", url="https://x/1")
    s1 = sources.SeenStore(p)
    assert not s1.seen(item)
    s1.mark(item)
    assert sources.SeenStore(p).seen(item), "delivery memory must outlive the process"


def test_unseen_filters_what_was_already_delivered(tmp_path):
    store = sources.SeenStore(tmp_path / "seen.jsonl")
    items = sources.parse_feed(FEED, "t")
    store.mark(items[0])
    assert [i.title for i in store.unseen(items)] == [i.title for i in items[1:]]


# --- the relevance verdict --- #

@pytest.mark.parametrize("raw,score,because", [
    ("8 - Is a painter.", 0.8, "Is a painter."),
    ("10 — Started painting again.", 1.0, "Started painting again."),
    ("0 - NONE", 0.0, ""),
    ("3", 0.3, ""),
    ("banana", 0.0, ""),
    ("", 0.0, ""),
])
def test_verdict_parsing_is_lenient_and_fails_to_zero(raw, score, because):
    assert relevance.parse_verdict(raw) == (score, because)


def test_two_thresholds_are_different_bars():
    """The design turns on this: "worth mentioning" changes what is said when the
    Keeper had already decided to speak; only "worth interrupting" may make it
    speak at all. One threshold would let mild relevance buy an interruption."""
    assert relevance.INTERRUPT > relevance.MENTION
    assert relevance.worth_mentioning(0.5) and not relevance.worth_interrupting(0.5)
    assert relevance.worth_interrupting(0.9)


def test_nothing_known_means_nothing_can_be_relevant(tmp_path):
    empty = memory.MemoryStore(tmp_path / "f.jsonl")
    item = sources.SourceItem(source="s", title="Paintings", url="https://x/1")
    assert relevance.score_item(item, empty, generate=lambda s, u: "9 - x") == (0.0, "")


def test_without_a_judge_it_never_claims_relevance(tmp_path):
    """No model available must mean silence, not a guess from cosine alone."""
    store = memory.MemoryStore(tmp_path / "f.jsonl")
    store.add("Is a painter.", "identity")
    item = sources.SourceItem(source="s", title="Paintings", url="https://x/1")
    assert relevance.score_item(item, store, generate=None) == (0.0, "")


def test_a_failing_judge_is_treated_as_irrelevant(tmp_path):
    store = memory.MemoryStore(tmp_path / "f.jsonl")
    store.add("Is a painter.", "identity")
    def boom(s, u): raise RuntimeError("429")
    item = sources.SourceItem(source="s", title="Paintings", url="https://x/1")
    assert relevance.score_item(item, store, generate=boom) == (0.0, "")
