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


# --- MCP tool output as a source --- #

GIT_LOG = """Commit history for 'main':

commit 4f2a1b9c
Author: Jess <jess@example.com>
Date:   Thu Sep 4 19:02:11 2026

    bound the history deque so a long session stops growing

commit 8c1d0e77
Author: Jess <jess@example.com>
Date:   Thu Sep 4 11:40:03 2026

    close the unauthenticated A2A endpoint
"""


def test_a_commit_reads_as_its_message_not_its_hash():
    """The title is what the person would be told. "commit 4f2a1b9c" is not a
    thing anyone says, and it scores against memory as noise."""
    items = sources.parse_tool_output(GIT_LOG, source="repo")
    titles = [i.title for i in items]
    assert "bound the history deque so a long session stops growing" in titles
    assert "close the unauthenticated A2A endpoint" in titles
    assert not any(t.startswith("commit ") for t in titles)


def test_the_commit_message_is_not_split_from_its_commit():
    """git prints the message as its own indented paragraph. Read naively that is
    a separate record, and the log arrives as twice as many half-items."""
    items = sources.parse_tool_output(GIT_LOG, source="repo")
    assert len(items) == 3      # the header line, then one record per commit
    body = next(i.body for i in items if "deque" in i.title)
    assert "4f2a1b9c" in body   # the hash stayed with its message


def test_the_same_commit_polled_twice_keeps_one_key():
    """Dedup is the whole basis of "new since I last looked"."""
    first = sources.parse_tool_output(GIT_LOG, source="repo")
    again = sources.parse_tool_output(GIT_LOG, source="repo")
    assert [i.key for i in first] == [i.key for i in again]


def test_two_commits_do_not_share_a_key():
    items = sources.parse_tool_output(GIT_LOG, source="repo")
    assert len({i.key for i in items}) == len(items)


def test_a_plain_listing_becomes_one_item_per_line():
    out = sources.parse_tool_output("gesso.txt\nstretchers.txt\nsam-letter.md",
                                    source="files")
    assert [i.title for i in out] == ["gesso.txt", "stretchers.txt", "sam-letter.md"]


def test_a_failed_tool_call_yields_nothing_rather_than_an_item():
    """mcp.call answers with "(tool error: ...)" or "(tool timed out ...)" instead
    of raising. Those must never reach the person as something that happened."""
    for text in ("(tool error: boom)", "(no output)",
                 "(tool timed out after 20s: git__git_log)", "", "   "):
        assert sources.parse_tool_output(text, source="repo") == []


def test_the_item_count_is_capped():
    many = "\n".join(f"file{n}.txt" for n in range(50))
    assert len(sources.parse_tool_output(many, max_items=8)) == 8


def test_the_source_label_is_carried_through():
    items = sources.parse_tool_output(GIT_LOG, source="repo")
    assert all(i.source == "repo" for i in items)


def test_a_watched_tool_item_is_scored_by_the_same_gate(tmp_path):
    """The point of the shape: a commit goes through the identical relevance pass
    a feed item does, so nothing about deciding to speak had to be duplicated."""
    store = memory.MemoryStore(tmp_path / "f.jsonl")
    store.add("Is a painter who stopped in March.", kind="fact")
    item = sources.parse_tool_output(GIT_LOG, source="repo")[1]
    score, because = relevance.score_item(item, store, generate=None, embed=None)
    assert 0.0 <= score <= 1.0
    assert isinstance(because, str)


def test_a_single_record_is_not_shredded_into_lines():
    """Regression: the fallback to one-item-per-line keyed off the MERGED block
    count, so a log holding exactly ONE commit looked like a listing. Its hash,
    its author and its date each arrived as a separate thing that had happened."""
    one = ("commit 4f2a1b9c\n"
           "Author: Jess <jess@example.com>\n"
           "Date:   Thu Sep 4 19:02:11 2026\n\n"
           "    finished stretching the big canvas\n")
    items = sources.parse_tool_output(one, source="repo")
    assert len(items) == 1
    assert items[0].title == "finished stretching the big canvas"
    assert not any(i.title.startswith("Author:") for i in items)


# The shape mcp-server-git ACTUALLY emits, captured from a live poll. The earlier
# tests in this file were written against what git's CLI prints, which is not what
# the MCP server prints: it labels every field and packs the whole commit message
# into one quoted string with escaped newlines. Parsing was written against the
# guess and had to be corrected against this.
REAL_GIT_MCP = (
    "Commit history:\n"
    "Commit: '0a08dd00d17e5e57237bfa742912faf07bc397f6'\n"
    "Author: <git.Actor \"Jess <jess@example.com>\">\n"
    "Date: 2026-09-06 22:44:04-05:00\n"
    "Message: \"Measure the interruption policy, cap the day\\n\\nFour things, "
    "prompted by a comparison with the other agent.\"\n"
    "\n"
    "Commit: '2b1bf53b56113a30a99fa2fcf9babdaf8d307688'\n"
    "Author: <git.Actor \"Jess <jess@example.com>\">\n"
    "Date: 2026-09-06 22:36:34-05:00\n"
    "Message: 'Give the Keeper another agent to actually talk to\\n\\nA2A was "
    "implemented on both sides but never used.'\n")


def test_the_real_git_server_shape_reads_as_commit_subjects():
    items = sources.parse_tool_output(REAL_GIT_MCP, source="repo")
    assert [i.title for i in items] == [
        "Measure the interruption policy, cap the day",
        "Give the Keeper another agent to actually talk to",
    ]


def test_no_field_label_or_quote_survives_into_the_title():
    """These end up in the Keeper's mouth. 'Message: "did the thing\\n\\nbecause"'
    is not a sentence anybody says."""
    for item in sources.parse_tool_output(REAL_GIT_MCP, source="repo"):
        assert not item.title.startswith(("Message:", "Commit:", "Author:"))
        assert "\\n" not in item.title
        assert item.title[0] not in "\"'"


def test_the_commit_body_is_kept_for_scoring():
    """Only the TITLE is trimmed. Relevance still sees the hash, author and date,
    which is what lets a commit match something the person told the Keeper."""
    first = sources.parse_tool_output(REAL_GIT_MCP, source="repo")[0]
    assert "0a08dd00" in first.body
    assert "Four things" in first.body


def test_real_shape_commits_keep_distinct_stable_keys():
    a = sources.parse_tool_output(REAL_GIT_MCP, source="repo")
    b = sources.parse_tool_output(REAL_GIT_MCP, source="repo")
    assert [i.key for i in a] == [i.key for i in b]
    assert len({i.key for i in a}) == 2
