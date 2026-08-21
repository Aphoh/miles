import asyncio
import json

import pytest

from miles.rollout.session.sglang_stream import consume_sglang_sse


_MISSING = object()


async def _lines(*values: str):
    for value in values:
        yield value


def _consume_lines(*values: str, decode=lambda token_ids: f"decoded:{token_ids}"):
    return asyncio.run(consume_sglang_sse(_lines(*values), decode=decode))


def _consume_chunks(*chunks: dict, decode=lambda token_ids: f"decoded:{token_ids}"):
    lines = [line for chunk in chunks for line in (f"data: {json.dumps(chunk)}", "")]
    lines.extend(("data: [DONE]", ""))
    return _consume_lines(*lines, decode=decode)


def _chunk(pairs, length, *, text="", output_ids=_MISSING, top_rows=_MISSING, **meta):
    meta_info = {
        "output_token_logprobs": pairs,
        "output_token_logprobs_length": length,
        **meta,
    }
    if top_rows is not _MISSING:
        meta_info["output_top_logprobs"] = top_rows

    if output_ids is _MISSING:
        output_ids = [pair[1] for pair in pairs]
    chunk = {"text": text, "meta_info": meta_info}
    if output_ids is not None:
        chunk["output_ids"] = output_ids
    return chunk


def test_incremental_sse_materializes_complete_native_response():
    first_top_row = [[-0.1, 101, "a"], [-0.4, 201, "x"]]
    response = _consume_chunks(
        _chunk(
            [[-0.1, 101, "a"]],
            1,
            text="a",
            top_rows=[first_top_row],
            finish_reason=None,
            routed_experts="stale",
        ),
        _chunk(
            [[-0.2, 102, "b"]],
            2,
            text="b",
            top_rows=[None],
            finish_reason=None,
        ),
        _chunk(
            [],
            2,
            top_rows=[],
            completion_tokens=2,
            finish_reason={"type": "stop"},
            routed_experts="terminal-routes",
            indexer_topk="terminal-indexer",
            weight_version="v7",
        ),
        decode=lambda _token_ids: pytest.fail("fully supplied text must not be decoded"),
    )

    assert response["text"] == "ab"
    assert response["output_ids"] == [101, 102]
    assert response["meta_info"]["output_token_logprobs"] == [
        [-0.1, 101, "a"],
        [-0.2, 102, "b"],
    ]
    assert response["meta_info"]["output_top_logprobs"] == [first_top_row, None]
    assert response["meta_info"]["routed_experts"] == "terminal-routes"
    assert response["meta_info"]["indexer_topk"] == "terminal-indexer"
    assert response["meta_info"]["weight_version"] == "v7"


def test_missing_text_decodes_ids_derived_from_logprob_pairs():
    response = _consume_chunks(
        _chunk(
            [[-0.1, 101, "a"]],
            1,
            text=None,
            output_ids=None,
            finish_reason=None,
        ),
        _chunk(
            [],
            1,
            completion_tokens=1,
            finish_reason={"type": "stop"},
        ),
    )

    assert response["text"] == "decoded:[101]"
    assert response["output_ids"] == [101]


@pytest.mark.parametrize(
    ("chunk", "match"),
    [
        (
            _chunk([[-0.1, 101]], 2, finish_reason={"type": "stop"}),
            "inconsistent output_token_logprobs_length",
        ),
        (
            _chunk([[-0.1, 101]], 1, output_ids=[999], finish_reason={"type": "stop"}),
            "output_ids disagree",
        ),
        (
            _chunk([[-0.1, 101]], 1, top_rows=[], finish_reason={"type": "stop"}),
            "one row per output token",
        ),
        (
            _chunk([[-0.1, 101]], 1, completion_tokens=2, finish_reason={"type": "stop"}),
            "terminal token counts disagree",
        ),
        (
            _chunk([[-0.1, 101]], 1, completion_tokens=1, finish_reason=None),
            "terminal finish_reason",
        ),
    ],
)
def test_stream_rejects_inconsistent_token_metadata(chunk, match):
    with pytest.raises(ValueError, match=match):
        _consume_chunks(chunk)


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (("event: error", 'data: {"error":{"message":"boom"}}'), "error event"),
        (('data: {"error":{"message":"boom"}}',), "error event"),
        (("data: {not-json}",), "not valid JSON"),
        (("data: []",), "decode to an object"),
        (("data: [DONE]",), "without any data chunks"),
    ],
)
def test_sse_rejects_error_or_malformed_events(lines, match):
    with pytest.raises(ValueError, match=match):
        _consume_lines(*lines)
