import asyncio
import json

import pytest

from miles.rollout.session.sglang_stream import SGLangStreamAccumulator, consume_sglang_sse


async def _lines(*values: str):
    for value in values:
        yield value


def test_incremental_sse_accumulates_tokens_and_terminal_metadata():
    first = {
        "text": "a",
        "output_ids": [101],
        "meta_info": {
            "output_token_logprobs": [[-0.1, 101, "a"]],
            "output_token_logprobs_length": 1,
            "completion_tokens": 1,
            "finish_reason": None,
        },
    }
    final = {
        "text": "b",
        "output_ids": [102],
        "meta_info": {
            "output_token_logprobs": [[-0.2, 102, "b"]],
            "output_token_logprobs_length": 2,
            "completion_tokens": 2,
            "finish_reason": {"type": "stop"},
            "routed_experts": [[1, 2]],
            "indexer_topk": [[3, 4]],
        },
    }

    response = asyncio.run(
        consume_sglang_sse(
            _lines(
                f"data: {json.dumps(first)}",
                "",
                f"data: {json.dumps(final)}",
                "",
                "data: [DONE]",
                "",
            ),
            decode=lambda token_ids: f"decoded:{token_ids}",
        )
    )

    assert response["text"] == "ab"
    assert response["output_ids"] == [101, 102]
    assert response["meta_info"]["output_token_logprobs"] == [
        [-0.1, 101, "a"],
        [-0.2, 102, "b"],
    ]
    assert response["meta_info"]["routed_experts"] == [[1, 2]]
    assert response["meta_info"]["indexer_topk"] == [[3, 4]]


def test_cumulative_stream_accepts_repeated_prefix():
    accumulator = SGLangStreamAccumulator(output_mode="cumulative")
    accumulator.add(
        {
            "text": "a",
            "output_ids": [101],
            "meta_info": {
                "output_token_logprobs": [[-0.1, 101]],
                "output_token_logprobs_length": 1,
                "finish_reason": None,
            },
        }
    )
    accumulator.add(
        {
            "text": "ab",
            "output_ids": [101, 102],
            "meta_info": {
                "output_token_logprobs": [[-0.1, 101], [-0.2, 102]],
                "output_token_logprobs_length": 2,
                "completion_tokens": 2,
                "finish_reason": {"type": "length"},
            },
        }
    )

    response = accumulator.finish(lambda token_ids: f"decoded:{token_ids}")

    assert response["text"] == "ab"
    assert response["output_ids"] == [101, 102]


def test_stream_rejects_output_id_disagreement():
    accumulator = SGLangStreamAccumulator()

    with pytest.raises(ValueError, match="output_ids disagree"):
        accumulator.add(
            {
                "text": "a",
                "output_ids": [999],
                "meta_info": {
                    "output_token_logprobs": [[-0.1, 101]],
                    "output_token_logprobs_length": 1,
                },
            }
        )


def test_stream_rejects_missing_terminal_finish_reason():
    accumulator = SGLangStreamAccumulator()
    accumulator.add(
        {
            "text": "a",
            "output_ids": [101],
            "meta_info": {
                "output_token_logprobs": [[-0.1, 101]],
                "output_token_logprobs_length": 1,
            },
        }
    )

    with pytest.raises(ValueError, match="terminal finish_reason"):
        accumulator.finish(lambda token_ids: "a")


def test_sse_error_event_is_not_silently_ignored():
    with pytest.raises(ValueError, match="error event"):
        asyncio.run(
            consume_sglang_sse(
                _lines('event: error', 'data: {"error":{"message":"boom"}}', "", "data: [DONE]", ""),
                decode=lambda token_ids: "",
            )
        )
