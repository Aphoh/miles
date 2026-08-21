"""Validation and accumulation for Dynamo's native SGLang ``/generate`` stream."""

import json
from collections.abc import AsyncIterator, Callable
from typing import Any


def _merge_incremental(previous: dict[str, Any] | None, chunk: dict[str, Any]) -> dict[str, Any]:
    """Merge one disjoint SGLang chunk while preserving its complete metadata.

    The cumulative-length check is adapted from THUDM/slime PR 2272
    (Apache-2.0), ``slime.rollout.streaming_utils``. Unlike Slime's rollout
    helper, this keeps the complete logprob tuples and top-logprob rows Miles
    needs when assembling training records.
    """
    meta_info = chunk.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("SGLang streaming responses must include an object meta_info.")
    if "output_token_logprobs_length" not in meta_info:
        raise ValueError("SGLang streaming responses must include output_token_logprobs_length.")

    pairs = meta_info.get("output_token_logprobs") or []
    if not isinstance(pairs, list) or any(not isinstance(item, (list, tuple)) or len(item) < 2 for item in pairs):
        raise ValueError("SGLang output_token_logprobs must be a list of [logprob, token_id, ...] entries.")

    previous_pairs = previous["output_token_logprobs"] if previous else []
    reported_length = int(meta_info["output_token_logprobs_length"])
    expected_length = len(previous_pairs) + len(pairs)
    if reported_length != expected_length:
        raise ValueError(f"SGLang incremental stream has inconsistent output_token_logprobs_length: received={len(pairs)}, previous={len(previous_pairs)}, expected={expected_length}, reported={reported_length}.")

    chunk_ids = [int(item[1]) for item in pairs]
    raw_output_ids = chunk.get("output_ids")
    if raw_output_ids is not None:
        if not isinstance(raw_output_ids, list):
            raise ValueError("SGLang output_ids must be a list when present.")
        if [int(token_id) for token_id in raw_output_ids] != chunk_ids:
            raise ValueError("SGLang output_ids disagree with meta_info.output_token_logprobs token IDs.")

    top_rows = meta_info.get("output_top_logprobs")
    previous_top_rows = previous["output_top_logprobs"] if previous else None
    if top_rows is not None:
        if not isinstance(top_rows, list) or len(top_rows) != len(pairs):
            raise ValueError("SGLang output_top_logprobs must have one row per output token in the chunk.")
        output_top_logprobs = [*(previous_top_rows or []), *top_rows]
    else:
        output_top_logprobs = previous_top_rows

    chunk_text = chunk.get("text")
    if chunk_text is not None and not isinstance(chunk_text, str):
        raise ValueError("SGLang stream chunk text must be a string or null.")
    previous_text = previous["text"] if previous else ""
    text = None if previous_text is None or (chunk_text is None and pairs) else previous_text + (chunk_text or "")

    return {
        "latest_chunk": dict(chunk),
        "meta_info": {**(previous["meta_info"] if previous else {}), **meta_info},
        "output_token_logprobs": [*previous_pairs, *pairs],
        "output_top_logprobs": output_top_logprobs,
        "text": text,
    }


def _finish_incremental(
    response: dict[str, Any] | None,
    decode: Callable[[list[int]], str],
) -> dict[str, Any]:
    """Validate terminal state and materialize the non-streaming SGLang shape."""
    if response is None:
        raise ValueError("SGLang streaming response ended without any data chunks.")

    meta_info = dict(response["meta_info"])
    if not meta_info.get("finish_reason"):
        raise ValueError("SGLang streaming response ended without a terminal finish_reason.")

    pairs = list(response["output_token_logprobs"])
    output_ids = [int(pair[1]) for pair in pairs]
    reported_length = int(meta_info["output_token_logprobs_length"])
    completion_tokens = int(meta_info.get("completion_tokens", reported_length))
    if reported_length != len(pairs) or completion_tokens != len(pairs):
        raise ValueError(f"SGLang terminal token counts disagree: completion_tokens={completion_tokens}, reported_length={reported_length}, logprob_pairs={len(pairs)}.")

    top_rows = response["output_top_logprobs"]
    if top_rows is not None and len(top_rows) != len(pairs):
        raise ValueError(f"SGLang terminal output_top_logprobs length disagrees with completion tokens: top_logprobs={len(top_rows)}, completion_tokens={len(pairs)}.")

    meta_info["completion_tokens"] = len(pairs)
    meta_info["output_token_logprobs_length"] = len(pairs)
    meta_info["output_token_logprobs"] = pairs
    if top_rows is not None:
        meta_info["output_top_logprobs"] = list(top_rows)

    result = dict(response["latest_chunk"])
    result["text"] = decode(output_ids) if response["text"] is None else response["text"]
    result["output_ids"] = output_ids
    result["meta_info"] = meta_info
    return result


async def consume_sglang_sse(
    lines: AsyncIterator[str],
    *,
    decode: Callable[[list[int]], str],
) -> dict[str, Any]:
    """Consume Dynamo's single-line, incremental SGLang SSE response."""
    response = None
    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue

        data = line[len("data:") :].lstrip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SGLang SSE event is not valid JSON: {data[:160]!r}.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"SGLang SSE data must decode to an object, got {type(payload).__name__}.")
        if payload.get("error"):
            raise ValueError(f"SGLang streaming backend returned an error event: {payload}.")
        response = _merge_incremental(response, payload)

    return _finish_incremental(response, decode)
