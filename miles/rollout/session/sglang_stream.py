"""Validation and accumulation for native SGLang ``/generate`` streams."""

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SGLangStreamAccumulator:
    """Accumulate one SGLang SSE response into its non-streaming shape.

    The length field is cumulative in both SGLang stream modes. Dynamo's
    native ``/generate`` frontend uses incremental mode, while accepting both
    here keeps the adapter usable with a direct SGLang endpoint as well.

    The length-validation algorithm is adapted from THUDM/slime PR 2272
    (Apache-2.0), ``slime.rollout.streaming_utils``.
    """

    output_mode: Literal["incremental", "cumulative"] = "incremental"
    output_length: int = 0
    output_ids: list[int] = field(default_factory=list)
    output_token_logprobs: list[list[Any] | tuple[Any, ...]] = field(default_factory=list)
    output_top_logprobs: list[Any] = field(default_factory=list)
    latest_meta_info: dict[str, Any] = field(default_factory=dict)
    latest_chunk: dict[str, Any] = field(default_factory=dict)
    _text_chunks: list[str] = field(default_factory=list)
    _cumulative_text: str | None = None
    _decode_text: bool = False

    def add(self, chunk: dict[str, Any]) -> None:
        """Validate and append one decoded SSE data object."""
        meta_info = chunk.get("meta_info")
        if not isinstance(meta_info, dict):
            raise ValueError("SGLang streaming responses must include an object meta_info.")
        if "output_token_logprobs_length" not in meta_info:
            raise ValueError("SGLang streaming responses must include output_token_logprobs_length.")

        pairs = meta_info.get("output_token_logprobs") or []
        if not isinstance(pairs, list) or any(not isinstance(item, (list, tuple)) or len(item) < 2 for item in pairs):
            raise ValueError("SGLang output_token_logprobs must be a list of [logprob, token_id, ...] entries.")
        chunk_ids = [int(item[1]) for item in pairs]
        reported_length = int(meta_info["output_token_logprobs_length"])
        previous_length = self.output_length
        cumulative = self.output_mode == "cumulative"
        expected_length = len(pairs) if cumulative else previous_length + len(pairs)
        if expected_length != reported_length:
            raise ValueError(f"SGLang {self.output_mode} streaming output has inconsistent " "output_token_logprobs_length: " f"received={len(pairs)}, previous={previous_length}, " f"expected={expected_length}, reported={reported_length}.")
        if reported_length < previous_length:
            raise ValueError("SGLang cumulative streaming output length decreased: " f"previous={previous_length}, reported={reported_length}.")

        raw_output_ids = chunk.get("output_ids")
        if raw_output_ids is not None:
            if not isinstance(raw_output_ids, list):
                raise ValueError("SGLang output_ids must be a list when present.")
            response_ids = [int(token_id) for token_id in raw_output_ids]
            if response_ids != chunk_ids:
                raise ValueError("SGLang output_ids disagree with meta_info.output_token_logprobs token IDs: " f"output_ids={response_ids}, logprob_ids={chunk_ids}.")

        if cumulative:
            if chunk_ids[:previous_length] != self.output_ids:
                raise ValueError("SGLang cumulative streaming output changed an already emitted token prefix.")
            new_pairs = pairs[previous_length:]
            self.output_token_logprobs = list(pairs)
            self.output_ids = chunk_ids
        else:
            new_pairs = pairs
            self.output_token_logprobs.extend(pairs)
            self.output_ids.extend(chunk_ids)

        self._add_top_logprobs(meta_info, previous_length=previous_length, cumulative=cumulative)
        self._add_text(chunk.get("text"), has_new_tokens=bool(new_pairs), cumulative=cumulative)
        self.output_length = reported_length
        self.latest_meta_info.update(meta_info)
        self.latest_chunk = dict(chunk)

    def finish(self, decode: Callable[[list[int]], str]) -> dict[str, Any]:
        """Return a complete native response after validating terminal state."""
        if not self.latest_chunk:
            raise ValueError("SGLang streaming response ended without any data chunks.")
        if not self.latest_meta_info.get("finish_reason"):
            raise ValueError("SGLang streaming response ended without a terminal finish_reason.")
        completion_tokens = self.latest_meta_info.get("completion_tokens", self.output_length)
        if int(completion_tokens) != self.output_length:
            raise ValueError("SGLang terminal completion_tokens disagrees with the streamed output length: " f"completion_tokens={completion_tokens}, output_length={self.output_length}.")

        meta_info = dict(self.latest_meta_info)
        meta_info["completion_tokens"] = self.output_length
        meta_info["output_token_logprobs_length"] = self.output_length
        meta_info["output_token_logprobs"] = list(self.output_token_logprobs)
        if self.output_top_logprobs:
            meta_info["output_top_logprobs"] = list(self.output_top_logprobs)

        response = dict(self.latest_chunk)
        response["text"] = self._response_text(decode)
        response["output_ids"] = list(self.output_ids)
        response["meta_info"] = meta_info
        return response

    def _add_top_logprobs(self, meta_info: dict[str, Any], *, previous_length: int, cumulative: bool) -> None:
        top_logprobs = meta_info.get("output_top_logprobs")
        if top_logprobs is None:
            return
        if not isinstance(top_logprobs, list):
            raise ValueError("SGLang output_top_logprobs must be a list when present.")
        if cumulative:
            if len(top_logprobs) != self.output_length and len(top_logprobs) != int(meta_info["output_token_logprobs_length"]):
                raise ValueError("SGLang cumulative output_top_logprobs has an inconsistent length.")
            if top_logprobs[:previous_length] != self.output_top_logprobs:
                raise ValueError("SGLang cumulative output_top_logprobs changed an already emitted prefix.")
            self.output_top_logprobs = list(top_logprobs)
        else:
            self.output_top_logprobs.extend(top_logprobs)

    def _add_text(self, text: Any, *, has_new_tokens: bool, cumulative: bool) -> None:
        if text is not None and not isinstance(text, str):
            raise ValueError("SGLang stream chunk text must be a string or null.")
        if cumulative:
            if text is not None:
                self._cumulative_text = text
            elif has_new_tokens:
                self._decode_text = True
        elif text is None:
            if has_new_tokens:
                self._decode_text = True
        elif text:
            self._text_chunks.append(text)

    def _response_text(self, decode: Callable[[list[int]], str]) -> str:
        if self.output_mode == "cumulative":
            if self._cumulative_text is not None and not self._decode_text:
                return self._cumulative_text
            return decode(self.output_ids)
        if self._decode_text:
            return decode(self.output_ids)
        return "".join(self._text_chunks)


async def consume_sglang_sse(
    lines: AsyncIterator[str],
    *,
    decode: Callable[[list[int]], str],
    output_mode: Literal["incremental", "cumulative"] = "incremental",
) -> dict[str, Any]:
    """Consume SSE lines and materialize one complete SGLang response."""
    accumulator = SGLangStreamAccumulator(output_mode=output_mode)
    event_name = "message"
    data_lines: list[str] = []

    async def flush_event() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return
        data = "\n".join(data_lines).strip()
        data_lines = []
        current_event = event_name
        event_name = "message"
        if not data or data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SGLang SSE event is not valid JSON: {data[:160]!r}.") from exc
        if current_event == "error" or (isinstance(payload, dict) and payload.get("error")):
            raise ValueError(f"SGLang streaming backend returned an error event: {payload}.")
        if not isinstance(payload, dict):
            raise ValueError(f"SGLang SSE data must decode to an object, got {type(payload).__name__}.")
        accumulator.add(payload)

    async for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line:
            await flush_event()
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[len("event:") :].strip() or "message"
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    await flush_event()
    return accumulator.finish(decode)
