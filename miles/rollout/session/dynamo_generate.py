"""OpenAI chat semantics over Dynamo's native streaming SGLang endpoint."""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ToolChoice
from sglang.srt.function_call.core_types import ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.parser.reasoning_parser import ReasoningParser

from miles.rollout.session.errors import MessageValidationError, UpstreamResponseError
from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DynamoChatRequest:
    """The validated chat request and native request sent to Dynamo."""

    request: ChatCompletionRequest
    payload: dict[str, Any]
    response_id: str


class DynamoGenerateChatAdapter:
    """Translate pretokenized chat requests to and from native ``/generate``."""

    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.reasoning_parser, self.tool_call_parser = resolve_reasoning_and_tool_call_parser(
            getattr(args, "tito_model", "default"),
            getattr(args, "sglang_reasoning_parser", None),
            getattr(args, "sglang_tool_call_parser", None),
        )
        logger.info(
            "[session] Dynamo /generate chat adapter: reasoning_parser=%r tool_call_parser=%r",
            self.reasoning_parser,
            self.tool_call_parser,
        )

    def build_generate_request(self, request_body: dict[str, Any]) -> DynamoChatRequest:
        """Validate an extended chat request and build native SGLang input."""
        try:
            request = ChatCompletionRequest.model_validate(request_body)
        except ValidationError as exc:
            raise MessageValidationError(f"invalid chat completion request: {exc}") from exc

        if request.n != 1:
            raise MessageValidationError("Dynamo native /generate supports only n=1.")
        if not request.input_ids:
            raise MessageValidationError("Dynamo native /generate requires non-empty pretokenized input_ids.")
        if isinstance(request.rid, list):
            raise MessageValidationError("Dynamo native /generate requires rid to be a string when provided.")

        require_reasoning = self._requires_reasoning(request)
        tool_call_constraint = self._tool_call_constraint(request, require_reasoning=require_reasoning)
        sampling_params = request.to_sampling_params(
            stop=request.stop or [],
            model_generation_config={},
            tool_call_constraint=tool_call_constraint,
        )
        response_id = request.rid or f"chatcmpl-{uuid.uuid4().hex}"
        payload = self._native_payload(
            request,
            request_body=request_body,
            sampling_params=sampling_params,
            response_id=response_id,
            require_reasoning=require_reasoning,
        )
        return DynamoChatRequest(request=request, payload=payload, response_id=response_id)

    def build_chat_response(self, prepared: DynamoChatRequest, native_response: dict[str, Any]) -> dict[str, Any]:
        """Apply SGLang's non-streaming reasoning/tool parsing to one result."""
        meta_info = native_response.get("meta_info")
        if not isinstance(meta_info, dict):
            raise UpstreamResponseError("Dynamo /generate response is missing meta_info.")
        text = native_response.get("text")
        if not isinstance(text, str):
            raise UpstreamResponseError("Dynamo /generate response text must be a string.")

        request = prepared.request
        reasoning_content, content = self._parse_reasoning(text, request)
        finish_reason = _finish_reason_type(meta_info.get("finish_reason"))
        tool_calls, content, finish_reason = self._parse_tool_calls(
            content,
            request=request,
            finish_reason=finish_reason,
        )
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = tool_calls

        prompt_tokens = int(meta_info.get("prompt_tokens", len(request.input_ids or [])))
        completion_tokens = int(meta_info.get("completion_tokens", 0))
        choice: dict[str, Any] = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            # Private compatibility lane consumed by Miles' sample assembler.
            # The client renderer removes this entire field for this backend.
            "meta_info": meta_info,
        }
        if request.logprobs:
            choice["logprobs"] = _openai_logprobs(meta_info, self.tokenizer)
        if request.return_prompt_token_ids:
            choice["prompt_token_ids"] = list(request.input_ids or [])

        return {
            "id": prepared.response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Decode a complete output when SGLang omitted stream text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _native_payload(
        self,
        request: ChatCompletionRequest,
        *,
        request_body: dict[str, Any],
        sampling_params: dict[str, Any],
        response_id: str,
        require_reasoning: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "rid": response_id,
            "input_ids": list(request.input_ids or []),
            "sampling_params": sampling_params,
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": request.top_logprobs or 0,
            "return_text_in_logprobs": True,
            "stream": True,
            "return_hidden_states": request.return_hidden_states,
            "return_routed_experts": request.return_routed_experts,
            "routed_experts_start_len": request.routed_experts_start_len,
            "return_indexer_topk": bool(request_body.get("return_indexer_topk", False)),
            "require_reasoning": require_reasoning,
            "return_prompt_token_ids": request.return_prompt_token_ids,
        }
        cache_salt = request.cache_salt
        extra_key = request.extra_key
        if not isinstance(cache_salt, (str, type(None))) or not isinstance(extra_key, (str, type(None))):
            raise MessageValidationError("cache_salt and extra_key must be strings when provided.")
        combined_extra_key = "".join(value for value in (cache_salt, extra_key) if value)
        for name, value in (
            ("lora_path", request.lora_path),
            ("session_id", request.session_id),
            ("extra_key", combined_extra_key or None),
            ("priority", request.priority),
            ("custom_logit_processor", request.custom_logit_processor),
        ):
            if value is not None:
                payload[name] = value
        return payload

    def _tool_call_constraint(self, request: ChatCompletionRequest, *, require_reasoning: bool):
        if not request.tools or request.tool_choice == "none":
            return None
        request.skip_special_tokens = False
        constraint = None
        if self.tool_call_parser:
            parser = FunctionCallParser(
                request.tools,
                self.tool_call_parser,
                tokenizer=self.tokenizer,
            )
            constraint = parser.get_structure_constraint(
                request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                thinking_mode=require_reasoning and self.reasoning_parser is None,
            )
        if constraint is None and _tool_choice_is_required(request.tool_choice):
            schema = get_json_schema_constraint(
                request.tools,
                request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
            )
            constraint = ("json_schema", schema)
        return constraint

    def _requires_reasoning(self, request: ChatCompletionRequest) -> bool:
        if not self.reasoning_parser:
            return False
        kwargs = request.chat_template_kwargs or {}
        if self.reasoning_parser == "minimax-m3":
            return kwargs.get("thinking_mode") == "enabled"
        if self.reasoning_parser == "hunyuan":
            return request.reasoning_effort not in (None, "none", "no_think")
        try:
            parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                request=request,
                tokenizer=self.tokenizer,
            )
        except Exception as exc:
            raise MessageValidationError(f"failed to initialize reasoning parser: {exc}") from exc
        mode = getattr(parser.detector, "reasoning_default", None)
        if mode == "always":
            return True
        if mode == "mistral":
            return request.reasoning_effort is not None and request.reasoning_effort != "none"
        if mode in ("thinking", "enable_thinking"):
            return not kwargs or kwargs.get(mode) is not False
        if mode in ("explicit_thinking", "explicit_enable_thinking"):
            return kwargs.get(mode.removeprefix("explicit_")) is True
        return False

    def _parse_reasoning(self, text: str, request: ChatCompletionRequest) -> tuple[str | None, str]:
        if not self.reasoning_parser or not request.separate_reasoning:
            return None, text
        try:
            parser = ReasoningParser(
                model_type=self.reasoning_parser,
                stream_reasoning=False,
                force_reasoning=self._requires_reasoning(request),
                request=request,
                tokenizer=self.tokenizer,
            )
            reasoning, content = parser.parse_non_stream(text)
        except Exception as exc:
            raise UpstreamResponseError(f"failed to parse reasoning content: {exc}") from exc
        return reasoning, content or ""

    def _parse_tool_calls(
        self,
        text: str,
        *,
        request: ChatCompletionRequest,
        finish_reason: str,
    ) -> tuple[list[dict[str, Any]] | None, str, str]:
        if not request.tools or request.tool_choice == "none":
            return None, text, finish_reason

        required = _tool_choice_is_required(request.tool_choice)
        parser = FunctionCallParser(request.tools, self.tool_call_parser, tokenizer=self.tokenizer) if self.tool_call_parser else None
        should_try_native = parser is not None and (not required or parser.detector.supports_structural_tag())
        if should_try_native and parser.has_tool_call(text):
            try:
                content, calls = parser.parse_non_stream(text)
                tool_calls = [self._render_tool_call(call, request=request) for call in calls]
                parsed_finish = "tool_calls" if finish_reason == "stop" and tool_calls else finish_reason
                return tool_calls or None, content, parsed_finish
            except Exception:
                logger.exception("Dynamo chat adapter failed to parse a native tool call")
                return None, text, finish_reason

        if required:
            try:
                calls = json.loads(text)
                if not isinstance(calls, list):
                    raise ValueError("required tool output is not a JSON array")
                rendered = []
                for index, call in enumerate(calls):
                    item = ToolCallItem(
                        tool_index=index,
                        name=call["name"],
                        parameters=json.dumps(call["parameters"], ensure_ascii=False),
                    )
                    rendered.append(self._render_tool_call(item, request=request))
                parsed_finish = "tool_calls" if finish_reason == "stop" else finish_reason
                return rendered, "", parsed_finish
            except Exception:
                logger.exception("Dynamo chat adapter failed to parse required JSON tool calls")
        return None, text, finish_reason

    def _render_tool_call(self, call: ToolCallItem, *, request: ChatCompletionRequest) -> dict[str, Any]:
        history_count = sum(len(message.tool_calls or []) for message in request.messages if message.role == "assistant")
        if self.tool_call_parser == "kimi_k2":
            tool_call_id = f"functions.{call.name}:{history_count + call.tool_index}"
        elif self.tool_call_parser == "kimi_k2_raw_id":
            tool_call_id = getattr(call, "tool_call_id", None) or f"functions.{call.name}:{call.tool_index}"
        else:
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        return {
            "id": tool_call_id,
            "index": call.tool_index,
            "type": "function",
            "function": {"name": call.name, "arguments": call.parameters},
        }


def _tool_choice_is_required(tool_choice: Any) -> bool:
    return tool_choice == "required" or isinstance(tool_choice, ToolChoice)


def _finish_reason_type(finish_reason: Any) -> str:
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    if not finish_reason:
        raise UpstreamResponseError("Dynamo /generate response is missing a terminal finish reason.")
    return str(finish_reason)


def _openai_logprobs(meta_info: dict[str, Any], tokenizer) -> dict[str, Any]:
    pairs = meta_info.get("output_token_logprobs") or []
    top_rows = meta_info.get("output_top_logprobs") or []
    content = []
    for index, pair in enumerate(pairs):
        token_id = int(pair[1])
        token_text = pair[2] if len(pair) > 2 and isinstance(pair[2], str) else tokenizer.decode([token_id])
        top_logprobs = []
        if index < len(top_rows) and top_rows[index]:
            for top_pair in top_rows[index]:
                top_id = int(top_pair[1])
                top_text = top_pair[2] if len(top_pair) > 2 and isinstance(top_pair[2], str) else tokenizer.decode([top_id])
                top_logprobs.append(
                    {
                        "token": top_text,
                        "bytes": list(top_text.encode("utf-8")),
                        "logprob": top_pair[0],
                    }
                )
        content.append(
            {
                "token": token_text,
                "bytes": list(token_text.encode("utf-8")),
                "logprob": pair[0],
                "top_logprobs": top_logprobs,
            }
        )
    return {"content": content}
