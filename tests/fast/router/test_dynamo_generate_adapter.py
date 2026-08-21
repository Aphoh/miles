from types import SimpleNamespace

from miles.rollout.session.dynamo_generate import DynamoGenerateChatAdapter


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def _adapter() -> DynamoGenerateChatAdapter:
    return DynamoGenerateChatAdapter(
        _Tokenizer(),
        SimpleNamespace(
            tito_model="default",
            sglang_reasoning_parser=None,
            sglang_tool_call_parser=None,
        ),
    )


def _tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }


def test_build_native_generate_request_preserves_sampling_and_metadata_flags():
    prepared = _adapter().build_generate_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "input_ids": [1, 2, 3],
            "temperature": 0.25,
            "max_completion_tokens": 17,
            "logprobs": True,
            "return_routed_experts": True,
            "routed_experts_start_len": 2,
            "return_indexer_topk": True,
        }
    )

    assert prepared.payload["input_ids"] == [1, 2, 3]
    assert prepared.payload["stream"] is True
    assert prepared.payload["return_logprob"] is True
    assert prepared.payload["return_routed_experts"] is True
    assert prepared.payload["routed_experts_start_len"] == 2
    assert prepared.payload["return_indexer_topk"] is True
    assert prepared.payload["sampling_params"]["temperature"] == 0.25
    assert prepared.payload["sampling_params"]["max_new_tokens"] == 17


def test_build_chat_response_keeps_private_meta_for_miles():
    adapter = _adapter()
    prepared = adapter.build_generate_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
            "input_ids": [1, 2, 3],
            "logprobs": True,
        }
    )
    meta_info = {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "output_token_logprobs": [[-0.1, 104, "h"], [-0.2, 105, "i"]],
        "finish_reason": {"type": "stop"},
        "routed_experts": [[1, 2]],
    }

    response = adapter.build_chat_response(prepared, {"text": "hi", "meta_info": meta_info})
    choice = response["choices"][0]

    assert choice["message"] == {"role": "assistant", "content": "hi"}
    assert choice["finish_reason"] == "stop"
    assert choice["meta_info"] is meta_info
    assert choice["logprobs"]["content"][0]["token"] == "h"
    assert response["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_model_specific_tool_parser_runs_after_generate():
    adapter = _adapter()
    adapter.tool_call_parser = "qwen25"
    prepared = adapter.build_generate_request(
        {
            "messages": [{"role": "user", "content": "weather?"}],
            "input_ids": [1, 2, 3],
            "tools": [_tool()],
            "tool_choice": "auto",
        }
    )
    text = (
        "Let me check.\n<tool_call>\n"
        '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call>"
    )
    response = adapter.build_chat_response(
        prepared,
        {
            "text": text,
            "meta_info": {
                "completion_tokens": 1,
                "output_token_logprobs": [[-0.1, 1]],
                "finish_reason": {"type": "stop"},
            },
        },
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city": "Paris"}',
    }


def test_required_tool_choice_uses_generic_json_fallback_without_parser():
    adapter = _adapter()
    prepared = adapter.build_generate_request(
        {
            "messages": [{"role": "user", "content": "weather?"}],
            "input_ids": [1, 2, 3],
            "tools": [_tool()],
            "tool_choice": "required",
        }
    )
    response = adapter.build_chat_response(
        prepared,
        {
            "text": '[{"name":"get_weather","parameters":{"city":"Paris"}}]',
            "meta_info": {
                "completion_tokens": 1,
                "output_token_logprobs": [[-0.1, 1]],
                "finish_reason": {"type": "stop"},
            },
        },
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == ""
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
