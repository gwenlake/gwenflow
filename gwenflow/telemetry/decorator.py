import functools
import inspect
import json
from typing import Any

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import StatusCode, Tracer


def safe_serialize(obj: Any) -> str:
    try:
        if hasattr(obj, "model_dump_json"):
            return obj.model_dump_json()
        if hasattr(obj, "dict"):
            return json.dumps(obj.dict(), default=str)
        return str(obj)
    except Exception:
        return str(obj)

class DecoratorTracer:
    def __init__(self, tracer_name: str = "gwenflow"):
        self.tracer_name = tracer_name

    @property
    def tracer(self) -> Tracer:
        return trace.get_tracer(self.tracer_name)

    def _get_input_value(self, func, args, kwargs) -> str:
        """Fonction to capture the user's input."""
        try:
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            arg_dict = dict(bound_args.arguments)

            keys_to_remove = [k for k in arg_dict.keys() if k.startswith("self")]
            for k in keys_to_remove:
                del arg_dict[k]

            if "input" in arg_dict:
                return safe_serialize(arg_dict["input"])
            if "query" in arg_dict:
                return safe_serialize(arg_dict["query"])

            return json.dumps({k: str(v) for k, v in arg_dict.items()}, default=str)
        except Exception:
            return "Error capturing inputs"

    def _capture_usage(self, span, result):
        """Helper to extract usage for the telemetry endpoint."""
        if not hasattr(result, "usage") or not result.usage:
            return

        usage = result.usage

        input_tokens = getattr(usage, "input_tokens", None) 
        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", None)

        if input_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(input_tokens))

        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)

        if output_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(output_tokens))

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None:
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total_tokens))

        input_details = getattr(usage, "input_tokens_details", None)
        if input_details:
            cached = getattr(input_details, "cached_tokens", None)
            if cached is not None:
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT, int(cached))

        output_details = getattr(usage, "output_tokens_details", None)
        if output_details is None:
            output_details = getattr(usage, "completion_tokens_details", None)
        if output_details:
            reasoning = getattr(output_details, "reasoning_tokens", None)
            if reasoning is not None:
                span.set_attribute(SpanAttributes.LLM_COST_COMPLETION_DETAILS_REASONING, int(reasoning))

    def _inject_topology(self, span, instance):
        if hasattr(instance, "parent_flow_id") and instance.parent_flow_id:
            span.set_attribute("gwenflow.topology.parent_id", str(instance.parent_flow_id))
        if hasattr(instance, "depends_on") and instance.depends_on:
            span.set_attribute("gwenflow.topology.depends_on", str(instance.depends_on))

    def llm(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_inst, *args, **kwargs):
                span_name = name or f"LLM:{getattr(self_inst, 'model', 'unknown')}"

                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)

                    if hasattr(self_inst, "model"):
                        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self_inst.model)

                    if hasattr(self_inst, "_model_params"):
                        span.set_attribute(SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(self_inst._model_params, default=str))

                    input_val = self._get_input_value(func, (self_inst, *args), kwargs)
                    span.set_attribute(SpanAttributes.INPUT_VALUE, input_val)

                    try:
                        result = func(self_inst, *args, **kwargs)
                        self._capture_usage(span, result)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(result))
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def agent(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_agent, *args, **kwargs):
                span_name = name or f"Agent:{getattr(self_agent, 'name', 'unknown')}"

                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("agent.name", self_agent.name)
                    span.set_attribute("agent.model", self_agent.llm.model)

                    if hasattr(self_agent, "tools"):
                        span.set_attribute("agent.tools_available", str([t.name for t in self_agent.tools]))

                    sess_id = kwargs.get("session_id") or getattr(self_agent, "session_id", None)
                    if sess_id: span.set_attribute(SpanAttributes.SESSION_ID, str(sess_id))

                    span.set_attribute(SpanAttributes.INPUT_VALUE, self._get_input_value(func, (self_agent, *args), kwargs))

                    try:
                        result = func(self_agent, *args, **kwargs)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(result))
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def tool(self, name: str = None):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_inst, *args, **kwargs):
                span_name = name or f"Tool:{func.__name__}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value)
                    span.set_attribute(SpanAttributes.INPUT_VALUE, self._get_input_value(func, (self_inst, *args), kwargs))

                    try:
                        result = func(self_inst, *args, **kwargs)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(result))
                        span.set_status(StatusCode.OK)
                        return result
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator


    def stream(self, name: str = None, kind: str = OpenInferenceSpanKindValues.AGENT.value):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self_inst, *args, **kwargs):
                actual_kind = kind
                if hasattr(self_inst, "model") and hasattr(self_inst, "get_client"):
                     actual_kind = OpenInferenceSpanKindValues.LLM.value

                span_name = name or f"Stream:{getattr(self_inst, 'name', 'unknown')}"

                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, actual_kind)
                    self._inject_topology(span, self_inst)

                    full_content = []
                    try:
                        for chunk in func(self_inst, *args, **kwargs):
                            content = None
                            if hasattr(chunk, "content"): content = chunk.content
                            elif hasattr(chunk, "get_text"): content = chunk.get_text()
                            elif isinstance(chunk, str): content = chunk
                            if content: full_content.append(str(content))
                            yield chunk

                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(full_content))
                        span.set_status(StatusCode.OK)
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    def astream(self, name: str = None, kind: str = OpenInferenceSpanKindValues.AGENT.value):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(self_inst, *args, **kwargs):
                actual_kind = kind
                if hasattr(self_inst, "model") and hasattr(self_inst, "get_client"):
                     actual_kind = OpenInferenceSpanKindValues.LLM.value

                span_name = name or f"AsyncStream:{getattr(self_inst, 'name', 'unknown')}"
                with self.tracer.start_as_current_span(span_name) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, actual_kind)
                    self._inject_topology(span, self_inst)

                    full_content = []
                    try:
                        async for chunk in func(self_inst, *args, **kwargs):
                            content = None
                            if hasattr(chunk, "content"): content = chunk.content
                            elif hasattr(chunk, "get_text"): content = chunk.get_text()
                            elif isinstance(chunk, str): content = chunk

                            if content: full_content.append(str(content))
                            yield chunk

                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(full_content))
                        span.set_status(StatusCode.OK)
                    except Exception as e:
                        span.set_status(StatusCode.ERROR, str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

    # def flow(self, name: str = None):
    #     def decorator(func):
    #         @functools.wraps(func)
    #         def wrapper(*args, **kwargs):
    #             span_name = name or f"Flow:{func.__name__}"
    #             with self.tracer.start_as_current_span(span_name) as span:
    #                 span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
    #                 span.set_attribute(SpanAttributes.INPUT_VALUE, self._get_input_value(func, args, kwargs))

    #                 try:
    #                     result = func(*args, **kwargs)
    #                     span.set_attribute(SpanAttributes.OUTPUT_VALUE, safe_serialize(result))
    #                     span.set_status(StatusCode.OK)
    #                     return result
    #                 except Exception as e:
    #                     span.set_status(StatusCode.ERROR, str(e))
    #                     span.record_exception(e)
    #                     raise
    #         return wrapper
    #     return decorator

Tracer = DecoratorTracer()