from abc import ABC, abstractmethod
from typing import Any, Callable, Type
from pydantic import BaseModel, Field, validator


from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from gwenflow.tools.utils import function_to_json
from gwenflow.utils import logger


class BaseTool(BaseModel, ABC):

    # class _ArgsSchemaPlaceholder(BaseModel):
    #     pass

    name: str
    """The unique name of the tool that clearly communicates its purpose."""

    description: str
    """Used to tell the model how to use the tool."""

    # args_schema: Type[BaseModel] = Field(default_factory=_ArgsSchemaPlaceholder)
    # """The schema for the arguments that the tool accepts."""

    openai_schema: dict = None
    """OpenAI JSON schema"""

    tool_type: str = "function"
    """Tool type: function, langchain, llamaindex."""

    # @validator("args_schema", always=True, pre=True)
    # def _default_args_schema(cls, v: Type[BaseModel]) -> Type[BaseModel]:
    #     if not isinstance(v, cls._ArgsSchemaPlaceholder):
    #         return v

    #     return type(
    #         f"{cls.__name__}Schema",
    #         (BaseModel,),
    #         {
    #             "__annotations__": {
    #                 k: v for k, v in cls._run.__annotations__.items() if k != "return"
    #             },
    #         },
    #     )

    def run(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """Actual implementation of the tool."""

    @classmethod
    def from_langchain(cls, tool: StructuredTool) -> "BaseTool":
        if cls == Tool:
            if tool.run is None:
                raise ValueError("StructuredTool must have a callable 'func'")
            return Tool(
                name=tool.name,
                description=tool.description,
                # args_schema=tool.args_schema,
                openai_schema=convert_to_openai_tool(tool),
                func=tool.run,
                tool_type="langchain",
            )
        raise NotImplementedError(f"from_langchain not implemented for {cls.__name__}")

    @classmethod
    def from_function(cls, func: Callable) -> "BaseTool":
        if cls == Tool:
            def _make_with_name(tool_name: str) -> Callable:
                def _make_tool(f: Callable) -> BaseTool:
                    if f.__doc__ is None:
                        raise ValueError("Function must have a docstring")
                    if f.__annotations__ is None:
                        raise ValueError("Function must have type annotations")

                    class_name = "".join(tool_name.split()).title()
                    args_schema = type(
                        class_name,
                        (BaseModel,),
                        {
                            "__annotations__": {
                                k: v for k, v in f.__annotations__.items() if k != "return"
                            },
                        },
                    )

                    return Tool(
                        name=tool_name,
                        description=f.__doc__,
                        func=f,
                        # args_schema=args_schema,
                        openai_schema=function_to_json(f),
                        tool_type="function",
                    )

                return _make_tool

            if callable(func):
                return _make_with_name(func.__name__)(func)

            if isinstance(func, str):
                return _make_with_name(func)

        raise ValueError(f"Invalid arguments for {cls.__name__}")


class Tool(BaseTool):

    func: Callable
    """The function that will be executed when the tool is called."""

    def _run(self, **kwargs: Any) -> Any:
        if self.tool_type == "langchain":
            return self.func(kwargs)
        return self.func(**kwargs)
