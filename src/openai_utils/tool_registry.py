import json
from dataclasses import dataclass, field
from typing import Callable, Any

from pydantic import TypeAdapter
from openai.types.responses import FunctionToolParam
from openai.types.responses.response_input_param import FunctionCallOutput


@dataclass(kw_only=True)
class Tool:
    """
    Represents a callable tool with a defined schema and description.

    This class wraps a Python callable, its JSON schema, and an optional description, allowing it to be registered and invoked in a structured way (e.g., for OpenAI function calling).
    """

    name: str
    function: Callable[..., Any]
    schema: dict[str, Any]
    description: str | None = None

    def __call__(self, call_id: str, kwargs: str | dict[str, Any]) -> FunctionCallOutput:
        """
        Invoke the tool with the given arguments and return a FunctionCallOutput.

        Example:
            ```python
            tool = Tool.from_function(my_func)
            output = tool(call_id="abc123", kwargs={"x": 1, "y": 2})
            ```
        """
        args: dict[str, Any] = json.loads(kwargs) if isinstance(kwargs, str) else kwargs

        return FunctionCallOutput(type="function_call_output", call_id=call_id, output=str(self.function(**args)))

    def to_param(self) -> FunctionToolParam:
        """
        Convert this tool to a FunctionToolParam for OpenAI API registration.
        """
        return FunctionToolParam(
            name=self.name, parameters=self.schema, description=self.description, strict=True, type="function"
        )

    @staticmethod
    def from_function(function: Callable[..., Any], description: str | None = None) -> "Tool":
        """
        Create a Tool instance from a Python function, validating type hints and generating a schema.

        Raises:
            TypeError: If any function argument lacks a type hint.

        Example:
            ```python
            def add(x: int, y: int) -> int:
                return x + y
            tool = Tool.from_function(add, description="Adds two numbers.")
            ```
        """
        # Validate that the function has type hints for it's arguments
        schema = TypeAdapter(function).json_schema(mode="serialization")
        for name, prop_schema in schema["properties"].items():
            if "type" not in prop_schema:
                raise TypeError(f"Function {function.__name__} has no type hints for parameter {name}")

        return Tool(name=function.__name__, function=function, description=description, schema=schema)


@dataclass(kw_only=True)
class ToolRegistry:
    """
    Registry for managing and invoking callable tools.

    This class allows registering, unregistering, and retrieving tools by name, as well as generating OpenAI-compatible schemas for all registered tools.
    """

    _tool_map: dict[str, Tool] = field(default_factory=dict)

    def register(self, function: Callable[..., Any], description: str | None = None) -> None:
        """
        Register a tool with the registry.

        Example:
            ```python
            registry = ToolRegistry()
            registry.register(add, description="Adds two numbers.")
            ```
        """
        self._tool_map[function.__name__] = Tool.from_function(function, description)

    def unregister(self, function: Callable[..., Any]) -> None:
        """
        Unregister a tool from the registry.
        """
        self._tool_map.pop(function.__name__)

    def schema(self) -> list[FunctionToolParam]:
        """
        Get the OpenAI-compatible schema for all registered tools.
        """
        return [tool.to_param() for tool in self._tool_map.values()]

    def __getitem__(self, tool_name: str) -> Tool:
        """
        Retrieve a registered tool by name.

        Raises:
            ValueError: If the tool is not registered.
        """
        if tool_name not in self._tool_map:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        return self._tool_map[tool_name]
