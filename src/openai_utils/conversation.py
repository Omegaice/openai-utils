import atexit
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self, cast

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.responses import (
    ParsedResponseOutputMessage,
    ParsedResponseOutputText,
    Response,
    ResponseFunctionToolCall,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_input_param import ResponseInputItemParam
from typing_extensions import TypeVar, overload

from openai_utils.models import Model
from openai_utils.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class ConversationManager:
    def __init__(self, client: OpenAI | None = None, model: Model = Model.GPT_4_1_MINI, log_on_exit: bool = True):
        """Initialize the conversation manager"""

        self.client = client if client is not None else OpenAI()
        self.model = model
        self.total_cost = 0.0

        if log_on_exit:
            self.register_exit_handler()

    def register_exit_handler(self) -> None:
        def log_on_exit() -> None:
            logger.info(f"Conversation total cost: ${self.total_cost:.6f}")

        atexit.register(log_on_exit)

    def new_conversation(self, instructions: str | NotGiven = NOT_GIVEN) -> "Conversation":
        """Create a new conversation"""

        return Conversation(model=self.model, client=self.client, manager=self, instructions=instructions)

    def add_cost(self, cost: float) -> None:
        self.total_cost += cost


T = TypeVar("T")


@dataclass(kw_only=True)
class Conversation(AbstractContextManager["Conversation"]):
    client: OpenAI
    model: Model
    instructions: str | NotGiven = NOT_GIVEN
    manager: ConversationManager | None = None
    tools: ToolRegistry = field(default_factory=ToolRegistry)

    # State Data
    previous_response_id: str | NotGiven = field(init=False, default=NOT_GIVEN)
    total_cost: float = field(init=False, default=0.0)

    # Usage
    input_tokens: int = field(init=False, default=0)
    cached_tokens: int = field(init=False, default=0)
    output_tokens: int = field(init=False, default=0)

    @overload
    def ask(self, input_text: str, format: type[T]) -> T | None: ...

    @overload
    def ask(self, input_text: str, format: NotGiven = NOT_GIVEN) -> str | None: ...

    def ask(self, input_text: str, format: type[T] | NotGiven = NOT_GIVEN) -> T | None:
        """Send a new message to the conversation, supporting multi-turn tool-calling (no structured formatting)."""
        input_message: ResponseInputItemParam = EasyInputMessageParam(role="user", content=input_text)

        while True:
            response = self.client.responses.parse(
                model=self.model.value,
                tools=self.tools.schema(),
                text_format=format,
                input=[input_message],
                previous_response_id=self.previous_response_id,
            )

            # Save the response ID for the next request
            self.previous_response_id = response.id

            # Update the usage
            self._update_usage(response)

            # Early return if there is no output
            if response.output is None:
                raise ValueError("No output from the model")

            # Get the latest output
            last_output = response.output[-1]

            # Handle tool calls
            if isinstance(last_output, ResponseFunctionToolCall):
                input_message = self.tools[last_output.name](last_output.call_id, last_output.arguments)

            # Handle result
            if isinstance(last_output, ParsedResponseOutputMessage):
                content = last_output.content[0]
                if not isinstance(content, ParsedResponseOutputText):
                    raise ValueError(f"Unexpected content type: {type(content)}")
                if format is not NOT_GIVEN:
                    return cast(T, content.parsed)
                return cast(T, content.text)

    def _update_usage(self, response: Response) -> None:
        """Update the token counts and total cost for the message."""

        if response.usage is None:
            return

        # Update token counts
        self.input_tokens += response.usage.input_tokens
        self.cached_tokens += response.usage.input_tokens_details.cached_tokens
        self.output_tokens += response.usage.output_tokens

        # Update total cost
        cost = self.model.cost(
            input_tokens=self.input_tokens, cached_tokens=self.cached_tokens, output_tokens=self.output_tokens
        )
        self.total_cost += cost

        # Log the cost for the message
        logger.debug(
            f"Conversation message cost=${cost:.6f} input={self.input_tokens} cached={self.cached_tokens} output={self.output_tokens}"
        )

    # Context Manager Methods
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        """When we exit the context manager, we add the total cost to the manager if there is one and log the total cost."""
        if self.manager is not None:
            self.manager.add_cost(self.total_cost)

        # Log the total cost for the conversation
        logger.info(f"Conversation total cost: ${self.total_cost:.6f}")
