import atexit
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Self, cast
from typing_extensions import TypeVar, overload

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.responses import Response

from openai_utils.models import Model

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
            print("Logging on exit")
            logger.info(f"Conversation total cost: ${self.total_cost:.6f}")

        atexit.register(log_on_exit)

    def new_conversation(self, instructions: str | NotGiven = NOT_GIVEN) -> "Conversation":
        """Create a new conversation"""

        return Conversation(model=self.model, client=self.client, manager=self, instructions=instructions)

    def add_cost(self, cost: float) -> None:
        self.total_cost += cost


T = TypeVar("T", default=str)


@dataclass(kw_only=True)
class Conversation(AbstractContextManager):
    client: OpenAI
    model: Model
    instructions: str | NotGiven = NOT_GIVEN
    manager: ConversationManager | None = None

    # State Data
    previous_response_id: str | NotGiven = field(init=False, default=NOT_GIVEN)
    total_cost: float = field(init=False, default=0.0)

    # Usage
    input_tokens: int = field(init=False, default=0)
    cached_tokens: int = field(init=False, default=0)
    output_tokens: int = field(init=False, default=0)

    @overload
    def ask(self, input_text: str, format: None = None) -> str | None: ...

    @overload
    def ask(self, input_text: str, format: type[T]) -> T | None: ...

    def ask(self, input_text: str, format: type[T] | None = None) -> T | None:
        """Send a new message to the conversation"""

        # If we have a previous response, we don't need to provide instructions
        with_instructions = self.instructions if self.previous_response_id is NOT_GIVEN else NOT_GIVEN

        if format is None:
            return cast(T, self._ask_without_format(input_text, with_instructions))
        else:
            return self._ask_with_format(input_text, format, with_instructions)

    def _ask_without_format(self, input_text: str, with_instructions: str | NotGiven = NOT_GIVEN) -> str:
        """Send a new message to the conversation without a format"""
        # Send the request to OpenAI
        response = self.client.responses.create(
            model=self.model,
            input=input_text,
            instructions=with_instructions,
            previous_response_id=self.previous_response_id,
        )

        # Update token counts
        self._update_usage(response)

        # Update the previous response ID
        self.previous_response_id = response.id

        return response.output_text

    def _ask_with_format(
        self, input_text: str, format: type[T], with_instructions: str | NotGiven = NOT_GIVEN
    ) -> T | None:
        """Send a new message to the conversation with a format"""

        # Send the request to OpenAI
        response = self.client.responses.parse(
            model=self.model,
            input=input_text,
            instructions=with_instructions,
            previous_response_id=self.previous_response_id,
            text_format=format,
        )

        # Update token counts
        self._update_usage(response)

        # Update the previous response ID
        self.previous_response_id = response.id

        return response.output_parsed

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

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """When we exit the context manager, we add the total cost to the manager if there is one and log the total cost."""
        if self.manager is not None:
            self.manager.add_cost(self.total_cost)

        # Log the total cost for the conversation
        logger.info(f"Conversation total cost: ${self.total_cost:.6f}")
