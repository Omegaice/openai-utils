from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, kw_only=True)
class ModelCost:
    input: float
    cached: float
    output: float

    def __call__(self, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000) * self.input
            + (cached_tokens / 1_000_000) * self.cached
            + (output_tokens / 1_000_000) * self.output
        )


class Model(StrEnum):
    cost: ModelCost

    """
    Each member's .value is the string to send to OpenAI,
    and each member also gets a `.cost: ModelCost` attribute.
    """

    def __new__(cls, value: str, input_cost: float, cached_cost: float, output_cost: float):
        obj = str.__new__(cls, value)
        obj._value_ = value
        object.__setattr__(obj, "cost", ModelCost(input=input_cost, cached=cached_cost, output=output_cost))
        return obj

    # GPT-4.1 Series
    GPT_4_1 = ("gpt-4.1", 2.00, 0.50, 8.00)
    GPT_4_1_MINI = ("gpt-4.1-mini", 0.40, 0.10, 1.60)
    GPT_4_1_NANO = ("gpt-4.1-nano", 0.10, 0.025, 0.40)

    # GPT-4o Series
    GPT_4O = ("gpt-4o", 2.50, 1.25, 10.00)
    GPT_4O_MINI = ("gpt-4o-mini", 0.15, 0.075, 0.60)

    # o-Series Reasoning Models
    O3 = ("o3", 10.00, 2.50, 40.00)
    O3_MINI = ("o3-mini", 1.10, 0.55, 4.40)
    O3_MINI_HIGH = ("o3-mini-high", 1.10, 0.55, 4.40)
    O4_MINI = ("o4-mini", 1.10, 0.275, 4.40)
    O4_MINI_HIGH = ("o4-mini-high", 1.10, 0.275, 4.40)

    # GPT-4.5 (Deprecated)
    GPT_4_5 = ("gpt-4.5", 75.00, 37.50, 150.00)

    # GPT-3.5 (Legacy)
    GPT_3_5 = ("gpt-3.5", 0.50, 0.125, 1.50)
