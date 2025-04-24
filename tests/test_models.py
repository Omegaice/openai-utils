import pytest
from openai_utils.models import Model, ModelCost

# List of (enum member, expected value, input, cached, output)
MODEL_TEST_DATA = [
    (Model.GPT_4_1, "gpt-4.1", 2.00, 0.50, 8.00),
    (Model.GPT_4_1_MINI, "gpt-4.1-mini", 0.40, 0.10, 1.60),
    (Model.GPT_4_1_NANO, "gpt-4.1-nano", 0.10, 0.025, 0.40),
    (Model.GPT_4O, "gpt-4o", 2.50, 1.25, 10.00),
    (Model.GPT_4O_MINI, "gpt-4o-mini", 0.15, 0.075, 0.60),
    (Model.O3, "o3", 10.00, 2.50, 40.00),
    (Model.O3_MINI, "o3-mini", 1.10, 0.55, 4.40),
    (Model.O3_MINI_HIGH, "o3-mini-high", 1.10, 0.55, 4.40),
    (Model.O4_MINI, "o4-mini", 1.10, 0.275, 4.40),
    (Model.O4_MINI_HIGH, "o4-mini-high", 1.10, 0.275, 4.40),
    (Model.GPT_4_5, "gpt-4.5", 75.00, 37.50, 150.00),
    (Model.GPT_3_5, "gpt-3.5", 0.50, 0.125, 1.50),
]


@pytest.mark.parametrize("model, expected_value, input_cost, cached_cost, output_cost", MODEL_TEST_DATA)
def test_model_enum_values(model, expected_value, input_cost, cached_cost, output_cost):
    # Check .value
    assert model.value == expected_value
    # Check .cost is a ModelCost
    assert isinstance(model.cost, ModelCost)
    # Check .cost fields
    assert model.cost.input == input_cost
    assert model.cost.cached == cached_cost
    assert model.cost.output == output_cost


@pytest.mark.parametrize("model, _, input_cost, cached_cost, output_cost", MODEL_TEST_DATA)
def test_model_cost_call(model, _, input_cost, cached_cost, output_cost):
    # Use 1,000,000 tokens for each to make the cost equal to the rate
    tokens = 1_000_000
    expected = input_cost + cached_cost + output_cost
    result = model.cost(tokens, tokens, tokens)
    assert pytest.approx(result, rel=1e-6) == expected
