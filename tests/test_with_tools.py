import httpx
import pytest
from openai import BaseModel, OpenAI
from pydantic import ConfigDict

from openai_utils.conversation import Conversation
from openai_utils.models import Model
from openai_utils.tool_registry import ToolRegistry


@pytest.mark.vcr
def test_with_tools():
    def get_weather(latitude: float, longitude: float) -> float:
        response = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    conversation = Conversation(client=OpenAI(), model=Model.GPT_4O_MINI)
    conversation.tools.register(get_weather, "Get the weather for a given latitude and longitude")

    with conversation:
        result = conversation.ask("What's the weather like in Paris today?")
        assert isinstance(result, str)
        assert (
            result
            == "The weather in Paris today is approximately 28.2°C. Would you like to know more details, such as humidity or wind speed?"
        )


@pytest.mark.vcr
def test_with_tools_split_registry():
    def get_weather(latitude: float, longitude: float) -> float:
        response = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    registry = ToolRegistry()
    registry.register(get_weather, "Get the weather for a given latitude and longitude")

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI, tools=registry) as conversation:
        result = conversation.ask("What's the weather like in Paris today?")
        assert isinstance(result, str)
        assert (
            result
            == "The weather in Paris today is around 28.2°C. If you need more details like humidity or precipitation, just let me know!"
        )


@pytest.mark.vcr
def test_with_tools_and_structured_output():
    def get_weather(latitude: float, longitude: float) -> float:
        response = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    # Setup the conversation and register the tool
    conversation = Conversation(client=OpenAI(), model=Model.GPT_4O_MINI)
    conversation.tools.register(get_weather, "Get the weather for a given latitude and longitude")

    # Define the structured output
    class Weather(BaseModel):
        temperature: float

        model_config = ConfigDict(extra="forbid")

    # Run the conversation
    with conversation:
        result = conversation.ask("What's the weather like in Paris today?", format=Weather)
        assert isinstance(result, Weather)
        assert result.temperature == 28.2


def test_register_tool_without_argument_type_hints():
    def get_weather(latitude, longitude):
        response = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    conversation = Conversation(client=OpenAI(), model=Model.GPT_4O_MINI)
    with pytest.raises(TypeError):
        conversation.tools.register(get_weather, "Get the weather for a given latitude and longitude")


def test_register_tool_without_return_type_hints():
    def get_weather(latitude: float, longitude: float):
        response = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    conversation = Conversation(client=OpenAI(), model=Model.GPT_4O_MINI)

    # This should not raise an error
    conversation.tools.register(get_weather, "Get the weather for a given latitude and longitude")
