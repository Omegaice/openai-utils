import httpx
import pytest
from openai import OpenAI

from openai_utils.conversation import Conversation
from openai_utils.models import Model


@pytest.mark.vcr
def test_with_tools():
    def get_weather(latitude, longitude):
        response = httpx.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        data = response.json()
        return data["current"]["temperature_2m"]

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        conversation.register_tool(get_weather)

        result = conversation.ask("What's the weather like in Paris today?")
        assert isinstance(result, str)
        assert result.startswith("The weather in Paris today is")