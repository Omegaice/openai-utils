from openai import OpenAI
from pydantic import BaseModel

from openai_utils.conversation import Conversation
from openai_utils.models import Model


def test_structured_output():
    class User(BaseModel):
        name: str

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        result = conversation.ask("What is your name?", format=User)
        assert isinstance(result, User)

        result = conversation.ask("What is your name?")
        assert isinstance(result, str)
