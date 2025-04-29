<div align="center">

# openai-utils

[<img alt="github" src="https://img.shields.io/badge/github-Omegaice/openai--utils-8da0cb?style=for-the-badge&logo=github" height="20">](https://github.com/Omegaice/openai-utils)
[<img alt="license" src="https://img.shields.io/github/license/Omegaice/openai-utils?style=for-the-badge&color=green" height="20">](https://github.com/Omegaice/openai-utils/blob/master/LICENSE)

</div>

A Python utility library for managing OpenAI API conversations, tracking usage costs, and working with structured outputs. Provides convenient abstractions for conversation management, model selection, and cost calculation.

## Features

- 💬 **Conversation Management**: Easily manage multi-turn conversations with the OpenAI API
- 💸 **Cost Tracking**: Track token usage and estimate costs for each conversation
- 🏷️ **Model Utilities**: Enum-based model selection with built-in cost rates
- 🧩 **Structured Output**: Support for requesting and parsing structured responses (e.g., Pydantic models)
- 🧪 **Testable**: Designed for easy testing and extension

## Installation

Clone this repository and install dependencies using [UV](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/Omegaice/openai-utils.git
cd openai-utils
uv sync
```

Alternatively, add this repository directly to your own project's dependencies with UV:

```bash
uv add "openai-utils @ git+https://github.com/Omegaice/openai-utils.git@master"
```

## Quick Example

```python
from openai import OpenAI
from pydantic import BaseModel
from openai_utils import Conversation, ConversationManager, Model

# Setup manager with model and client
manager = ConversationManager(client=OpenAI(), model=Model.GPT_4O_MINI, log_on_exit=True)

# Standard conversation
with manager.new_conversation() as conversation:
    result = conversation.ask("What is your name?")
    print(result)  # str


# Structured response conversation
class User(BaseModel):
    name: str

with manager.new_conversation() as conversation:
    result = conversation.ask("What is your name?", format=User)
    print(result)  # User(name=...)
```

## Requirements

- Python 3.11+
- openai >= 1.75.0
- [See `pyproject.toml` for full dependency list]

## License

MIT