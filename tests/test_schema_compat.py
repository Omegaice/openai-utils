import pytest
from pydantic import BaseModel, ConfigDict, RootModel
from openai import OpenAI
from openai_utils.conversation import Conversation
from openai_utils.models import Model
from openai_utils.pydantic.schema_compat import make_openai_compatible


@pytest.mark.vcr
def test_make_openai_compatible_with_additional_properties():
    """Test that make_openai_compatible fixes additionalProperties requirement"""

    # Create a model with additionalProperties set to True (extra="allow")
    class UserWithAdditionalProperties(BaseModel):
        name: str
        model_config = ConfigDict(extra="allow")

    # Make the model OpenAI compatible
    CompatibleModel = make_openai_compatible(UserWithAdditionalProperties)

    # Verify the compatible model has additionalProperties: false
    compat_schema = CompatibleModel.model_json_schema()
    assert compat_schema.get("additionalProperties") is False, (
        "Compatible schema should have additionalProperties set to False"
    )

    # Test that it works with OpenAI
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        result = conversation.ask("What is your name?", format=CompatibleModel)
        assert result is not None

        # Convert result to original model type using model_dump
        original_model = UserWithAdditionalProperties(**result.model_dump())

        # Verify we got a valid response
        assert isinstance(result, CompatibleModel), "Result should be an instance of the compatible model type"
        assert isinstance(original_model, UserWithAdditionalProperties), (
            "Should be able to create original model from result"
        )
        assert isinstance(original_model.name, str), "Name should be a string"
        assert len(original_model.name) > 0, "Name should not be empty"


def test_make_openai_compatible_with_nested_additional_properties():
    """Test that make_openai_compatible fixes additionalProperties in nested models"""

    # Create a nested model with additionalProperties set to True
    class NestedModel(BaseModel):
        value: str
        model_config = ConfigDict(extra="allow")

    # Create a parent model that contains the nested model
    class ParentModel(BaseModel):
        name: str
        nested: NestedModel
        model_config = ConfigDict(extra="forbid")

    # Make the parent model OpenAI compatible
    CompatibleModel = make_openai_compatible(ParentModel)

    # Get the schema to check
    schema = CompatibleModel.model_json_schema()

    # Verify parent model has additionalProperties: false
    assert schema.get("additionalProperties") is False, "Parent schema should have additionalProperties set to False"

    # Verify nested model has additionalProperties: false
    nested_ref = schema["properties"]["nested"].get("$ref")
    if nested_ref:
        # Get the definition name from the reference
        def_name = nested_ref.split("/")[-1]
        nested_schema = schema["$defs"][def_name]
        assert nested_schema.get("additionalProperties") is False, (
            "Nested schema should have additionalProperties set to False"
        )
    else:
        # If there's no reference, the nested model should be inlined
        nested_schema = schema["properties"]["nested"]
        assert nested_schema.get("additionalProperties") is False, (
            "Nested schema should have additionalProperties set to False"
        )


def test_make_openai_compatible_rejects_root_union():
    """Test that make_openai_compatible rejects RootModel with Union types"""

    class DogType(BaseModel):
        pet_type: str = "dog"
        breed: str

    class CatType(BaseModel):
        pet_type: str = "cat"
        lives_left: int

    # Use RootModel with Union type
    class Pet(RootModel):
        root: DogType | CatType

    # Trying to make this compatible should raise a ValueError
    with pytest.raises(ValueError) as exc_info:
        make_openai_compatible(Pet)

    # Verify the error message mentions Union types
    error_msg = str(exc_info.value)
    assert "Union" in error_msg or "anyOf" in error_msg, "Error should mention Union or anyOf restriction"
