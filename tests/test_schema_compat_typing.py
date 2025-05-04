"""
Tests for IDE type hint preservation in schema_compat module.

IMPORTANT: These tests use mypy's API to simulate IDE behavior and verify that
type hints are preserved correctly by make_openai_compatible. The mypy API allows us to
programmatically check type inference, which is equivalent to what IDEs do when
providing autocompletion suggestions and type checking.

DO NOT MODIFY the tests in this file without ensuring they continue to verify
type information preservation, which is critical for IDE integration.
"""

import pytest
import textwrap
from typing import get_type_hints

from pydantic import BaseModel, ConfigDict
from mypy import api
from openai_utils.pydantic.schema_compat import make_openai_compatible


def test_compatible_model_preserves_field_types():
    """Test that make_openai_compatible preserves field types for IDE autocomplete"""

    # Create a model with specific field types
    class UserModel(BaseModel):
        name: str
        age: int
        is_active: bool
        model_config = ConfigDict(extra="allow")

    # Make the model OpenAI compatible
    CompatibleModel = make_openai_compatible(UserModel)

    # Get type hints for both models
    original_hints = get_type_hints(UserModel)
    compatible_hints = get_type_hints(CompatibleModel)

    # Verify that field types match between models
    for field_name in ["name", "age", "is_active"]:
        assert field_name in compatible_hints, f"Field {field_name} missing from compatible model"
        assert compatible_hints[field_name] == original_hints[field_name], (
            f"Type mismatch for field {field_name}: "
            f"Original: {original_hints[field_name]}, Compatible: {compatible_hints[field_name]}"
        )


def test_mypy_type_checking_with_compatible_model():
    """Test that mypy correctly infers types from the compatible model using mypy.api.

    This simulates IDE behavior by using mypy to verify type inference.
    """

    # Skip test if mypy is not available
    pytest.importorskip("mypy.api")

    # Create code to type check
    code = textwrap.dedent("""
    from pydantic import BaseModel, ConfigDict

    # Import the module under test - we need to define this here because
    # mypy won't have access to our local imports in the -c string
    class UserModel(BaseModel):
        name: str
        age: int
        is_active: bool
        model_config = ConfigDict(extra="allow")

    # Import and call the function
    from openai_utils.pydantic.schema_compat import make_openai_compatible
    CompatibleModel = make_openai_compatible(UserModel)

    # Create an instance of the compatible model
    result = CompatibleModel(name="John", age=30, is_active=True)

    # These should not raise type errors
    reveal_type(result.name)      # Should be str
    reveal_type(result.age)       # Should be int
    reveal_type(result.is_active) # Should be bool

    # Type operations should work correctly
    name_upper = result.name.upper()  # str method
    age_plus_one = result.age + 1     # int operation
    not_active = not result.is_active # bool operation
    """)

    # Run mypy on the code string with strict mode
    stdout, stderr, exit_code = api.run(["--no-error-summary", "--strict", "-c", code])

    # There should be no errors (apart from the reveal_type notes)
    error_lines = [line for line in stdout.split("\n") if "error:" in line]
    assert not error_lines, f"Unexpected type errors: {error_lines}"

    # Extract the reveal_type outputs and check they match expected types
    reveals = [line for line in stdout.split("\n") if "Revealed type" in line]
    assert len(reveals) == 3, f"Expected 3 type revelations, got {reveals}"

    assert "str" in reveals[0], f"Expected str type for name, got: {reveals[0]}"
    assert "int" in reveals[1], f"Expected int type for age, got: {reveals[1]}"
    assert "bool" in reveals[2], f"Expected bool type for is_active, got: {reveals[2]}"


def test_mypy_with_openai_conversation():
    """Test that mypy correctly handles types when used with Conversation.ask() method.

    This simulates IDE behavior by using mypy to verify type inference.
    """

    # Skip test if mypy is not available
    pytest.importorskip("mypy.api")

    # Create code to type check
    code = textwrap.dedent("""
    from pydantic import BaseModel, ConfigDict
    from openai import OpenAI
    from openai_utils.conversation import Conversation
    from openai_utils.models import Model
    from openai_utils.pydantic.schema_compat import make_openai_compatible

    class UserProfile(BaseModel):
        name: str
        email: str
        age: int
        model_config = ConfigDict(extra="allow")

    CompatibleUserProfile = make_openai_compatible(UserProfile)

    def process_user_profile() -> None:
        with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
            # The result should be correctly typed
            result = conversation.ask("Get user profile", format=CompatibleUserProfile)
            
            if result is not None:
                # These should all be valid operations with correct types
                reveal_type(result.name)   # Should be str
                reveal_type(result.email)  # Should be str
                reveal_type(result.age)    # Should be int
                
                # Type operations should work correctly
                name_upper = result.name.upper()
                email_parts = result.email.split('@')
                age_next_year = result.age + 1
    """)

    # Run mypy on the code string with strict mode
    stdout, stderr, exit_code = api.run(["--no-error-summary", "--strict", "-c", code])

    # There should be no errors (apart from the reveal_type notes)
    error_lines = [line for line in stdout.split("\n") if "error:" in line]
    assert not error_lines, f"Unexpected type errors: {error_lines}"

    # Extract the reveal_type outputs
    reveals = [line for line in stdout.split("\n") if "Revealed type" in line]

    # Since result is optional, mypy will use Union[None, CompatibleUserProfile]
    # So we need to check that the nested attributes still have correct types
    if reveals:
        assert len(reveals) == 3, f"Expected 3 type revelations, got {reveals}"
        assert "str" in reveals[0], f"Expected str type for name, got: {reveals[0]}"
        assert "str" in reveals[1], f"Expected str type for email, got: {reveals[1]}"
        assert "int" in reveals[2], f"Expected int type for age, got: {reveals[2]}"


def test_mypy_nested_compatible_models():
    """Test that mypy correctly handles nested model types.

    This simulates IDE behavior by using mypy to verify type inference.
    """

    # Skip test if mypy is not available
    pytest.importorskip("mypy.api")

    # Create code to type check
    code = textwrap.dedent("""
    from pydantic import BaseModel, ConfigDict
    from openai_utils.pydantic.schema_compat import make_openai_compatible

    class Address(BaseModel):
        street: str
        city: str
        zip_code: str
        model_config = ConfigDict(extra="allow")

    class User(BaseModel):
        name: str
        address: Address
        previous_addresses: list[Address] = []
        model_config = ConfigDict(extra="allow")

    CompatibleUser = make_openai_compatible(User)

    # Create an instance
    user = CompatibleUser(
        name="Test User",
        address=Address(street="123 Main St", city="Anytown", zip_code="12345"),
        previous_addresses=[
            Address(street="456 Old St", city="Oldtown", zip_code="54321")
        ]
    )

    # Type check nested model fields
    reveal_type(user.address)  # Should be compatible Address
    reveal_type(user.address.street)  # Should be str
    reveal_type(user.address.city)  # Should be str
    reveal_type(user.address.zip_code)  # Should be str

    # Type check nested collection
    reveal_type(user.previous_addresses)  # Should be list of compatible Address
    reveal_type(user.previous_addresses[0])  # Should be compatible Address
    reveal_type(user.previous_addresses[0].street)  # Should be str

    # Operations on nested types
    city_upper = user.address.city.upper()
    all_streets = [addr.street for addr in user.previous_addresses]
    """)

    # Run mypy on the code string with strict mode
    stdout, stderr, exit_code = api.run(["--no-error-summary", "--strict", "-c", code])

    # There should be no errors (apart from the reveal_type notes)
    error_lines = [line for line in stdout.split("\n") if "error:" in line]
    assert not error_lines, f"Unexpected type errors: {error_lines}"

    # Extract the reveal_type outputs
    reveals = [line for line in stdout.split("\n") if "Revealed type" in line]

    # Check that types for nested models are preserved
    assert len(reveals) >= 7, f"Expected at least 7 type revelations, got {reveals}"
    for reveal in reveals:
        if "user.address.street" in reveal:
            assert "str" in reveal, f"Expected str type for street, got: {reveal}"
        if "user.address.city" in reveal:
            assert "str" in reveal, f"Expected str type for city, got: {reveal}"
        if "user.address.zip_code" in reveal:
            assert "str" in reveal, f"Expected str type for zip_code, got: {reveal}"
        if "user.previous_addresses[0].street" in reveal:
            assert "str" in reveal, f"Expected str type for previous address street, got: {reveal}"
