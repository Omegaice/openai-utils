from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, RootModel
import pytest
from openai_utils.conversation import Conversation
from openai_utils.models import Model


@pytest.mark.vcr
def test_structured_output_anyof_fails():
    """
    Test that using anyOf at the root level fails with OpenAI structured output.

    Reference: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#root-objects-must-not-be-anyof
    """

    class DogType(BaseModel):
        pet_type: str = "dog"
        breed: str

    class CatType(BaseModel):
        pet_type: str = "cat"
        lives_left: int

    # Use RootModel instead of __root__ in Pydantic v2
    class Pet(RootModel):
        root: DogType | CatType

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError since OpenAI doesn't support anyOf at the root level
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("What type of pet do you have?", format=Pet)

        error_msg = str(exc_info.value)
        assert any(
            msg in error_msg
            for msg in ["schema must be a JSON Schema of 'type: \"object\"'", "Invalid schema for response_format"]
        )


@pytest.mark.vcr
def test_structured_output_optional_field_supported():
    """Test that optional fields are actually supported by OpenAI structured output API.

    Reference:
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#optional-fields

    Note: This test demonstrates the actual API behavior which differs from documentation.
    OpenAI documentation states all fields must be marked as required, but in reality
    the API accepts standard Pydantic models with optional fields.
    """

    class UserWithOptional(BaseModel):
        name: str  # Required field
        email: str  # Another required field
        age: int | None = None  # Optional field with None default
        model_config = ConfigDict(extra="forbid")

    # Get the standard Pydantic schema
    schema = UserWithOptional.model_json_schema()

    # Get required fields from schema
    required_fields = schema.get("required", [])

    # Verify that Pydantic doesn't include optional fields in required list by default
    assert "name" in required_fields, "Required field 'name' should be in required list"
    assert "email" in required_fields, "Required field 'email' should be in required list"
    assert "age" not in required_fields, "Optional field 'age' should not be in required list"

    # Test with standard Pydantic model that has an optional field
    # Despite documentation stating all fields must be in required list,
    # the API actually accepts optional fields not in the required list
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        result = conversation.ask("Tell me about yourself", format=UserWithOptional)
        assert isinstance(result, UserWithOptional)
        assert isinstance(result.name, str)
        assert isinstance(result.email, str)
        assert result.age is None or isinstance(result.age, int)


@pytest.mark.vcr
def test_structured_output_missing_additional_properties():
    """
    Test that missing additionalProperties: false fails with OpenAI structured output.

    Reference: https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#additionalproperties-false-must-always-be-set-in-objects
    """

    class UserWithoutAdditionalProperties(BaseModel):
        name: str

        # Allow extra fields
        model_config = ConfigDict(extra="allow")

    # Get the schema
    schema = UserWithoutAdditionalProperties.model_json_schema()
    assert "additionalProperties" in schema, "additionalProperties should be in the schema"
    assert schema["additionalProperties"] is True, "additionalProperties should be True"

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("What is your name?", format=UserWithoutAdditionalProperties)

        error_msg = str(exc_info.value)
        assert any(
            msg in error_msg
            for msg in ["Invalid schema", "`additionalProperties: false` must always be set in objects"]
        )


@pytest.mark.vcr
def test_structured_output_unsupported_string_keywords():
    """
    Test that using unsupported string keywords fails with OpenAI structured output.

    Reference:
    * For strings: minLength, maxLength, pattern, format
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#some-type-specific-keywords-are-not-yet-supported
    """

    class UserWithStringConstraints(BaseModel):
        name: str = Field(min_length=3, max_length=50)
        email: str = Field(pattern=r"[^@]+@[^@]+\.[^@]+")

    # Get the schema to verify it contains the unsupported keywords
    schema = UserWithStringConstraints.model_json_schema()
    assert "minLength" in schema["properties"]["name"] or "maxLength" in schema["properties"]["name"]
    assert "pattern" in schema["properties"]["email"]

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Tell me about yourself", format=UserWithStringConstraints)

        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["minLength", "maxLength", "pattern", "Invalid schema"])


@pytest.mark.vcr
def test_structured_output_unsupported_number_keywords():
    """
    Test that using unsupported number keywords fails with OpenAI structured output.

    Reference:
    * For numbers: minimum, maximum, multipleOf
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#some-type-specific-keywords-are-not-yet-supported
    """

    class UserWithNumberConstraints(BaseModel):
        age: int = Field(gt=0, lt=150)
        score: float = Field(ge=0, le=100, multiple_of=0.5)

    # Get the schema to verify it contains the unsupported keywords
    schema = UserWithNumberConstraints.model_json_schema()
    # Check for gt/lt which translate to minimum/maximum
    assert any(
        key in schema["properties"]["age"] for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
    )
    assert any(key in schema["properties"]["score"] for key in ["multipleOf", "minimum", "maximum"])

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Tell me about yourself", format=UserWithNumberConstraints)

        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["minimum", "maximum", "multipleOf", "Invalid schema"])


@pytest.mark.vcr
def test_structured_output_unsupported_object_keywords():
    """
    Test that using unsupported object keywords fails with OpenAI structured output.

    Reference:
    * For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#some-type-specific-keywords-are-not-yet-supported
    """

    class ObjectWithConstraints(BaseModel):
        model_config = ConfigDict(
            json_schema_extra={
                "minProperties": 2,
                "maxProperties": 5,
                "propertyNames": {"pattern": "^[a-z]+$"},
            }
        )
        name: str
        age: int

    # Get the schema to verify it contains the unsupported keywords
    schema = ObjectWithConstraints.model_json_schema()
    # Verify schema has the unsupported object keywords
    assert "minProperties" in schema or "maxProperties" in schema or "propertyNames" in schema

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Tell me about yourself", format=ObjectWithConstraints)

        error_msg = str(exc_info.value)
        assert any(
            keyword in error_msg for keyword in ["minProperties", "maxProperties", "propertyNames", "Invalid schema"]
        )


@pytest.mark.vcr
def test_structured_output_unsupported_array_keywords():
    """
    Test that using unsupported array keywords fails with OpenAI structured output.

    Reference:
    * For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#some-type-specific-keywords-are-not-yet-supported
    """

    class UserWithArrayConstraints(BaseModel):
        model_config = ConfigDict(
            json_schema_extra={
                "properties": {
                    "tags": {
                        "minItems": 1,
                        "maxItems": 5,
                        "uniqueItems": True,
                    }
                }
            }
        )
        name: str
        tags: list[str]

    # Get the schema to verify it contains the unsupported keywords
    schema = UserWithArrayConstraints.model_json_schema()
    # Verify schema has at least one of the unsupported array keywords in properties.tags
    tags_schema = schema.get("properties", {}).get("tags", {})
    if "minItems" in tags_schema or "maxItems" in tags_schema or "uniqueItems" in tags_schema:
        pass  # Good, found at least one of the unsupported keywords
    else:
        # Look in json_schema_extra if not found directly
        extra = schema.get("properties", {}).get("tags", {})
        assert any(key in str(extra) for key in ["minItems", "maxItems", "uniqueItems"])

    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Tell me about yourself", format=UserWithArrayConstraints)

        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["minItems", "maxItems", "uniqueItems", "Invalid schema"])


@pytest.mark.vcr
def test_structured_output_deep_nesting():
    """
    Test that excessive deep nesting is rejected by OpenAI structured output.

    Reference:
    * A schema may have up to 100 object properties total, with up to 5 levels of nesting.
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#objects-have-limitations-on-nesting-depth-and-size
    """

    # Create a model with manually crafted deep nesting to ensure we exceed the limit
    class DeepNestedModel(BaseModel):
        top_level: str
        model_config = ConfigDict(extra="forbid")

        @classmethod
        def model_json_schema(cls, **kwargs):
            schema = super().model_json_schema(**kwargs)

            # Create a deeply nested schema with more than 5 levels
            # Level 1
            schema["properties"]["nested"] = {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "level1_field": {"type": "string"},
                    # Level 2
                    "level2": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "level2_field": {"type": "string"},
                            # Level 3
                            "level3": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "level3_field": {"type": "string"},
                                    # Level 4
                                    "level4": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "properties": {
                                            "level4_field": {"type": "string"},
                                            # Level 5
                                            "level5": {
                                                "type": "object",
                                                "additionalProperties": False,
                                                "properties": {
                                                    "level5_field": {"type": "string"},
                                                    # Level 6 (exceeds the 5 level limit)
                                                    "level6": {
                                                        "type": "object",
                                                        "additionalProperties": False,
                                                        "properties": {"level6_field": {"type": "string"}},
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }

            # Make sure nested is required
            if "required" not in schema:
                schema["required"] = []
            if "top_level" not in schema["required"]:
                schema["required"].append("top_level")
            if "nested" not in schema["required"]:
                schema["required"].append("nested")

            return schema

    # Get schema and verify it has deep nesting structure
    schema = DeepNestedModel.model_json_schema()
    # Verify that our schema has deep nesting (at least to level 6)
    assert "nested" in schema.get("properties", {})
    level1 = schema.get("properties", {}).get("nested", {}).get("properties", {})
    assert "level2" in level1, "Schema should have at least level 2 nesting"
    level2 = level1.get("level2", {}).get("properties", {})
    assert "level3" in level2, "Schema should have at least level 3 nesting"
    level3 = level2.get("level3", {}).get("properties", {})
    assert "level4" in level3, "Schema should have at least level 4 nesting"
    level4 = level3.get("level4", {}).get("properties", {})
    assert "level5" in level4, "Schema should have at least level 5 nesting"
    level5 = level4.get("level5", {}).get("properties", {})
    assert "level6" in level5, "Schema should have at least level 6 nesting"

    # Test with model that exceeds the 5 level nesting limit - should fail
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Give me deeply nested data", format=DeepNestedModel)

        # Verify the error message
        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["nesting", "depth", "level", "limit", "5", "Invalid schema"])


@pytest.mark.vcr
def test_structured_output_property_limit():
    """
    Test the 100 object properties limit.

    Reference:
    * A schema may have up to 100 object properties total
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#objects-have-limitations-on-nesting-depth-and-size
    """
    # Using a correctly annotated approach to create a model with many properties
    properties = {}
    annotations = {}

    # Create 101 fields to exceed the 100 properties limit
    for i in range(1, 102):
        field_name = f"field_{i}"
        annotations[field_name] = str  # Add type annotation
        properties[field_name] = Field()  # Use Field without defaults

    # Create model class attributes
    attrs = {"__annotations__": annotations, "model_config": ConfigDict(extra="forbid"), **properties}

    # Dynamically create the model class with too many properties
    TooManyPropsModel = type("TooManyPropsModel", (BaseModel,), attrs)

    # Get the schema to verify the number of properties
    schema = TooManyPropsModel.model_json_schema()
    property_count = len(schema.get("properties", {}))
    assert property_count > 100, f"Test model should have more than 100 properties, found {property_count}"

    # Test with model that exceeds the 100 properties limit - should fail
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        # This will fail with a BadRequestError
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Give me data for this model", format=TooManyPropsModel)

        # Verify the error message
        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["100", "properties", "limit", "exceed", "Invalid schema"])


@pytest.mark.vcr
def test_structured_output_recursive_schema():
    """
    Test that recursive schemas are supported with proper configuration.

    Reference:
    * Recursive schemas are supported with proper configuration
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#supported-features
    """

    # Create a model with recursion (a tree node that can have children of the same type)
    class TreeNode(BaseModel):
        name: str
        # Important: Don't use field with default for recursive fields
        # as OpenAI doesn't support default in recursive schemas
        children: list["TreeNode"] = Field(default_factory=list)

        # Set additionalProperties to false explicitly as required by OpenAI
        model_config = ConfigDict(extra="forbid")

    # Ensure the recursion is properly resolved
    TreeNode.model_rebuild()

    # Get the schema to verify it contains the recursion reference
    schema = TreeNode.model_json_schema()
    assert "$defs" in schema, "Schema should contain $defs for recursion"

    # Test with properly configured recursive model
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        result = conversation.ask("Create a simple tree with one parent and two children", format=TreeNode)

        # Verify result structure
        assert isinstance(result, TreeNode)
        assert isinstance(result.name, str)
        assert isinstance(result.children, list)

        # If children are returned, verify they are TreeNode instances
        if result.children:
            for child in result.children:
                assert isinstance(child, TreeNode)
                assert isinstance(child.name, str)


@pytest.mark.vcr
def test_structured_output_recursive_schema_with_defaults():
    """
    Test that recursive schemas with defaults are not supported.

    Reference:
    * Recursive schemas are supported, but with limitations
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#supported-features
    """

    # Create a model with recursion that uses defaults
    class InvalidTreeNode(BaseModel):
        name: str
        # Using direct default assignment instead of default_factory causes issues
        children: list["InvalidTreeNode"] = []

        # Set additionalProperties to false explicitly as required by OpenAI
        model_config = ConfigDict(extra="forbid")

    # Ensure the recursion is properly resolved
    InvalidTreeNode.model_rebuild()

    # Get the schema to verify it contains the recursion reference
    schema = InvalidTreeNode.model_json_schema()
    assert "$defs" in schema, "Schema should contain $defs for recursion"

    # Test with improperly configured recursive model - should fail
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Create a simple tree with one parent and two children", format=InvalidTreeNode)

        # Verify the error message
        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["default", "not permitted", "Invalid schema"])


# The following tests are for restrictions mentioned in the OpenAI documentation
# but not currently enforced by the API. They are skipped for now but kept in the
# codebase to document the behavior and to be ready if API behavior changes in the future.


@pytest.mark.skip(reason="OpenAI docs state this restriction but API accepts optional fields")
@pytest.mark.vcr
def test_structured_output_optional_field():
    """Test that optional fields work with OpenAI structured output despite documentation.

    Reference:
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#optional-fields
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#all-fields-or-function-parameters-must-be-specified-as-required

    This test checks:
    1. How standard Pydantic schema represents optional fields (union with null type)
    2. That Pydantic excludes optional fields from the 'required' list by default
    3. Whether OpenAI accepts optional fields not listed in the 'required' list

    Note: OpenAI documentation states all fields must be marked as required, but
    their implementation appears to accept standard Pydantic models with optional fields.
    """

    class UserWithOptional(BaseModel):
        name: str  # Required field
        email: str  # Another required field
        age: int | None = None  # Optional field with None default
        model_config = ConfigDict(extra="forbid")

    # Get the standard Pydantic schema
    schema = UserWithOptional.model_json_schema()

    # Get required fields from schema
    required_fields = schema.get("required", [])

    # Verify that Pydantic doesn't include optional fields in required list by default
    assert "name" in required_fields, "Required field 'name' should be in required list"
    assert "email" in required_fields, "Required field 'email' should be in required list"
    assert "age" not in required_fields, "Optional field 'age' should not be in required list"

    # Examine the optional field schema
    age_property = schema["properties"]["age"]

    # Verify the schema correctly represents optional fields as a union with null
    assert "anyOf" in age_property or "type" in age_property, "Schema should have type info for age"

    # Verify correct null representation
    if "anyOf" in age_property:
        type_options = [t.get("type") for t in age_property["anyOf"] if "type" in t]
        assert "integer" in type_options, "Schema should allow integer type for age"
        assert "null" in type_options, "Schema should allow null type for age"
    elif isinstance(age_property.get("type"), list):
        assert "integer" in age_property["type"], "Schema should allow integer type for age"
        assert "null" in age_property["type"], "Schema should allow null type for age"

    # Test with standard Pydantic model
    # Despite documentation stating all fields must be in required list,
    # OpenAI appears to accept standard Pydantic models with optional fields
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        result = conversation.ask("Tell me about yourself", format=UserWithOptional)
        assert isinstance(result, UserWithOptional)
        assert isinstance(result.name, str)
        assert isinstance(result.email, str)
        assert result.age is None or isinstance(result.age, int)


@pytest.mark.skip(reason="OpenAI docs state this restriction but API accepts schema with >500 enum values")
@pytest.mark.vcr
def test_structured_output_enum_limits():
    """
    Test enum size limitations.

    Reference:
    * A schema may have up to 500 enum values across all enum properties.
    * For a single enum property with string values, the total string length of all
      enum values cannot exceed 7,500 characters when there are more than 250 enum values.
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#a-schema-may-have-up-to-500-enum-values-across-all-enum-properties

    Note: While OpenAI's documentation states these limits, it appears that as of the test date,
    the API actually accepts enums with more than 500 values. This test validates that our schema
    correctly includes the large enum definition, but the assertion for raising an exception is
    commented out as it's not currently enforced by the API.
    """

    # Create an enum with too many values (over 500) to exceed the documented limit
    # Create 550 enum string constants (exceeding the 500 limit)
    enum_values = {f"VALUE_{i}": f"value_{i}" for i in range(1, 551)}

    # Create custom model that directly produces modified schema with large enum
    class EnumTest(BaseModel):
        category: str
        model_config = ConfigDict(extra="forbid")

        @classmethod
        def model_json_schema(cls, **kwargs):
            schema = super().model_json_schema(**kwargs)
            # Inject large enum directly into schema
            schema["properties"]["category"]["enum"] = list(enum_values.values())
            return schema

    # Verify schema contains enum reference and has enough values
    schema = EnumTest.model_json_schema()
    assert "properties" in schema
    assert "category" in schema["properties"]
    assert "enum" in schema["properties"]["category"]
    assert len(schema["properties"]["category"]["enum"]) > 500, "Schema should contain more than 500 enum values"

    # According to documentation, this should fail with BadRequestError
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Choose a category", format=EnumTest)

        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["enum", "too large", "limit", "Invalid schema", "500"])


@pytest.mark.skip(reason="OpenAI docs state this restriction but API accepts schemas with >15,000 char length")
@pytest.mark.vcr
def test_structured_output_total_string_size():
    """
    Test total string size limitation.

    Reference:
    * In a schema, total string length of all property names, definition names,
      enum values, and const values cannot exceed 15,000 characters.
    * https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses#total-string-length-of-all-property-names-definition-names-enum-values-and-const-values-cannot-exceed-15000-characters

    Note: While OpenAI's documentation states there's a 15,000 character limit, it appears that
    as of the test date, the API actually accepts schemas with total string length exceeding this limit.
    This test validates that our schema correctly includes the large string values, but the assertion
    for raising an exception is commented out as it's not currently enforced by the API.
    """

    # Create a model with an enum that will have very long string values
    class LargeStringModel(BaseModel):
        name: str
        model_config = ConfigDict(extra="forbid")

        @classmethod
        def model_json_schema(cls, **kwargs):
            schema = super().model_json_schema(**kwargs)

            # Create many enum values to exceed the 15,000 character limit
            # Each value has 150 characters, and we'll create 120 of them,
            # resulting in 150 * 120 = 18,000 characters
            padding = "x" * 140
            enum_values = [f"value_{i}_{padding}" for i in range(1, 121)]

            # Add an enum with excessive string length to the name property
            schema["properties"]["name"]["enum"] = enum_values
            return schema

    # Get schema and check total string size
    schema = LargeStringModel.model_json_schema()
    enum_values = schema["properties"]["name"]["enum"]
    total_length = sum(len(val) for val in enum_values)
    assert total_length > 15000, f"Total enum string length should exceed 15,000, got {total_length}"

    # According to documentation, this should fail with BadRequestError
    with Conversation(client=OpenAI(), model=Model.GPT_4O_MINI) as conversation:
        with pytest.raises(BadRequestError) as exc_info:
            conversation.ask("Tell me your name", format=LargeStringModel)

        # Verify the error message
        error_msg = str(exc_info.value)
        assert any(keyword in error_msg for keyword in ["exceed", "limit", "15000", "characters", "Invalid schema"])
