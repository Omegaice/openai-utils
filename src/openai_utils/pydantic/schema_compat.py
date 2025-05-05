from typing import Any, get_type_hints, get_origin, get_args, TypeVar, cast, Set, Dict, Union
import warnings

from pydantic import BaseModel, ConfigDict, RootModel

T = TypeVar("T", bound=BaseModel)

# Unsupported validation keywords by type
UNSUPPORTED_STRING_KEYWORDS = {"minLength", "maxLength", "pattern", "format"}
UNSUPPORTED_NUMBER_KEYWORDS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
UNSUPPORTED_OBJECT_KEYWORDS = {
    "patternProperties",
    "unevaluatedProperties",
    "propertyNames",
    "minProperties",
    "maxProperties",
}
UNSUPPORTED_ARRAY_KEYWORDS = {
    "unevaluatedItems",
    "contains",
    "minContains",
    "maxContains",
    "minItems",
    "maxItems",
    "uniqueItems",
}


def make_openai_compatible(model: type[T], warn_on_changes: bool = True) -> type[T]:
    """Make a Pydantic model compatible with OpenAI's structured output schema.

    This function creates a new model class that:
    - Has the same fields as the original model
    - Sets model_config with extra="forbid" to ensure additionalProperties: false
    - Applies this recursively to nested models
    - Removes unsupported validation keywords from the schema

    OpenAI's structured output has several restrictions on JSON Schema keywords. This function
    automatically removes these unsupported keywords:

    - For strings: minLength, maxLength, pattern, format
    - For numbers: minimum, maximum, exclusiveMinimum, exclusiveMaximum, multipleOf
    - For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
    - For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems

    By default, warnings are emitted when unsupported keywords are removed, which can help
    identify potential validation issues.

    Args:
        model: The Pydantic model to make compatible with OpenAI's structured output
        warn_on_changes: Whether to emit warnings when unsupported keywords are removed.
                         Set to False to disable warnings.

    Returns:
        A new Pydantic model class that is compatible with OpenAI's schema
        and preserves the type information of the original model

    Raises:
        ValueError: If the model cannot be made compatible with OpenAI's restrictions,
                   such as if it uses a RootModel with Union types

    Examples:
        ```python
        from pydantic import BaseModel, Field
        from openai_utils.pydantic.schema_compat import make_openai_compatible

        class User(BaseModel):
            name: str = Field(min_length=3, max_length=50)  # These will be removed
            age: int = Field(ge=0, le=120)  # These will be removed

        # Create OpenAI-compatible version
        OpenAIUser = make_openai_compatible(User)

        # Use with conversation.ask
        result = conversation.ask("Give me user info", format=OpenAIUser)
        ```
    """
    # Track processed models to avoid infinite recursion
    processed_models: Dict[type[BaseModel], type[BaseModel]] = {}
    # Track issued warnings to avoid duplicates
    issued_warnings: Set[str] = set()

    def process_model(model_cls: type[BaseModel]) -> type[BaseModel]:
        """Process a model and create an OpenAI-compatible version."""
        # Check if we already processed this model
        if model_cls in processed_models:
            return processed_models[model_cls]

        # Check for RootModel with Union types - not supported by OpenAI
        if issubclass(model_cls, RootModel):
            hints = get_type_hints(model_cls)
            if "root" in hints and "|" in str(hints["root"]):
                raise ValueError(
                    "RootModel with Union type is not supported by OpenAI. "
                    "The root level schema must be an object, not anyOf/Union."
                )

        # Get type hints for field types
        hints = get_type_hints(model_cls)

        # Get field values from model's defaults
        field_definitions = {}
        annotations = {}

        for name, field in model_cls.model_fields.items():
            # Skip private fields
            if name.startswith("_"):
                continue

            # Get the field type (process nested models)
            field_type = hints.get(name)
            try:
                processed_type = process_field_type(field_type)
                annotations[name] = processed_type
            except TypeError as e:
                # Handle error for union types
                if "not subscriptable" in str(e) and "|" in str(field_type):
                    # For pipe operator unions, just keep the original type
                    # This preserves the model structure while allowing the schema to be processed
                    annotations[name] = field_type
                else:
                    raise

            # Add the field definition with its default if any
            if not field.is_required():
                field_definitions[name] = field.get_default()

        # Create the new model class with explicit class definition
        class_dict = {
            "__annotations__": annotations,
            "model_config": ConfigDict(extra="forbid"),
            **field_definitions,
        }

        # Create the class
        new_model = type(f"OpenAICompatible{model_cls.__name__}", (BaseModel,), class_dict)

        # Set the docstring
        if model_cls.__doc__:
            new_model.__doc__ = f"OpenAI-compatible version of {model_cls.__name__}.\n\n{model_cls.__doc__}"

        # Set the module to match the original
        new_model.__module__ = model_cls.__module__

        # Cache the processed model
        processed_models[model_cls] = new_model
        return new_model

    def process_field_type(field_type: Any) -> Any:
        """Process a field type, recursively handling nested BaseModels."""
        # Handle direct BaseModel fields
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            return process_model(field_type)

        # Handle container types (list, dict, Optional, Union, etc.)
        origin = get_origin(field_type)
        if origin is not None:
            args = get_args(field_type)

            # Special case for Union/Optional types
            if origin is Union:
                # Process each type argument
                processed_args = tuple(process_field_type(arg) for arg in args)
                # Return a new Union with processed args
                return Union[processed_args]

            # Process arguments for other container types
            processed_args = tuple(process_field_type(arg) for arg in args)
            # Reconstruct the container type with processed arguments
            if processed_args != args:
                return origin[processed_args]

        # Return unchanged for non-model types or new union types (|)
        return field_type

    def _process_schema(schema: Dict[str, Any], path: str = "$") -> Dict[str, Any]:
        """Process a JSON schema to remove unsupported validation keywords.

        This helper function recursively traverses a JSON schema and removes
        any unsupported keywords for OpenAI's structured output. The function
        processes nested schemas, including those in properties, items, anyOf/oneOf/allOf,
        and $defs sections.

        Args:
            schema: The JSON schema to process
            path: The current path in the schema (used for warning messages)

        Returns:
            The processed schema with unsupported keywords removed
        """
        # Make a copy to avoid modifying the original
        schema = schema.copy()

        # Process by schema type
        if schema.get("type") == "string":
            _remove_unsupported_keywords(schema, UNSUPPORTED_STRING_KEYWORDS, "string", path)
        elif schema.get("type") in ("number", "integer"):
            _remove_unsupported_keywords(schema, UNSUPPORTED_NUMBER_KEYWORDS, "number", path)
        elif schema.get("type") == "object":
            _remove_unsupported_keywords(schema, UNSUPPORTED_OBJECT_KEYWORDS, "object", path)

            # Process properties
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    prop_path = f"{path}.properties.{prop_name}"
                    schema["properties"][prop_name] = _process_schema(prop_schema, prop_path)
        elif schema.get("type") == "array":
            _remove_unsupported_keywords(schema, UNSUPPORTED_ARRAY_KEYWORDS, "array", path)

            # Process items
            if "items" in schema:
                schema["items"] = _process_schema(schema["items"], f"{path}.items")

        # Process anyOf, oneOf, allOf
        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                schema[key] = [
                    _process_schema(subschema, f"{path}.{key}[{i}]") for i, subschema in enumerate(schema[key])
                ]

        # Process $defs section
        if "$defs" in schema:
            for def_name, def_schema in schema["$defs"].items():
                def_path = f"{path}.$defs.{def_name}"
                schema["$defs"][def_name] = _process_schema(def_schema, def_path)

        return schema

    def _remove_unsupported_keywords(schema: Dict[str, Any], keywords: Set[str], type_name: str, path: str) -> None:
        """Remove unsupported keywords from a schema.

        This helper function removes unsupported validation keywords for a specific
        schema type (string, number, object, array) and emits warnings if requested.

        Args:
            schema: The schema object to modify
            keywords: Set of unsupported keywords to check for and remove
            type_name: Type name for the warning message (string, number, object, array)
            path: JSON path to the current schema location for warning messages
        """
        for keyword in keywords:
            if keyword in schema:
                if warn_on_changes and f"{type_name}.{keyword}" not in issued_warnings:
                    warnings.warn(
                        f"Removing unsupported {type_name} validation keyword '{keyword}' at {path}. "
                        f"OpenAI does not support {keyword} in structured output schemas.",
                        UserWarning,
                    )
                    issued_warnings.add(f"{type_name}.{keyword}")
                del schema[keyword]

    # Process the model and cast the result to preserve IDE type information
    result = process_model(model)

    # Process the schema to remove unsupported keywords once on creation
    # This ensures warnings are emitted during model creation
    schema = result.model_json_schema()
    processed_schema = _process_schema(schema)

    # Create a method that returns our pre-processed schema
    # We do this to avoid monkey-patching the model_json_schema method directly
    # which would cause type errors
    setattr(result, "__openai_compatible_schema__", processed_schema)

    # Create a custom descriptor to override the model_json_schema method
    # This approach preserves type information
    class ModelJsonSchemaDescriptor:
        def __get__(self, obj: Any, objtype: Any = None) -> Any:
            # Return a function that always returns our processed schema
            def model_json_schema_override(*args: Any, **kwargs: Any) -> Dict[str, Any]:
                return processed_schema

            return model_json_schema_override

    # Use the descriptor to replace the model_json_schema method
    setattr(type(result), "model_json_schema", ModelJsonSchemaDescriptor())

    return cast(type[T], result)
