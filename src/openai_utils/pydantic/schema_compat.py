from typing import Any, get_type_hints, get_origin, get_args

from pydantic import BaseModel, ConfigDict, RootModel, create_model


def make_openai_compatible(model: type[BaseModel]) -> type[BaseModel]:
    """Make a Pydantic model compatible with OpenAI's structured output schema.

    This function creates a new model class using create_model() with:
    - The same fields as the original model
    - model_config with extra="forbid" to ensure additionalProperties: false
    - Recursively applied to nested models

    Args:
        model: The Pydantic model to make compatible

    Returns:
        A new Pydantic model class that generates OpenAI-compatible schemas

    Raises:
        ValueError: If the model cannot be made compatible with OpenAI's restrictions
    """
    # Track processed models to avoid infinite recursion
    processed_models: dict[type[BaseModel], type[BaseModel]] = {}

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

        # Get field definitions for create_model
        field_definitions: dict[str, Any] = {}

        # Get type hints to get field types
        hints = get_type_hints(model_cls)

        # Get field values from model's defaults
        for name, field in model_cls.model_fields.items():
            # Skip private fields
            if name.startswith("_"):
                continue

            # Get the field type (process nested models)
            field_type = hints.get(name)
            processed_type = process_field_type(field_type)

            # Add the field definition with its type and default if any
            if field.is_required():
                # No default value
                field_definitions[name] = processed_type
            else:
                # Has a default value
                field_definitions[name] = (processed_type, field.get_default())

        # Create a new model with default fields plus forbidding extra fields
        new_model = create_model(
            f"OpenAICompatible{model_cls.__name__}",
            __base__=None,  # Don't inherit from original to avoid field conflicts
            __module__=model_cls.__module__,
            __config__=ConfigDict(extra="forbid"),
            **field_definitions,
        )

        # Add docstring
        if model_cls.__doc__:
            new_model.__doc__ = f"OpenAI-compatible version of {model_cls.__name__}.\n\n{model_cls.__doc__}"

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
            # Process each type argument
            processed_args = tuple(process_field_type(arg) for arg in args)
            # Reconstruct the container type with processed arguments
            if processed_args != args:
                return origin[processed_args]

        # Return unchanged for non-model types
        return field_type

    # Process the model
    return process_model(model)
