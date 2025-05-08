# JSON Schema Restrictions for Structured Outputs

## Supported Types
- String
- Number
- Boolean
- Integer
- Object
- Array
- Enum
- anyOf

## Root Object Requirements
- The root level object of a schema must be an object, not anyOf
- A pattern that appears in Zod using a discriminated union produces an anyOf at the top level, which won't work with Structured Outputs

## Field Requirements
- All fields or function parameters must be specified as `required`
- Optional parameters can be emulated by using a union type with `null` (e.g., `["string", "null"]`)

## Object Limitations
- Schema may have up to 100 object properties total
- Up to 5 levels of nesting are allowed
- `additionalProperties: false` must always be set in objects to opt into Structured Outputs

## String Size Limitations
- Total string length of all property names, definition names, enum values, and const values cannot exceed 15,000 characters

## Enum Limitations
- A schema may have up to 500 enum values across all enum properties
- For a single enum property with string values, the total string length of all enum values cannot exceed 7,500 characters when there are more than 250 enum values

## Key Ordering
- Outputs will be produced in the same order as the ordering of keys in the schema

## Unsupported Keywords
For strings:
- minLength
- maxLength
- pattern
- format

For numbers:
- minimum
- maximum
- multipleOf

For objects:
- patternProperties
- unevaluatedProperties
- propertyNames
- minProperties
- maxProperties

For arrays:
- unevaluatedItems
- contains
- minContains
- maxContains
- minItems
- maxItems
- uniqueItems

## Supported Features
- Definitions (`$defs`) are supported to define subschemas
- Recursive schemas are supported, using either:
  - `#` to indicate root recursion
  - Explicit recursion with named definitions

## Models Supporting Structured Outputs
- gpt-4o-mini
- gpt-4o-mini-2024-07-18
- gpt-4o-2024-08-06 
- Later models