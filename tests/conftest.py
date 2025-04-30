from typing import Any
import pytest


def remove_response_headers(headers):
    def before_record_response(response: dict[str, Any]):
        if "headers" in response:
            response["headers"] = {name: value for name, value in response["headers"].items() if name not in headers}
        return response

    return before_record_response


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", None),
            ("cookie", None),
            ("x-stainless-arch", None),
            ("x-stainless-async", None),
            ("x-stainless-lang", None),
            ("x-stainless-os", None),
            ("x-stainless-package-version", None),
            ("x-stainless-read-timeout", None),
            ("x-stainless-retry-count", None),
            ("x-stainless-runtime", None),
            ("x-stainless-runtime-version", None),
        ],
        "before_record_response": remove_response_headers(
            [
                "Set-Cookie",
                "openai-organization",
                "x-request-id",
                "CF-RAY",
                "cf-cache-status",
            ]
        ),
    }
