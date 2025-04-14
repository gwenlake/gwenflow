import gzip
import json
import os

import pytest


def decompress_json_matcher(r1, r2):
    assert r1.headers.get("content-type") == r2.headers.get("content-type")
    assert r1.headers.get("content-encoding") == r2.headers.get("content-encoding")
    if r1.headers.get("content-type") == "application/json" and r1.headers.get("content-encoding") == "gzip":
        assert json.loads(gzip.decompress(r1.body)) == json.loads(gzip.decompress(r2.body))


@pytest.fixture
def vcr_config():
    return {
        "ignore_hosts": [
            # Ignore httpclient test host
            "testserver",
            # Ignore tiktoken cache, since otherwise VCR complains here
            # Details: https://github.com/openai/tiktoken/issues/232
            "openaipublic.blob.core.windows.net",
        ]
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    test_dir = request.node.fspath.dirname
    return os.path.join(test_dir, "__cassettes__")
