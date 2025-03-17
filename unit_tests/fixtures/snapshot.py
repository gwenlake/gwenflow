import pytest
from syrupy.extensions.json import JSONSnapshotExtension


# Snapshot testing
@pytest.fixture
def snapshot_json(snapshot):
    return snapshot.with_defaults(extension_class=JSONSnapshotExtension)
