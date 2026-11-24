"""Claim check: payloads above a threshold are stored in a blob store, history keeps a reference."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, List, Protocol

from temporalio.api.common.v1 import Payload
from temporalio.converter import PayloadCodec


class BlobStore(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes: ...


class InMemoryBlobStore:
    """For tests and single-process demos only."""

    def __init__(self) -> None:
        self.blobs: Dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.blobs[key] = data

    async def get(self, key: str) -> bytes:
        return self.blobs[key]


class S3BlobStore:
    """Any S3-compatible bucket (AWS, Scaleway, MinIO...). Extra kwargs go to `boto3.client`."""

    def __init__(self, bucket: str, prefix: str = "temporal-payloads/", **client_kwargs: Any) -> None:
        import boto3

        self.bucket, self.prefix = bucket, prefix
        self.client = boto3.client("s3", **client_kwargs)

    async def put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(self.client.put_object, Bucket=self.bucket, Key=self.prefix + key, Body=data)

    async def get(self, key: str) -> bytes:
        response = await asyncio.to_thread(self.client.get_object, Bucket=self.bucket, Key=self.prefix + key)
        return response["Body"].read()


class ClaimCheckCodec(PayloadCodec):
    """Payloads above `threshold` bytes go to a blob store; history keeps a reference.

    Applies to every payload (activity inputs and results, workflow inputs and
    results), so a 50 MB CSV node or a long agent history never hits Temporal's
    2 MB limit. Keys are content hashes: the same data is stored once. Blobs
    must live as long as the histories that reference them can be replayed.

        converter = dataclasses.replace(DataConverter.default, payload_codec=ClaimCheckCodec(store))
        client = await Client.connect(address, data_converter=converter)
    """

    ENCODING = b"gwenflow/claim-check"

    def __init__(self, store: BlobStore, threshold: int = 128 * 1024) -> None:
        self.store, self.threshold = store, threshold

    async def encode(self, payloads: Any) -> List[Payload]:
        encoded = []
        for payload in payloads:
            if payload.ByteSize() <= self.threshold:
                encoded.append(payload)
                continue
            data = payload.SerializeToString()
            key = hashlib.sha256(data).hexdigest()
            await self.store.put(key, data)
            encoded.append(Payload(metadata={"encoding": self.ENCODING}, data=key.encode()))
        return encoded

    async def decode(self, payloads: Any) -> List[Payload]:
        decoded = []
        for payload in payloads:
            if payload.metadata.get("encoding") != self.ENCODING:
                decoded.append(payload)
                continue
            decoded.append(Payload.FromString(await self.store.get(payload.data.decode())))
        return decoded
