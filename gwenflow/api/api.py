import os
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from httpx import AsyncClient, Client

from gwenflow.version import __version__


@dataclass(kw_only=True)
class Api:
    base_url: str = "https://api.gwenlake.com"
    api_key: Optional[str] = None
    timeout: int = 300

    def _get_headers(self) -> dict:
        api_key = self.api_key or os.environ.get("GWENLAKE_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the GWENLAKE_API_KEY environment variable"
            )
        return {
            "Content-Type": "application/json",
            "user-agent": f"gwenflow/{__version__}",
            "Authorization": f"Bearer {api_key}",
        }

    @cached_property
    def client(self) -> Client:
        return Client(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )

    @cached_property
    def async_client(self) -> AsyncClient:
        return AsyncClient(
            base_url=self.base_url,
            headers=self._get_headers(),
            timeout=self.timeout,
        )


api = Api()
