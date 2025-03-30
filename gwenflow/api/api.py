import os
from httpx import Client, AsyncClient
from typing import Optional
from pydantic import BaseModel

from gwenflow.version import __version__


class Api(BaseModel):
 
    base_url: Optional[str] = "https://api.gwenlake.com"
    api_key: Optional[str] = None
    timeout: Optional[int] = 300

    def get_headers(self) -> dict:
        api_key = self.api_key
        if api_key is None:
            api_key = os.environ.get("GWENLAKE_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the GWENLAKE_API_KEY environment variable"
            )
        headers = {
            "Content-Type": "application/json",
            "user-agent": f"gwenflow/{__version__}",
            "Authorization": f"Bearer {api_key}",
        }
        return headers

    @property
    def client(self) -> Client:
        return Client(
            base_url=self.base_url,
            headers=self.get_headers(),
            timeout=self.timeout,
        )

    @property
    def async_client(self) -> AsyncClient:
        return AsyncClient(
            base_url=self.base_url,
            headers=self.get_headers(),
            timeout=self.timeout,
        )

api = Api()
