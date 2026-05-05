import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from pydantic import Field

from gwenflow.tools.tool import BaseTool


@dataclass(kw_only=True)
class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = "Searches the web for information related to a given query."
    search_engine_id: Optional[str] = field(default_factory=lambda: os.getenv("WEBSEARCH_SEARCH_ENGINE_ID"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("WEBSEARCH_API_KEY"))
    base_url: str = "https://www.googleapis.com/customsearch/v1"

    def _run(self, query: str = Field(description="The search query."), num_results: int = 10) -> Dict[str, Any]:
        try:
            params = {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(num_results, 10),
                "safe": "active",
            }
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return {
                "success": True,
                "query": query,
                "total_results": data.get("searchInformation", {}).get("totalResults", "0"),
                "search_time": data.get("searchInformation", {}).get("searchTime", "0"),
                "results": self._parse_results(data),
            }
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Erreur de requête: {str(e)}", "query": query, "results": []}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Erreur de parsing JSON: {str(e)}", "query": query, "results": []}
        except Exception as e:
            return {"success": False, "error": f"Erreur inattendue: {str(e)}", "query": query, "results": []}

    def _parse_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        results = []
        for item in data.get("items", []):
            result = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "display_link": item.get("displayLink", ""),
                "formatted_url": item.get("formattedUrl", ""),
            }
            if "pagemap" in item:
                pagemap = item["pagemap"]
                if "metatags" in pagemap and pagemap["metatags"]:
                    metatag = pagemap["metatags"][0]
                    result["description"] = metatag.get("og:description", result["snippet"])
                    result["image"] = metatag.get("og:image", "")
            results.append(result)
        return results
