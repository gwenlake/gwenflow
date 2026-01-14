import os

from pydantic import Field
import requests
import json
from typing import Dict, Any, List

from gwenflow.tools import BaseTool


class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = "Searches the web for information related to a given query."

    search_engine_id: str = Field(default_factory=lambda: os.getenv("WEBSEARCH_SEARCH_ENGINE_ID"))
    api_key: str = Field(default_factory=lambda: os.getenv("WEBSEARCH_API_KEY"))
    base_url: str = "https://www.googleapis.com/customsearch/v1"

    def _run(self, query: str = Field(description="The search query."), num_results: int = 10) -> Dict[str, Any]:
        """
        Effectue une recherche web en utilisant l'API Google Custom Search.

        Args:
            query: La requête de recherche
            num_results: Le nombre de résultats à retourner (max 10)

        Returns:
            Dict contenant les résultats de recherche
        """
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

            results = self._parse_results(data)

            return {
                "success": True,
                "query": query,
                "total_results": data.get("searchInformation", {}).get("totalResults", "0"),
                "search_time": data.get("searchInformation", {}).get("searchTime", "0"),
                "results": results,
            }

        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Erreur de requête: {str(e)}", "query": query, "results": []}
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"Erreur de parsing JSON: {str(e)}", "query": query, "results": []}
        except Exception as e:
            return {"success": False, "error": f"Erreur inattendue: {str(e)}", "query": query, "results": []}

    def _parse_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Parse les résultats de l'API Google Custom Search.

        Args:
            data: Réponse JSON de l'API

        Returns:
            Liste des résultats parsés
        """
        results = []
        items = data.get("items", [])

        for item in items:
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


if __name__ == "__main__":
    search_tool = WebSearchTool()

    results = search_tool._run("Python programmation tutoriel")

    if results["success"]:
        print(f"Requête: {results['query']}")
        print(f"Nombre total de résultats: {results['total_results']}")
        print(f"Temps de recherche: {results['search_time']} secondes")
        print("\nRésultats:")

        for i, result in enumerate(results["results"], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['link']}")
            print(f"   Description: {result['snippet'][:100]}...")
    else:
        print(f"Erreur: {results['error']}")
