import aiohttp
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BaseAPIClient:
    def __init__(self, api_key: str, base_url: str):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        if not base_url or not base_url.startswith("http"):
            raise ValueError("Base URL must be a valid HTTP(S) URL.")

        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    async def search(self, query: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement the 'search' method.")

    async def _make_request(self, endpoint: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("Client session is not initialized. Please use 'async with' context manager.")
        
        request_headers = headers if headers is not None else {}

        try:
            # Handle empty endpoint for APIs that query the base URL directly
            url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
            
            async with self.session.get(url, params=params, headers=request_headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f"API request failed with status {e.status} for {endpoint}: {e.message}")
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred during API request to {endpoint}: {e}")
            raise


class GoogleFactCheckClient(BaseAPIClient):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://factchecktools.googleapis.com/v1alpha1"
        )
    
    async def search(self, query: str, language_code: str = "en") -> Dict[str, Any]:
        params = {
            "query": query,
            "key": self.api_key,
            "languageCode": language_code
        }
        return await self._make_request("claims:search", params)


class GoogleCustomSearchClient(BaseAPIClient):
    def __init__(self, api_key: str, cse_id: str):
        super().__init__(
            api_key=api_key,
            base_url="https://www.googleapis.com/customsearch/v1"
        )
        self.cse_id = cse_id

    async def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        params = {
            "q": query,
            "key": self.api_key,
            "cx": self.cse_id,
            "num": num_results,
            "lr": "lang_en"
        }
        return await self._make_request("", params)


class SemanticScholarClient(BaseAPIClient):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.semanticscholar.org/graph/v1"
        )
    
    async def search(self, query: str, limit: int = 10) -> Dict[str, Any]:
        params = {"query": query, "limit": limit}
        headers = {"x-api-key": self.api_key}
        return await self._make_request("paper/search", params, headers=headers)


class IndianKanoonClient(BaseAPIClient):
    def __init__(self, api_key: str):
        super().__init__(
            api_key=api_key,
            base_url="https://api.indiankanoon.org"
        )
    
    async def search(self, query: str, pagenum: int = 0) -> Dict[str, Any]:
        """
        Searches Indian Kanoon.
        query: input string (can include operators like ANDD, ORR)
        """
        if not query:
            raise ValueError("Search query cannot be empty.")

        params = {
            "formInput": query,
            "pagenum": pagenum
        }
        # Server-side authentication using Token
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Accept": "application/json"
        }
        return await self._make_request("search/", params, headers=headers)