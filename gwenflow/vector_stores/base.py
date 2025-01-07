from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional

from gwenflow.types import Document


class VectorStoreBase(ABC):

    @abstractmethod
    def get_collections(self):
        """List all collections."""
        pass

    @abstractmethod
    def create(self):
        """Create collection."""
        pass

    @abstractmethod
    def drop(self):
        """Delete collection."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count points in collection."""
        pass

    @abstractmethod
    def info(self):
        """Get information about collection."""
        pass

    @abstractmethod
    def insert(self, documents: list[Document]):
        """Insert documents into collection."""
        pass

    @abstractmethod
    def search(self, query, limit=5, filters=None) -> list[Document]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, id):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def get(self, id):
        """Retrieve a vector by ID."""
        pass
