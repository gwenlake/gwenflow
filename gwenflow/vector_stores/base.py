from abc import ABC, abstractmethod


class VectorStoreBase(ABC):

    @abstractmethod
    def create_collection(self, name, vector_size, distance):
        """Create a new collection."""
        pass

    @abstractmethod
    def delete_collection(self, name):
        """Delete a collection."""
        pass

    @abstractmethod
    def get_collections(self):
        """List all collections."""
        pass

    @abstractmethod
    def collection_info(self, name):
        """Get information about a collection."""
        pass

    @abstractmethod
    def insert(self, name, vectors, payloads=None, ids=None):
        """Insert vectors into a collection."""
        pass

    @abstractmethod
    def search(self, name, query, limit=5, filters=None):
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, name, vector_id):
        """Delete a vector by ID."""
        pass

    @abstractmethod
    def update(self, name, vector_id, vector=None, payload=None):
        """Update a vector and its payload."""
        pass

    @abstractmethod
    def get(self, name, vector_id):
        """Retrieve a vector by ID."""
        pass
