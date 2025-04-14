import context
from gwenflow import Document

cluster_chat = [
    Document(content="Le chat dort sur le canapé."),
    Document(content="Le chat se repose sur le canapé."),
    Document(content="Un chat paresseux dort sur le canapé."),
    Document(content="Le félin se prélasse sur le canapé.")
]

cluster_chien = [
    Document(content="Le chien court dans le jardin."),
    Document(content="Un chien joue dans le jardin."),
    Document(content="Le chien s'amuse dans le jardin ensoleillé.")
]

cluster_pluie = [
    Document(content="Il pleut abondamment en ville."),
    Document(content="La pluie tombe sans relâche sur la ville."),
    Document(content="Les rues de la ville sont inondées par la pluie.")
]

divers = [
    Document(content="La technologie évolue rapidement."),
    Document(content="L'intelligence artificielle transforme le monde."),
    Document(content="Les ordinateurs quantiques pourraient révolutionner l'informatique."),
    Document(content="La bibliothèque regorge de livres anciens."),
    Document(content="Les montagnes offrent des panoramas spectaculaires.")
]

from typing import List
import numpy as np

from pydantic import BaseModel
from gwenflow.types import Document
from gwenflow.embeddings import GwenlakeEmbeddings


class MMR(BaseModel):
    lambda_param: float = 0.5
    top_k: int = 10

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15))

    def rerank(self, query_embedding: List[float], documents: List["Document"]) -> List["Document"]:
        selected = []
        remaining = documents.copy()
        while len(selected) < self.top_k and remaining:
            scores = [
                self.lambda_param * self.cosine_similarity(query_embedding, doc.embedding)
                - (1 - self.lambda_param) * (max(self.cosine_similarity(doc.embedding, sel.embedding) for sel in selected) if selected else 0)
                for doc in remaining
            ]
            idx = int(np.argmax(scores))
            selected.append(remaining.pop(idx))
        return selected
    
embeddings_model = GwenlakeEmbeddings(model="e5-base-v2")

query = "Le chat chasse une souris"
query_embedding = embeddings_model(query)
print(query_embedding)
#selct = MMR.rerank(query_embedding="")