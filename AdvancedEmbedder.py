from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AdvancedEmbedder():
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.article_id = None
        self.abstract_embedding = None

    def abstract_embed(self, abstract):
        self.abstract_embedding = self.model.encode([abstract])

    # def calculate_similarity(self, phrase: str, art_id: str, abstract: str) -> float:
    #     """
    #     Compute cosine similarity between the embedding of a given phrase and the embedding of an abstract.

    #     Args:
    #         phrase (str): The phrase whose similarity to the abstract will be calculated.
    #         art_id (str): The ID of the abstract in `self.abstract_embeddings`.

    #     Returns:
    #         float: Cosine similarity score between the phrase and the abstract embedding.
    #     """

    #     if art_id != self.article_id:
    #         # print("Hold on, re-training the model")
    #         self.article_id = art_id
    #         self.abstract_embed(abstract)

    #     # print("No re-training needed")
    #     phrase_embedding = self.model.encode([phrase])
    #     paragraph_embedding = self.abstract_embedding
        
    #     # if paragraph_embedding.ndim == 1:
    #     #     paragraph_embedding = np.expand_dims(paragraph_embedding, axis=0)

    #     similarity = cosine_similarity(phrase_embedding, paragraph_embedding)[0][0]
    #     return float(similarity)
    

    def calculate_relevance(self, phrase: str, abstract: str) -> float:
        """
        Compute cosine similarity between the embedding of a given phrase and the embedding of an abstract.

        Args:
            phrase (str): The phrase whose similarity to the abstract will be calculated.
            art_id (str): The ID of the abstract in `self.abstract_embeddings`.

        Returns:
            float: Cosine similarity score between the phrase and the abstract embedding.
        """

        # if art_id != self.article_id:
        #     # print("Hold on, re-training the model")
        #     self.article_id = art_id
        #     self.abstract_embed(abstract)

        # print("No re-training needed")
        phrase_embedding = self.model.encode([phrase])
        paragraph_embedding = self.model.encode([abstract])
        
        # if paragraph_embedding.ndim == 1:
        #     paragraph_embedding = np.expand_dims(paragraph_embedding, axis=0)

        similarity = cosine_similarity(phrase_embedding, paragraph_embedding)[0][0]
        return float(similarity)

