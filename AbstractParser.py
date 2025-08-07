from general_utils import *
from config import *
from typing import List, Dict
import AdvancedEmbedder
import ConceptExtractor

class AbstractParser:
    """
    Parses a scientific abstract to extract concepts and compute their relevance
    to the abstract using a given concept extractor and embedder.
    """
    def __init__(self, my_abstract:str, concept_extractor: ConceptExtractor, embedder: AdvancedEmbedder):
        """
        Initializes the parser with a cleaned abstract, concept extractor, and embedder.
        
        Args:
            my_abstract (str): The raw abstract text.
            concept_extractor (ConceptExtractor): The object for extracting and lemmatizing concepts.
            embedder (AdvancedEmbedder): The object for computing semantic relevance.
        """
        self.clean_abstract = clean_text(my_abstract) 
        self.concept_extractor = concept_extractor
        self.embedder = embedder
        
    def clean_text(self) -> str:
        """
        Returns the cleaned abstract text.

        Returns:
            str: Cleaned abstract.
        """
        return self.clean_abstract
    
    def raw_concepts(self) -> List[str]:
        """
        Extracts raw noun phrases from the abstract.

        Returns:
            List[str]: List of raw concept phrases.
        """
        return self.concept_extractor.extract_noun_phrases(self.clean_abstract)
    
    def clean_concepts(self) -> List[str]:
        """
        Returns a list of lemmatized concept phrases.

        Returns:
            List[str]: Cleaned (lemmatized) concept phrases.
        """
        return [self.concept_extractor.lemmatize(concept) for concept in self.raw_concepts()]
    
    def concept_relevances(self) -> List[float]:
        """
        Computes a relevance score for each concept with respect to the abstract.

        Returns:
            Dict[str, float]: Dictionary mapping each concept to its relevance score.
        """
        return [self.embedder.calculate_relevance(concept, self.clean_abstract) for concept in self.raw_concepts()]

