"""
This module defines a ConceptExtractor class for extracting meaningful noun phrases and lemmatizing phrases from text using a spaCy model. It disables named entity recognition for performance, removes pronouns, and supports lemmatization with stop word filtering.
"""

import numpy as np
import importlib
import subprocess
import sys
import general_utils 
importlib.reload(general_utils)
import spacy
from typing import List, Tuple, Dict
from spacy.matcher import Matcher
import config


class ConceptExtractor:
    def __init__(self, model_name):

        self.nlp = spacy.load(model_name, disable=["ner"])
        self.stop_words = self.nlp.Defaults.stop_words.copy()  # make a copy, just to be safe
        self.stop_words.discard("of")
        self.stop_words.discard("and")


    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases and selected named entities from the input text.

        Pronouns are excluded from noun chunks. Named entities of types PERSON, ORG,
        GPE, PRODUCT, EVENT, and WORK_OF_ART are included.

        Args:
            text (str): The input text to process.

        Returns:
            List[str]: List of extracted noun phrases and named entities.
        """
        doc = self.nlp(text)
        noun_phrases = []

        #STEP 1

        for chunk in doc.noun_chunks:
            # if not chunk.root.pos_ in ['PRON']:
                noun_phrases.append(chunk) #pronouns to be removed later as stop words
         
        # STEP 2

        
        #catch nested noun phrases with preposition of in between

        matcher = Matcher(self.nlp.vocab)

        # Define a generic noun + preposition + noun/verb pattern
        pattern = [
                {"POS": "DET", "OP": "?"},           # Optional determiner before first noun (e.g., "the", "a")
                {"POS": "NOUN"},                     # First noun (e.g., "system")
                {"POS": "ADP", "LOWER": "of"},       # Preposition "of"
                {"POS": "ADJ", "OP": "*"},           # Zero or more adjectives between preposition and second noun
                {"POS": "NOUN"},                     # Second noun (e.g., "equations")
            ]

        matcher.add("NOUN_PREP_NOUNVERB", [pattern])

        matches = matcher(doc)
        # temp = []
        for match_id, start, end in matches:
            span = doc[start:end]
            noun_phrases.append(span)
            # temp.append(span.text)

        #TODO: step 3: remove sub-np (ones that are originally part of bigger np, noun chunks in step 1 that are sub-chunks of the ones in step 2, using ranges as uniq id)
        final_spans = []

        for span in noun_phrases:
            if not any(
                self.is_subrange(span, other) and span != other
                for other in noun_phrases
            ):
                final_spans.append(span.text)
                       
        return final_spans
            

    def is_subrange(self, subr, r):
        """
        Checks if `subr` is entirely within the range of `r`. Only works when both params are spacy objects
        
        Parameters:
            subr (Span): The possible subrange span (e.g., "model")
            r (Span): The parent range span (e.g., "model of robust estimation")
        
        Returns:
            bool: True if `subr` is fully inside `r`, False otherwise
        """
        return subr.start_char >= r.start_char and subr.end_char <= r.end_char
         

    def lemmatize(self, phrase: str):
        """
        Lemmatize the given phrase and remove stop words.

        Args:
            phrase (str): The phrase to lemmatize.

        Returns:
            str: The lemmatized phrase with stop words removed.
        """
        lemma = self.nlp(phrase)
        lemma = [token.lemma_ for token in lemma]
        
        return general_utils.join_words(lemma,  self.stop_words)

    

