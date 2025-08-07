"""
This module provides utility functions for cleaning text content by removing HTML and LaTeX/math elements, primarily using Pandoc with a Lua filter to strip math. It also includes helper functions for joining tokenized words into cleaned phrases and postprocessing a DataFrame of extracted concepts by filtering, counting, and aggregating relevant statistics.

"""


from bs4 import BeautifulSoup
import re
import config
import subprocess
import string
import pypandoc
import matplotlib.pyplot as plt

def convert_with_pandoc(text: str) -> str:
    """
    Convert input HTML/LaTeX text to plain text using Pandoc, applying a Lua filter to remove math elements.

    Args:
        text (str): The input text containing HTML and/or LaTeX content.

    Returns:
        str: Cleaned plain text with math elements removed. Returns original text on failure.
    """

    try:
       
        plain_text = pypandoc.convert_text(text, to='plain', format='html')
        plain_text = pypandoc.convert_text(plain_text, to='plain', format='latex',
                      extra_args=['--lua-filter=remove-math.lua']) 
        return plain_text
    except Exception as e:
        print("Pandoc error:", e)
        return text  # Return original text if Pandoc fails



def clean_text(text): #change function name to remove_html_latex()
    text = convert_with_pandoc(text) 
    text = " ".join(text.split())
    return text

def join_words(words: list[str], stop_words) -> str:
    """
    Join a list of words into a cleaned phrase by removing stop words and punctuation, and normalizing case.

    Args:
        words (list[str]): List of word tokens.
        stop_words (set or list): Collection of stop words to exclude.

    Returns:
        str: A single cleaned phrase string.
    """
    noun_phrase = [word.strip().lower() for word in words]
    noun_phrase = " ".join([word.strip() for word in noun_phrase if word not in stop_words
                                                            and word not in string.punctuation                                 
                                                                    ])
    return noun_phrase


def postprocess(df_temp, col_name: str):
    """
    Filter and aggregate concept data from a DataFrame.

    - Keeps rows where specified column contains alphabetic characters.
    - Filters by length constraints (min and max concept length from config).
    - Computes concept frequency per article and average similarity score.
    - Removes duplicate concepts per article.
    - Adds counts of concepts per article and counts of articles per concept.

    Args:
        df_temp (pandas.DataFrame): Input DataFrame with extracted concepts.
        col_name (str): Column name in DataFrame containing concept strings.

    Returns:
        pandas.DataFrame: Processed DataFrame with aggregated statistics.
    """
    df_temp = df_temp[df_temp[col_name].str.contains("[a-zA-Z]", regex=True, na=False)]
    
    df_temp[col_name] = df_temp[col_name].str.strip()
    
    df_temp = df_temp[
        (df_temp[col_name].str.len() >= config.MIN_CONCEPT_LEN) &
        (df_temp[col_name].str.len() <= config.MAX_CONCEPT_LEN)
    ]

    df_temp["concept_freq_per_art"] = df_temp.groupby(["article_id", "clean_concept"])["clean_concept"].transform("count") 
    df_temp["avg_relevance"] = df_temp.groupby(["article_id", "clean_concept"])["concept_relevance"].transform("mean")
    df_final = df_temp[["article_id", "clean_concept", "avg_relevance", "concept_freq_per_art"]].drop_duplicates() #ensure unique count of concepts per article 
    df_final["concept_count_per_art"] = df_final.groupby(["article_id"])["clean_concept"].transform("count")
    df_final["art_count"] = df_final.groupby(["clean_concept"])["clean_concept"].transform("count")
    
    return df_final






