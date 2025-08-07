
"""
This configuration file defines key parameters and model names used across the NLP pipeline, including length thresholds for extracted concepts and pretrained model identifiers for spaCy and SentenceTransformer. These settings ensure consistency and easy customization of model usage and text filtering criteria.
"""
import os

NUM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

BATCH_SIZE = 32  

MIN_CONCEPT_LEN = 3
MAX_CONCEPT_LEN = 100

SPACY_MOD = "en_core_web_lg"
TRANSFORMERS_MOD = "all-MiniLM-L6-v2" 



