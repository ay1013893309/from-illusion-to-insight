import os
from dotenv import load_dotenv

load_dotenv()

# Dataset configuration
DATASET_NAME_RUN = os.getenv('DATASET_NAME_RUN', 'camel')

# LLM configuration
LLM7_KEYS = os.getenv('LLM7_KEYS', 'your-llm7-key-here').split(',')
BASE_URL = os.getenv('BASE_URL', 'https://api.llm7.io/v1')

# Data directory
DATA_DIR = os.getenv('DATA_DIR', 'data/')

# LLM models list
llm7_all_models = [
    "gpt-4.1-nano-2025-04-14",
    "gpt-5-mini",
    "deepseek-v4-flash",
    "mistral-small-3.1-24b-instruct-2503",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "codestral-2405",
    "codestral-2501",
    "ministral-8b-2512",
    "gemini-2.5-flash-lite",
    "gemini-search",
    "llama-3.1-8B-instruct",
    "bidara",
    "glm-4.5-flash",
]
