# config.py
import os

# API keys
os.environ["GROQ_API_KEY"] = "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"
os.environ["HF_TOKEN"] = "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"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Paths
USER_DATA_PATH = "/home/sracha/Sattvastha/pipeline/new_pipeline/user_data.json"
RAG_BASE_DIRECTORY = "/home/sracha/Sattvastha/pipeline/all_content"
RAG_CATEGORIES = ["Ayurveda", "Lifestyle", "psychology", "Yoga", "Mental_health"]
# Prompt templates
QUESTION_CLASSIFICATION_PROMPT = """
Classify the following user input into one of these categories:
1. "question" - If the user is asking a factual question that could be answered with knowledge
2. "general" - If the user is just chatting or expressing feelings

User Input: {user_input}

Respond with only one word: either "question" or "general"
"""