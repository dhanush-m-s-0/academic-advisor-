"""Traffic Analyzer — configuration loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK   = "mixtral-8x7b-32768"

TRAFFIC_CHROMA_DB_PATH = "./traffic_chroma_db"
EMBED_DIMENSION        = 1536
MAX_TOKENS             = 2048
TEMPERATURE            = 0.4
MEMORY_WINDOW          = 8
RAG_TOP_K              = 5
CHUNK_SIZE             = 900
CHUNK_OVERLAP          = 150
