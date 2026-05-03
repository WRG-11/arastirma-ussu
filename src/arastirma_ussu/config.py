"""Project-wide configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "dolphin-mistral"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 120


@dataclass(frozen=True)
class IngestConfig:
    doc_dir: str = "data/documents"
    index_dir: str = "data/index"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 3


@dataclass(frozen=True)
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = False
    documents_collection: str = "documents"
    conversations_collection: str = "conversations"
    max_conversation_points: int = 5000
    memory_score_threshold: float = 0.65


@dataclass(frozen=True)
class CrewConfig:
    timeout: int = 600               # global crew execution timeout (seconds)
    max_agent_iter: int = 3          # max ReAct iterations per agent
    temperature: float = 0.4         # LLM temperature for crew agents
    verbose: bool = False            # CrewAI verbose logging
    llm_request_timeout: int = 60    # per-request Ollama timeout (seconds)


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig = OllamaConfig()
    ingest: IngestConfig = IngestConfig()
    qdrant: QdrantConfig = QdrantConfig()
    crew: CrewConfig = CrewConfig()
    language: str = "tr"
    data_dir: str = "data"
    verbose: bool = False
