"""Project-wide configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    temperature: float = 0.7
    max_tokens: int = 1024
    intermediate_max_tokens: int = 512
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
class GuardConfig:
    min_answer_length: int = 10
    warn_answer_length: int = 30
    repetition_warn_ratio: float = 0.5
    repetition_fail_ratio: float = 0.3
    rouge_warn_threshold: float = 0.10
    min_tr_stopwords: int = 2
    enable_pii_check: bool = True
    enable_injection_check: bool = True
    enable_rouge: bool = True


@dataclass(frozen=True)
class AppConfig:
    ollama: OllamaConfig = OllamaConfig()
    ingest: IngestConfig = IngestConfig()
    qdrant: QdrantConfig = QdrantConfig()
    crew: CrewConfig = CrewConfig()
    guard: GuardConfig = GuardConfig()
    language: str = "tr"
    data_dir: str = "data"
    verbose: bool = False
