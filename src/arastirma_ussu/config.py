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
class AppConfig:
    ollama: OllamaConfig = OllamaConfig()
    language: str = "tr"
    data_dir: str = "data"
    verbose: bool = False
    crew_timeout_warn: int = 600  # seconds — Layer 4 stopwatch threshold
