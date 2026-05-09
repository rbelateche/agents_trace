from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://agents_trace:agents_trace@localhost:5432/agents_trace"
    redis_url: str = "redis://localhost:6379"
    environment: str = "development"


settings = Settings()
