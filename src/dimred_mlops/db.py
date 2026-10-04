"""Database schema shared by the API (writes) and the monitoring flow (reads).

PostgreSQL in Docker Compose; any SQLAlchemy URL works (SQLite is used in tests).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One row per /predict request: the input, both classifiers' answers and the flag."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    source: Mapped[str] = mapped_column(String(32), default="api")
    pixels: Mapped[list] = mapped_column(JSON)  # 784 integers 0-255
    digit_raw: Mapped[int] = mapped_column(Integer)
    digit_pca: Mapped[int] = mapped_column(Integer)
    latency_raw_ms: Mapped[float] = mapped_column(Float)
    latency_pca_ms: Mapped[float] = mapped_column(Float)
    reconstruction_error: Mapped[float] = mapped_column(Float)
    error_threshold: Mapped[float] = mapped_column(Float)
    is_unusual: Mapped[bool] = mapped_column(Boolean, index=True)
    model_versions: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)


def make_engine(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
