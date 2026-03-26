"""
app/core/database.py - SQLAlchemy async database setup
"""
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


# ─── Engine ───────────────────────────────────────────────────────────────────

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


# ─── Base ─────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ─── Models ───────────────────────────────────────────────────────────────────

class FusionJob(Base):
    __tablename__ = "fusion_jobs"

    id = Column(String(36), primary_key=True)
    status = Column(String(20), default="pending")
    progress = Column(Integer, default=0)

    # Input
    primary_language = Column(String(20))
    secondary_language = Column(String(20))
    target_language = Column(String(20))
    strategy = Column(String(20))
    primary_code = Column(Text)
    secondary_code = Column(Text)

    # Output
    fused_code = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    test_cases = Column(Text, nullable=True)
    agent_traces = Column(JSON, default=list)
    warnings = Column(JSON, default=list)

    # Metrics
    cosine_similarity = Column(Float, nullable=True)
    structural_overlap = Column(Float, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)

    # Quality
    quality_score = Column(Float, nullable=True)
    complexity = Column(Float, nullable=True)

    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CodeIndex(Base):
    __tablename__ = "code_index"

    id = Column(String(36), primary_key=True)
    code = Column(Text, nullable=False)
    language = Column(String(20))
    description = Column(Text, nullable=True)
    namespace = Column(String(100), default="default")
    faiss_id = Column(Integer, nullable=True)
    code_metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemStat(Base):
    __tablename__ = "system_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event = Column(String(50))
    value = Column(Float)
    stat_metadata = Column("metadata", JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─── Init ─────────────────────────────────────────────────────────────────────

async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
