from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(
    str(settings.database_url),
    echo=False,
    connect_args={"timeout": 15},   # SQLite busy timeout in seconds
)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Migrate existing tables: add columns that may not exist in older DBs
    _NEW_COLS = [
        ("slice_results", "liver_perimeter_px", "INTEGER DEFAULT 0"),
        ("slice_results", "tumor_perimeter_px",  "INTEGER DEFAULT 0"),
    ]
    async with engine.begin() as conn:
        for table, col, coldef in _NEW_COLS:
            try:
                await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {coldef}"))
            except Exception:
                pass  # column already exists
