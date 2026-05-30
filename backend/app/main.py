from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import cases, export, inference, nifti


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="CT Segmentation System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(cases.router,      prefix=API_PREFIX)
app.include_router(nifti.router,      prefix=API_PREFIX)
app.include_router(inference.router,  prefix=API_PREFIX)
app.include_router(export.router,     prefix=API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok"}
