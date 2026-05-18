from fastapi import FastAPI
from routers import fraud

app = FastAPI(title="GO르덕 사기 탐지 AI")

app.include_router(fraud.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
