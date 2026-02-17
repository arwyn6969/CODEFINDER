"""Legacy compatibility entrypoint.

Canonical app lives in app.api.main.
"""
from app.api.main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000)
