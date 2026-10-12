"""
BharatFinanceEdu - Render App Entrypoint Alias
Ensures Render can launch via `uvicorn main:app` or `uvicorn main1:app`.
"""
from main1 import app

__all__ = ["app"]
