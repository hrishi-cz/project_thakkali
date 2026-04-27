"""Root package initialization for APEX framework."""

__version__ = "0.1.0"
__author__ = "Abhiram"

def create_app():
    """Factory function to create FastAPI application."""
    from .api.run_api import app
    return app

__all__ = ["create_app"]

