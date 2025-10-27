"""Compatibility wrapper for the Dash app entry point."""
from dash_app import create_app

__all__ = ["create_app"]


if __name__ == "__main__":
    create_app().run_server(debug=True)
