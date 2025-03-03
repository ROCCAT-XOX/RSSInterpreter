# data/__init__.py

# Importiere den RSS Parser, der im Projekt verwendet wird.
from .rss_parser import parse_rss_feed

__all__ = ["parse_rss_feed"]
