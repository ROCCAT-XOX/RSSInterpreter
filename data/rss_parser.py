# data/rss_parser.py
import feedparser

def parse_rss_feed(url):
    """Lädt und parst einen RSS-Feed von der angegebenen URL."""
    feed = feedparser.parse(url)
    return feed.entries
