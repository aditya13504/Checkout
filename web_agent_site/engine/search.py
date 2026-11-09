"""
BM25 product search on SQLite FTS5.

This is a Java-free alternative to the Lucene index built with Pyserini. It
indexes the same document text and exposes the small part of the
`LuceneSearcher` interface that the engine uses (`search(...)` returning hits
with a `docid`).

Differences from Lucene, which change the ranking slightly: FTS5 fixes BM25 at
k1=1.2, b=0.75 (Pyserini defaults to k1=0.9, b=0.4), and its Porter tokenizer is
close to, but not the same as, Lucene's English analyzer.
"""
import os
import re
import sqlite3
from collections import namedtuple

Hit = namedtuple('Hit', ['docid'])

# Lucene's default English stop words.
STOP_WORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such that the
their then there these they this to was will with
""".split())

TOKEN_RE = re.compile(r'[a-z0-9]+')
CACHE_LIMIT = 200_000


def build_match_query(query):
    """Turns free text into an FTS5 OR query (the Lucene default operator)."""
    tokens = []
    for token in TOKEN_RE.findall(query.lower()):
        if token not in STOP_WORDS and token not in tokens:
            tokens.append(token)
    return ' OR '.join(f'"{token}"' for token in tokens)


class SqliteSearcher:
    """Searches an index made by `search_engine/build_sqlite_index.py`."""

    def __init__(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f'No search index at {index_path}. '
                'Run search_engine/build_sqlite_index.py first.'
            )
        self.index_path = index_path
        self._conn = None
        self._conn_pid = None
        self._cache = {}

    @property
    def conn(self):
        # SQLite connections must not cross a fork, so each process opens its own.
        if self._conn is None or self._conn_pid != os.getpid():
            uri = f'file:{self.index_path}?mode=ro'
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            self._conn_pid = os.getpid()
        return self._conn

    def search(self, query, k=10):
        key = (query, k)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        match = build_match_query(query)
        if not match:
            hits = []
        else:
            rows = self.conn.execute(
                'SELECT m.asin FROM docs JOIN doc_map m ON m.rowid = docs.rowid '
                'WHERE docs MATCH ? ORDER BY bm25(docs), docs.rowid LIMIT ?',
                (match, k),
            ).fetchall()
            hits = [Hit(asin) for (asin,) in rows]

        if len(self._cache) >= CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = hits
        return hits
