#!/usr/bin/env python3
"""A stand-in for the `substack-rag` service. Stdlib only.

Implements the documented routes — `GET /search`, `GET /doc/{slug}`,
`GET /stats`, `GET /health` — over a handful of fake passages, so the bridge in
`claw_py/rag.py` is exercised against real HTTP rather than a mock.

Point the agent at the real service instead with `--rag-url`.

    python examples/fake_rag_server.py 8000
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

CHUNKS = [
    {
        "slug": "a-new-oil-era",
        "title": "A New Oil Era",
        "author": "Alexander Campbell",
        "publication": "campbellramble",
        "date": "2026-04-29",
        "likes": 210,
        "text": (
            "The cartel's discipline is gone. Quota compliance has become "
            "optional, and the UAE in particular has spent two years building "
            "capacity it has no intention of leaving idle. What looks like a "
            "pricing dispute is really a capacity dispute."
        ),
    },
    {
        "slug": "a-new-oil-era",
        "title": "A New Oil Era",
        "author": "Alexander Campbell",
        "publication": "campbellramble",
        "date": "2026-04-29",
        "likes": 210,
        "text": (
            "Shipping through the Strait of Hormuz carries an insurance premium "
            "that no longer reflects the actual interdiction risk, which means "
            "the freight market is mispricing a tail it has already survived."
        ),
    },
    {
        "slug": "all-along-the-ai-watchtower",
        "title": "All Along the AI Watchtower",
        "author": "CitriniResearch",
        "publication": "citrini",
        "date": "2026-03-11",
        "likes": 480,
        "text": (
            "Leverage is the transmission mechanism. The unwind will not start "
            "in the names everyone is watching; it starts wherever the "
            "financing was cheapest, which today means the infrastructure "
            "layer rather than the model labs."
        ),
    },
    {
        "slug": "the-crowded-funds-problem",
        "title": "The Crowded Funds Problem",
        "author": "CitriniResearch",
        "publication": "citrini",
        "date": "2026-01-20",
        "likes": 155,
        "text": (
            "Position crowding is measurable before it is painful. When the "
            "same twelve names appear in the top ten of forty funds, the exit "
            "is priced for one seller and sized for forty."
        ),
    },
]

DOCS = {
    "a-new-oil-era": (
        "A New Oil Era\n\nThe cartel's discipline is gone... [full post text, "
        "overlap de-duplicated]"
    ),
    "all-along-the-ai-watchtower": (
        "All Along the AI Watchtower\n\nLeverage is the transmission "
        "mechanism... [full post text]"
    ),
    "the-crowded-funds-problem": (
        "The Crowded Funds Problem\n\nPosition crowding is measurable... "
        "[full post text]"
    ),
}


def score(chunk: dict, query: str) -> int:
    terms = {t for t in query.lower().split() if len(t) > 3}
    haystack = f"{chunk['text']} {chunk['title']}".lower()
    return sum(1 for term in terms if term in haystack)


def search(params: dict) -> list[dict]:
    query = params.get("q", [""])[0]
    k = int(params.get("k", ["5"])[0])
    author = params.get("author", [None])[0]
    publication = params.get("publication", [None])[0]
    date_from = params.get("date_from", [None])[0]
    date_to = params.get("date_to", [None])[0]
    min_likes = params.get("min_likes", [None])[0]

    results = []
    for chunk in CHUNKS:
        if author and author.lower() not in chunk["author"].lower():
            continue
        if publication and publication.lower() != chunk["publication"].lower():
            continue
        if date_from and chunk["date"] < date_from:
            continue
        if date_to and chunk["date"] > date_to:
            continue
        if min_likes and chunk["likes"] < int(min_likes):
            continue
        hit = dict(chunk)
        hit["score"] = score(chunk, query)
        hit["matched_by"] = "both" if hit["score"] > 1 else "fts"
        hit["citation"] = f"{chunk['author']} — {chunk['title']} · {chunk['date']}"
        results.append(hit)

    results.sort(key=lambda h: -h["score"])
    return [h for h in results if h["score"] > 0][:k]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass  # quiet

    def _send(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/health":
            self._send({"status": "ok"})
        elif parsed.path == "/stats":
            self._send({"documents": 3, "chunks": len(CHUNKS), "date_coverage": 1.0})
        elif parsed.path == "/search":
            self._send({"results": search(params)})
        elif parsed.path.startswith("/doc/"):
            slug = urllib.parse.unquote(parsed.path[len("/doc/"):])
            if slug not in DOCS:
                self._send({"detail": "not found"}, 404)
            else:
                sample = next(c for c in CHUNKS if c["slug"] == slug)
                self._send({
                    "slug": slug,
                    "title": sample["title"],
                    "author": sample["author"],
                    "date": sample["date"],
                    "text": DOCS[slug],
                })
        else:
            self._send({"detail": "not found"}, 404)


def serve(port: int = 8000) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"fake rag service on http://127.0.0.1:{port}", file=sys.stderr)
    serve(port).serve_forever()
