#!/usr/bin/env python3
"""Fetch X search results into the CSV shape used by this project."""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FIELDNAMES = [
    "created_at",
    "id_str",
    "full_text",
    "quote_count",
    "reply_count",
    "retweet_count",
    "favorite_count",
    "lang",
    "user_id_str",
    "conversation_id_str",
    "username",
    "tweet_url",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Xquik tweet search results as semicolon-delimited CSV.",
    )
    parser.add_argument("--query", required=True, help="X search query")
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows")
    parser.add_argument(
        "--query-type",
        choices=("Latest", "Top"),
        default="Latest",
        help="Xquik search ordering",
    )
    return parser.parse_args()


def request_page(
    *,
    base_url: str,
    api_key: Optional[str],
    query: str,
    query_type: str,
    limit: int,
    cursor: str,
) -> Dict[str, Any]:
    params = {
        "q": query,
        "queryType": query_type,
        "limit": str(min(limit, 200)),
    }
    if cursor:
        params["cursor"] = cursor

    url = f"{base_url.rstrip('/')}/x/tweets/search?{urlencode(params)}"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        message = f"Xquik request failed with HTTP {error.code}: {body}"
        raise RuntimeError(message) from error
    except URLError as error:
        raise RuntimeError(f"Xquik request failed: {error.reason}") from error

    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise RuntimeError("Xquik response was not a JSON object")
    return parsed


def format_created_at(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(timezone.utc).strftime("%a %b %d %H:%M:%S +0000 %Y")


def number(value: Any) -> int:
    return value if isinstance(value, int) else 0


def tweet_to_row(tweet: Dict[str, Any]) -> Dict[str, Any]:
    author = tweet.get("author")
    author_data = author if isinstance(author, dict) else {}
    tweet_id = str(tweet.get("id") or "")
    username = str(author_data.get("username") or "")
    return {
        "created_at": format_created_at(tweet.get("createdAt")),
        "id_str": tweet_id,
        "full_text": str(tweet.get("text") or ""),
        "quote_count": number(tweet.get("quoteCount")),
        "reply_count": number(tweet.get("replyCount")),
        "retweet_count": number(tweet.get("retweetCount")),
        "favorite_count": number(tweet.get("likeCount")),
        "lang": str(tweet.get("lang") or ""),
        "user_id_str": str(author_data.get("id") or ""),
        "conversation_id_str": str(tweet.get("conversationId") or tweet_id),
        "username": username,
        "tweet_url": (
            f"https://twitter.com/{username}/status/{tweet_id}"
            if username and tweet_id
            else ""
        ),
    }


def fetch_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base_url = os.environ.get("XQUIK_BASE_URL", "https://xquik.com/api/v1")
    api_key = os.environ.get("XQUIK_API_KEY")
    rows: List[Dict[str, Any]] = []
    cursor = ""

    while len(rows) < args.limit:
        remaining = args.limit - len(rows)
        page = request_page(
            base_url=base_url,
            api_key=api_key,
            query=args.query,
            query_type=args.query_type,
            limit=remaining,
            cursor=cursor,
        )
        tweets = page.get("tweets")
        if not isinstance(tweets, list) or not tweets:
            break
        rows.extend(tweet_to_row(tweet) for tweet in tweets if isinstance(tweet, dict))
        cursor_value = page.get("next_cursor") or page.get("nextCursor")
        has_next_page = bool(page.get("has_next_page") or page.get("hasNextPage"))
        if not has_next_page or not isinstance(cursor_value, str):
            break
        cursor = cursor_value

    return rows[: args.limit]


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise RuntimeError("--limit must be at least 1")

    rows = fetch_rows(args)
    with open(args.output, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} tweets to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
