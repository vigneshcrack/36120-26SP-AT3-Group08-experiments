"""Kraken daily OHLCV acquisition, per CONTRACT.md section 3.

Shared across the group so training and serving use one code path. Implements
the contract's rules exactly:

  - GET https://api.kraken.com/0/public/OHLC?pair=<PAIR>&interval=1440
  - the pair -> result-key mapping in section 3
  - prices arrive as strings and are cast to float
  - THE LAST ROW IS THE CURRENT, INCOMPLETE DAY AND IS DROPPED before any
    feature is computed
  - responses cached per token for 15 minutes
  - on HTTP 429/5xx, retry twice with exponential back-off (1 s, 2 s), then
    raise so the caller can return the contract's 503
"""

from __future__ import annotations

import logging
import time
from typing import Final

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KRAKEN_OHLC_URL: Final = "https://api.kraken.com/0/public/OHLC"
DAILY_INTERVAL_MINUTES: Final = 1440

# CONTRACT.md section 1: exactly these strings, upper-case.
SUPPORTED_TOKENS: Final[dict[str, str]] = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "XRP": "XRPUSD",
    "SOL": "SOLUSD",
    "DOGE": "XDGUSD",
    "XLM": "XLMUSD",
    "TRX": "TRXUSD",
    "XMR": "XMRUSD",
}

# CONTRACT.md section 3: Kraken answers under its own canonical key, which is
# not always the pair that was requested (XBTUSD -> XXBTZUSD). The mapping is
# agreed in the contract, so it is matched explicitly rather than by taking
# whichever key happens to come first in the response.
RESULT_KEYS: Final[dict[str, str]] = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "XRP": "XXRPZUSD",
    "SOL": "SOLUSD",
    "DOGE": "XDGUSD",
    "XLM": "XXLMZUSD",
    "TRX": "TRXUSD",
    "XMR": "XXMRZUSD",
}

# CONTRACT.md section 3: cache and retry policy.
CACHE_SECONDS: Final = 15 * 60
MAX_RETRIES: Final = 2
BACKOFF_SECONDS: Final = (1.0, 2.0)
RETRYABLE_STATUS: Final = {429, 500, 502, 503, 504}

_RAW_COLUMNS: Final = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]


class KrakenError(RuntimeError):
    """Kraken was unreachable or returned an error after retries."""


class KrakenClient:
    """Client for Kraken's public daily OHLC endpoint."""

    def __init__(self, timeout: float = 15.0, session: requests.Session | None = None):
        self.timeout = timeout
        self._session = session or requests.Session()
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    def fetch_ohlcv(self, token: str, drop_incomplete: bool = True) -> pd.DataFrame:
        """Return a daily OHLCV frame for `token`, newest complete day last.

        Args:
            token: one of the eight tokens in CONTRACT.md section 1.
            drop_incomplete: drop the current, still-forming UTC day. Required
                by the contract before computing any feature: that candle's
                volume is only a part-day, which makes RelativeVolume
                (= Volume_t / VolumeSMA20) read as a volume collapse.

        Raises:
            KeyError: token is not supported.
            KrakenError: the request failed after the contract's retries.
        """
        token = token.upper()
        if token not in SUPPORTED_TOKENS:
            raise KeyError(token)

        cache_key = f"{token}:{drop_incomplete}"
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < CACHE_SECONDS:
            return cached[1].copy()

        payload = self._get_with_retries(token)
        frame = self._to_frame(payload, token, drop_incomplete)
        self._cache[cache_key] = (time.monotonic(), frame)
        return frame.copy()

    def _get_with_retries(self, token: str) -> dict:
        """GET the OHLC endpoint, retrying on 429/5xx per the contract."""
        pair = SUPPORTED_TOKENS[token]
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._session.get(
                    KRAKEN_OHLC_URL,
                    params={"pair": pair, "interval": DAILY_INTERVAL_MINUTES},
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUS:
                    raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise KrakenError(f"Kraken error for {token}: {payload['error']}")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < MAX_RETRIES:
                    delay = BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "Kraken request for %s failed (%s); retrying in %.0fs",
                        token,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        raise KrakenError(
            f"Kraken request for {token} failed after {MAX_RETRIES} retries: {last_error}"
        )

    @staticmethod
    def _to_frame(payload: dict, token: str, drop_incomplete: bool) -> pd.DataFrame:
        result = payload.get("result", {})
        # Prefer the contract's agreed result key; fall back to the first
        # non-cursor key only if Kraken renames a pair, so a rename degrades to
        # a warning instead of a KeyError.
        expected = RESULT_KEYS[token]
        candles = result.get(expected)
        if candles is None:
            candles = next((v for k, v in result.items() if k != "last"), None)
            if candles is not None:
                logger.warning(
                    "Kraken did not return the agreed result key %s for %s; using %s instead",
                    expected,
                    token,
                    next(k for k in result if k != "last"),
                )
        if not candles:
            raise KrakenError(f"Kraken returned no candles for {token}")

        frame = pd.DataFrame(candles, columns=_RAW_COLUMNS)
        frame["date"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.normalize()
        numeric = ["open", "high", "low", "close", "vwap", "volume"]
        frame[numeric] = frame[numeric].astype(float)
        frame["token"] = token

        # Sort BEFORE de-duplicating: `keep="last"` only means "the most recent
        # candle for that day" once the rows are in date order.
        frame = (
            frame.loc[:, ["date", "token", "open", "high", "low", "close", "volume"]]
            .sort_values("date")
            .drop_duplicates(subset="date", keep="last")
            .reset_index(drop=True)
        )

        if drop_incomplete:
            today = pd.Timestamp.now(tz="UTC").normalize()
            frame = frame.loc[frame["date"] < today].reset_index(drop=True)

        if frame.empty:
            raise KrakenError(f"Kraken returned no completed candles for {token}")

        logger.info(
            "Fetched %d complete daily candles for %s (%s .. %s)",
            len(frame),
            token,
            frame["date"].iloc[0].date(),
            frame["date"].iloc[-1].date(),
        )
        return frame

    def fetch_many(self, tokens: list[str], drop_incomplete: bool = True) -> pd.DataFrame:
        """Fetch several tokens and stack them into one long frame."""
        frames = []
        for token in tokens:
            try:
                frames.append(self.fetch_ohlcv(token, drop_incomplete=drop_incomplete))
            except (KeyError, KrakenError) as exc:
                logger.warning("Skipping %s: %s", token, exc)
        if not frames:
            raise KrakenError("No token could be fetched from Kraken")
        return pd.concat(frames, ignore_index=True)
