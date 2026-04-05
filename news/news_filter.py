"""
news/news_filter.py

Scrapes Forex Factory for HIGH impact news events.
Blocks trades for affected currency pairs 15 minutes before/after each event.

Rules:
  - Only HIGH impact (red) events are used
  - News is ONLY a blocker, never a trade trigger
  - Refreshes every 30 minutes
  - Fail-safe: if scraping fails, trading continues (only logs warning)
"""

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config.settings import Settings

logger = logging.getLogger(__name__)

# Forex Factory news URL
_FF_URL = "https://www.forexfactory.com/calendar?day=today"

# Headers to avoid being blocked
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Currencies affected by major USD news that also impacts indices
_USD_ADJACENT = {"BTCUSD", "ETHUSD", "US100", "US500", "US30"}

# Symbol → currencies it's sensitive to
_SYMBOL_CURRENCIES: dict[str, list[str]] = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "EURGBP": ["EUR", "GBP"],
    "US100":  ["USD"],
    "US500":  ["USD"],
    "US30":   ["USD"],
    "DE40":   ["EUR"],
    "UK100":  ["GBP"],
    "JP225":  ["JPY"],
    "BTCUSD": ["USD"],
    "ETHUSD": ["USD"],
    "BNBUSD": ["USD"],
    "XRPUSD": ["USD"],
    "SOLUSD": ["USD"],
    "ADAUSD": ["USD"],
    "AAPL":   ["USD"],
    "MSFT":   ["USD"],
    "NVDA":   ["USD"],
    "TSLA":   ["USD"],
    "AMZN":   ["USD"],
    "META":   ["USD"],
    "GOOGL":  ["USD"],
    "NFLX":   ["USD"],
}


class NewsEvent:
    def __init__(self, currency: str, event_time: datetime, event_name: str):
        self.currency   = currency
        self.event_time = event_time
        self.event_name = event_name

    def __repr__(self):
        return f"NewsEvent({self.currency} @ {self.event_time.strftime('%H:%M')} | {self.event_name})"


class NewsFilter:
    """
    Scrapes and caches Forex Factory HIGH impact news.
    Thread-safe for concurrent symbol scanning.
    """

    def __init__(
        self,
        block_before_min: int = Settings.NEWS_BLOCK_BEFORE_MIN,
        block_after_min:  int = Settings.NEWS_BLOCK_AFTER_MIN,
        refresh_interval_min: int = Settings.NEWS_REFRESH_INTERVAL_MIN,
    ):
        self.block_before  = timedelta(minutes=block_before_min)
        self.block_after   = timedelta(minutes=block_after_min)
        self.refresh_every = timedelta(minutes=refresh_interval_min)

        self._events:       list[NewsEvent] = []
        self._last_refresh: Optional[datetime] = None
        self._lock = threading.Lock()

    def refresh(self, force: bool = False) -> None:
        """
        Fetch and parse today's HIGH impact events from Forex Factory.
        Called automatically if cache is stale. Thread-safe.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            if not force and self._last_refresh is not None:
                if now - self._last_refresh < self.refresh_every:
                    return  # still fresh

            logger.info("[News] Refreshing Forex Factory calendar...")
            try:
                events = self._scrape()
                self._events = events
                self._last_refresh = now
                logger.info(f"[News] {len(events)} HIGH impact events loaded today.")
                for e in events:
                    logger.info(f"  → {e}")
            except Exception as ex:
                logger.warning(f"[News] Scrape failed (trading continues): {ex}")
                # Fail-safe: keep old events

    def is_blocked(self, symbol: str, now: Optional[datetime] = None) -> bool:
        """
        Returns True if symbol should be blocked due to nearby high-impact news.

        Args:
            symbol: e.g. "EURUSD"
            now:    current UTC datetime (defaults to datetime.utcnow)
        """
        self.refresh()  # refresh if stale

        if now is None:
            now = datetime.now(timezone.utc)
        elif now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        symbol_currencies = _SYMBOL_CURRENCIES.get(symbol, [])
        if not symbol_currencies:
            return False

        with self._lock:
            for event in self._events:
                if event.currency not in symbol_currencies:
                    continue
                window_start = event.event_time - self.block_before
                window_end   = event.event_time + self.block_after
                if window_start <= now <= window_end:
                    logger.info(
                        f"[News] {symbol} BLOCKED: {event.event_name} "
                        f"@ {event.event_time.strftime('%H:%M')} UTC"
                    )
                    return True

        return False

    def get_events(self) -> list[NewsEvent]:
        with self._lock:
            return list(self._events)

    # ── Internal scraping ────────────────────────────────────────────────────

    def _scrape(self) -> list[NewsEvent]:
        """Parse Forex Factory calendar HTML for today's HIGH impact events."""
        resp = requests.get(_FF_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        events: list[NewsEvent] = []
        today = datetime.now(timezone.utc).date()
        last_time: Optional[datetime] = None

        table = soup.find("table", class_="calendar__table")
        if table is None:
            logger.warning("[News] Could not find calendar table on Forex Factory.")
            return []

        for row in table.find_all("tr", class_="calendar__row"):
            # Impact indicator
            impact_td = row.find("td", class_="calendar__impact")
            if impact_td is None:
                continue

            impact_icon = impact_td.find("span")
            if impact_icon is None:
                continue

            icon_class = impact_icon.get("class", [])
            is_high = any("high" in c.lower() for c in icon_class)
            if not is_high:
                continue

            # Time cell
            time_td = row.find("td", class_="calendar__time")
            time_str = time_td.get_text(strip=True) if time_td else ""

            if time_str and time_str.lower() not in ("", "all day", "tentative"):
                try:
                    t = datetime.strptime(time_str, "%I:%M%p")
                    event_time = datetime(
                        today.year, today.month, today.day,
                        t.hour, t.minute, tzinfo=timezone.utc
                    )
                    last_time = event_time
                except ValueError:
                    event_time = last_time
            else:
                event_time = last_time

            if event_time is None:
                continue

            # Currency
            currency_td = row.find("td", class_="calendar__currency")
            currency = currency_td.get_text(strip=True) if currency_td else ""
            if not currency:
                continue

            # Event name
            event_td = row.find("td", class_="calendar__event")
            event_name = event_td.get_text(strip=True) if event_td else "Unknown"

            events.append(NewsEvent(
                currency=currency.upper(),
                event_time=event_time,
                event_name=event_name,
            ))

        return events
