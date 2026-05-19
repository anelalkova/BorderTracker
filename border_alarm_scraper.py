"""
borderalarm_scraper.py
======================
Scrapes reported wait times from borderalarm.com for any crossing
and saves them to the `crowdsourced_waits` PostgreSQL table.

borderalarm.com uses a simple HTML structure that's easy to parse
without a headless browser.

Usage:
    python borderalarm_scraper.py --crossing bogorodica
    python borderalarm_scraper.py --crossing bogorodica --once
    python borderalarm_scraper.py --crossing bogorodica --interval 30

    # Run all crossings that have a borderalarm page
    python borderalarm_scraper.py --all

Options:
    --crossing   Crossing key (bogorodica, tabanovce, blace, ...)
    --once       Scrape once and exit (default: run every --interval minutes)
    --interval   Polling interval in minutes (default: 15)
    --all        Scrape all configured crossings in sequence
    --dry-run    Print scraped data without saving to DB

Requirements:
    pip install requests beautifulsoup4 psycopg2-binary
"""

import argparse
import time
import sys
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "border_crossing",
    "user":     "postgres",
    "password": "postgres",
}

# borderalarm.com page slugs for each crossing
BORDERALARM_SLUGS = {
    "bogorodica":  "bogorodica-evzoni",
    "tabanovce":   "tabanovce-presevo",
    "blace":       "blace-merdare",
    "medzitlija":  "medjitlija-niki",
    # deve_bair and kafasan may not have pages — add slugs if they do
}

BASE_URL = "https://borderalarm.com/bottlenecks/{slug}/"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# borderalarm shows times in local Macedonian time (CET/CEST = UTC+1/+2)
MK_TZ = ZoneInfo("Europe/Skopje")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_crossing_id(conn, crossing_name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (crossing_name,))
        row = cur.fetchone()
        return row[0] if row else None


def get_latest_report_time(conn, crossing_id: int) -> datetime | None:
    """Return the reported_at of the most recent row for this crossing."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT reported_at FROM crowdsourced_waits
            WHERE crossing_id = %s
            ORDER BY reported_at DESC
            LIMIT 1
        """, (crossing_id,))
        row = cur.fetchone()
        return row[0] if row else None


def save_reports(conn, crossing_id: int, reports: list[dict], dry_run: bool = False):
    """
    Insert new reports. Deduplicates by (crossing_id, reported_at, wait_minutes).
    Returns the number of rows inserted.
    """
    if not reports:
        return 0

    inserted = 0
    with conn.cursor() as cur:
        for r in reports:
            if dry_run:
                print(f"  [DRY RUN] {r}")
                continue
            cur.execute("""
                INSERT INTO crowdsourced_waits
                    (crossing_id, reported_at, wait_minutes, reported_by, source, raw_text)
                VALUES (%s, %s, %s, %s, 'borderalarm', %s)
                ON CONFLICT DO NOTHING
            """, (
                crossing_id,
                r["reported_at"],
                r["wait_minutes"],
                r.get("reported_by"),
                r.get("raw_text"),
            ))
            if cur.rowcount > 0:
                inserted += 1
    if not dry_run:
        conn.commit()
    return inserted

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def parse_wait_time(text: str) -> int | None:
    """
    Extract minutes from strings like "15 min", "1 h", "1 h 30 min", "45 min".
    Returns minutes as int, or None if unparseable.
    """
    import re
    text = text.strip().lower()
    hours   = re.search(r'(\d+)\s*h', text)
    minutes = re.search(r'(\d+)\s*min', text)
    total   = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return total if total > 0 else None


def parse_report_time(text: str) -> datetime | None:
    """
    Parse borderalarm date strings: "22.04.2026 18:53"
    Returns UTC-aware datetime.
    """
    text = text.strip()
    try:
        local_dt = datetime.strptime(text, "%d.%m.%Y %H:%M")
        mk_dt    = local_dt.replace(tzinfo=MK_TZ)
        return mk_dt.astimezone(timezone.utc)
    except ValueError:
        return None


def scrape_crossing(slug: str) -> list[dict]:
    """
    Fetch and parse borderalarm.com page for the given crossing slug.
    Returns a list of report dicts: {reported_at, wait_minutes, reported_by, raw_text}.
    """
    url = BASE_URL.format(slug=slug)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [SCRAPE ERROR] {url}: {e}")
        return []

    soup    = BeautifulSoup(resp.text, "html.parser")
    reports = []

    # borderalarm structure (as of Apr 2026):
    # Each report is a <div> or <li> containing:
    #   - wait time  (e.g. "15 min")
    #   - timestamp  (e.g. "22.04.2026 18:53")
    #   - reporter   (e.g. "by anonymous_")
    #
    # We look for elements with a class that contains "report" or "bottleneck-report"
    # and fall back to scanning all text if the structure changes.

    # Try structured selectors first
    items = (
        soup.select(".bottleneck-report")
        or soup.select(".report-item")
        or soup.select("article.report")
    )

    if items:
        for item in items:
            raw = item.get_text(separator=" ", strip=True)
            wait_text = ""
            time_text = ""
            reporter  = ""

            wait_el = item.select_one(".wait-time, .wait, .duration, strong")
            time_el = item.select_one(".report-time, .date, time")
            user_el = item.select_one(".reporter, .user, .by")

            wait_text = wait_el.get_text(strip=True) if wait_el else raw
            time_text = time_el.get_text(strip=True) if time_el else ""
            reporter  = user_el.get_text(strip=True).lstrip("by ") if user_el else ""

            wait_min  = parse_wait_time(wait_text)
            report_dt = parse_report_time(time_text)

            if wait_min and report_dt:
                reports.append({
                    "reported_at":  report_dt,
                    "wait_minutes": wait_min,
                    "reported_by":  reporter or None,
                    "raw_text":     raw[:500],
                })

    else:
        # Fallback: scan all text blocks for "min" + date patterns
        import re
        full_text = soup.get_text(separator="\n")
        lines     = [l.strip() for l in full_text.splitlines() if l.strip()]

        # Look for lines like: "15 min" followed by "22.04.2026 18:53" and "by anonymous_"
        i = 0
        while i < len(lines):
            wait_min = parse_wait_time(lines[i])
            if wait_min:
                report_dt = None
                reporter  = None
                # Look ahead up to 3 lines for a timestamp
                for j in range(i + 1, min(i + 4, len(lines))):
                    dt = parse_report_time(lines[j])
                    if dt:
                        report_dt = dt
                    if lines[j].startswith("by "):
                        reporter = lines[j][3:].strip()
                if report_dt:
                    raw = " | ".join(lines[max(0, i-1):i+4])
                    reports.append({
                        "reported_at":  report_dt,
                        "wait_minutes": wait_min,
                        "reported_by":  reporter,
                        "raw_text":     raw[:500],
                    })
            i += 1

    # Remove duplicates within the same scrape run
    seen   = set()
    unique = []
    for r in reports:
        key = (r["reported_at"], r["wait_minutes"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_crossing(crossing_name: str, conn, dry_run: bool = False, verbose: bool = True):
    slug = BORDERALARM_SLUGS.get(crossing_name)
    if not slug:
        print(f"  No borderalarm.com slug configured for '{crossing_name}'. Skipping.")
        return 0

    crossing_id = get_crossing_id(conn, crossing_name)
    if not crossing_id:
        print(f"  Crossing '{crossing_name}' not in DB. Run border_crossings.py first.")
        return 0

    latest = get_latest_report_time(conn, crossing_id)
    if verbose and latest:
        print(f"  Last stored report: {latest.strftime('%Y-%m-%d %H:%M UTC')}")

    url = BASE_URL.format(slug=slug)
    print(f"  Scraping {url} …")
    reports = scrape_crossing(slug)

    if not reports:
        print(f"  No reports found (page structure may have changed).")
        return 0

    # Only save reports newer than what we already have
    if latest:
        reports = [r for r in reports if r["reported_at"] > latest]

    print(f"  Found {len(reports)} new report(s).")
    if verbose:
        for r in reports:
            ts = r["reported_at"].strftime("%Y-%m-%d %H:%M UTC")
            print(f"    {ts}  {r['wait_minutes']} min  by={r.get('reported_by') or 'unknown'}")

    inserted = save_reports(conn, crossing_id, reports, dry_run=dry_run)
    print(f"  Inserted: {inserted}")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="borderalarm.com scraper for wait times")
    parser.add_argument("--crossing",  default=None,
                        choices=list(BORDERALARM_SLUGS.keys()),
                        help="Which crossing to scrape")
    parser.add_argument("--all",       action="store_true",
                        help="Scrape all configured crossings")
    parser.add_argument("--once",      action="store_true",
                        help="Scrape once and exit")
    parser.add_argument("--interval",  type=int, default=15,
                        help="Polling interval in minutes (default: 15)")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print scraped data without writing to DB")
    args = parser.parse_args()

    if not args.crossing and not args.all:
        parser.error("Specify --crossing <name> or --all")

    crossings_to_scrape = (
        list(BORDERALARM_SLUGS.keys()) if args.all
        else [args.crossing]
    )

    conn = psycopg2.connect(**DB_CONFIG)
    print(f"PostgreSQL connected.\n")

    def run_all():
        for name in crossings_to_scrape:
            print(f"\n[{name}]")
            try:
                run_crossing(name, conn, dry_run=args.dry_run)
            except Exception as e:
                print(f"  ERROR: {e}")

    run_all()

    if not args.once:
        interval_sec = args.interval * 60
        print(f"\nPolling every {args.interval} min. Ctrl-C to stop.\n")
        while True:
            time.sleep(interval_sec)
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{ts}] Polling …")
            run_all()

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()