"""
Enhanced GDELT Data Fetcher with Alternative Strategies for Historical Data
===========================================================================
This script uses multiple strategies to fetch stock market headlines:
1. GDELT API v2 (for 2017+)
2. NewsAPI (if API key available, for historical data)
3. GDELT Raw Data Files (instructions provided)
4. Google BigQuery (instructions provided)
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os

# API Configuration
GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = "gdelt_stock_headlines_historical.csv"
OUTPUT_JSON = "gdelt_stock_headlines_historical.json"

# Stock market related queries
STOCK_QUERIES = [
    "stock market",
    "stock exchange",
    "NASDAQ",
    "Dow Jones",
    "stock price",
    "market crash",
    "market rally",
    "bull market",
    "bear market",
    "stock trading",
    "market volatility",
    "stock index",
    "equity market",
    "financial market"
]

# Date range: Last 20 years
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=20*365)


def format_date_for_gdelt(date: datetime) -> str:
    """Format datetime to GDELT API format (YYYYMMDDHHMMSS)"""
    return date.strftime("%Y%m%d%H%M%S")


def format_date_for_newsapi(date: datetime) -> str:
    """Format datetime to NewsAPI format (YYYY-MM-DD)"""
    return date.strftime("%Y-%m-%d")


def fetch_gdelt_data(
    query: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_records: int = 250
) -> List[Dict]:
    """Fetch data from GDELT Project API"""
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records
    }
    
    if start_date and end_date:
        params["startdatetime"] = format_date_for_gdelt(start_date)
        params["enddatetime"] = format_date_for_gdelt(end_date)
    
    try:
        response = requests.get(GDELT_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        if not response.text or len(response.text.strip()) == 0:
            return []
        
        try:
            data = response.json()
        except json.JSONDecodeError:
            return []
        
        if "articles" in data:
            return data["articles"]
        return []
    except Exception as e:
        return []


def fetch_newsapi_data(
    query: str,
    start_date: datetime,
    end_date: datetime,
    api_key: Optional[str] = None,
    max_records: int = 100
) -> List[Dict]:
    """
    Fetch data from NewsAPI (requires API key)
    NewsAPI free tier: 100 requests/day, articles from last month
    Paid tier: Historical data available
    """
    if not api_key:
        return []
    
    params = {
        "q": query,
        "from": format_date_for_newsapi(start_date),
        "to": format_date_for_newsapi(end_date),
        "sortBy": "publishedAt",
        "pageSize": min(max_records, 100),  # NewsAPI max is 100 per request
        "apiKey": api_key
    }
    
    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "ok" and "articles" in data:
            # Convert NewsAPI format to GDELT-like format
            articles = []
            for article in data["articles"]:
                articles.append({
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "seendate": article.get("publishedAt", "").replace("T", "").replace("Z", "").replace(":", "").replace("-", "")[:14] + "Z",
                    "domain": article.get("source", {}).get("name", ""),
                    "language": "unknown",
                    "sourcecountry": "unknown"
                })
            return articles
    except Exception as e:
        print(f"    ⚠ NewsAPI error: {e}")
    
    return []


def fetch_gdelt_raw_data_instructions():
    """
    Generate instructions for accessing GDELT raw data files
    GDELT provides raw data files going back to 1979
    """
    instructions = """
    ======================================================================
    GDELT Raw Data Access Instructions
    ======================================================================
    
    GDELT API v2 has limitations for data before 2017. To get historical
    data (2005-2017), you can use GDELT's raw data files:
    
    1. GDELT Raw Data Files:
       - URL: https://www.gdeltproject.org/data.html
       - Files available from 1979 to present
       - Format: CSV files with event data
       - Download: Use wget or curl to download daily files
       
    2. Google BigQuery (Recommended):
       - GDELT data is available on Google BigQuery
       - Free tier: 1 TB queries/month
       - Can query historical data
       - Example query:
         SELECT * FROM `gdelt-bq.gdeltv2.gkg_partitioned`
         WHERE DATE(_PARTITIONTIME) BETWEEN '2005-01-01' AND '2017-12-31'
         AND (LOWER(V2DocumentIdentifier) LIKE '%stock market%'
              OR LOWER(V2DocumentIdentifier) LIKE '%NASDAQ%')
       - More info: https://www.gdeltproject.org/data.html#googlebigquery
    
    3. GDELT Analysis Service:
       - Web interface: https://analysis.gdeltproject.org/
       - Can export data for specific date ranges
       - User-friendly but limited export sizes
    
    ======================================================================
    """
    return instructions


def fetch_with_fallback_strategies(
    queries: List[str],
    start_date: datetime,
    end_date: datetime,
    newsapi_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch data using multiple strategies with fallbacks
    """
    all_articles = []
    
    print("=" * 70)
    print("Multi-Strategy Historical Data Fetching")
    print("=" * 70)
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Total queries: {len(queries)}\n")
    
    # Strategy 1: GDELT API for 2017+
    gdelt_cutoff = datetime(2017, 1, 1)
    
    if end_date >= gdelt_cutoff:
        print("[Strategy 1] Fetching from GDELT API (2017+)...")
        gdelt_start = max(start_date, gdelt_cutoff)
        
        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] {query}")
            
            # Fetch in 1-year chunks
            current_start = gdelt_start
            chunk_num = 1
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=365), end_date)
                
                articles = fetch_gdelt_data(query, current_start, current_end, 250)
                if articles:
                    all_articles.extend(articles)
                    print(f"    Chunk {chunk_num}: {len(articles)} articles")
                
                current_start = current_end + timedelta(days=1)
                chunk_num += 1
                time.sleep(1)
        
        print(f"  ✓ GDELT API: {len([a for a in all_articles if 'seendate' in a])} articles\n")
    
    # Strategy 2: NewsAPI for historical data (if API key provided)
    if newsapi_key and start_date < gdelt_cutoff:
        print("[Strategy 2] Fetching from NewsAPI (historical)...")
        print("  Note: NewsAPI free tier only covers last month")
        print("  For older data, use paid tier or GDELT raw files\n")
        
        # NewsAPI free tier limitation - only last month
        newsapi_start = max(start_date, end_date - timedelta(days=30))
        
        if newsapi_start < end_date:
            for i, query in enumerate(queries[:5], 1):  # Limit queries for API limits
                print(f"  [{i}/5] {query}")
                articles = fetch_newsapi_data(query, newsapi_start, end_date, newsapi_key, 100)
                if articles:
                    all_articles.extend(articles)
                    print(f"    ✓ {len(articles)} articles")
                time.sleep(1)
    
    # Strategy 3: Instructions for raw data
    if start_date < gdelt_cutoff:
        print("[Strategy 3] For data before 2017, use GDELT raw data files")
        print(fetch_gdelt_raw_data_instructions())
    
    if not all_articles:
        return pd.DataFrame()
    
    # Process articles
    processed = []
    for article in all_articles:
        try:
            processed_article = {
                "seendate": article.get("seendate", ""),
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "domain": article.get("domain", ""),
                "language": article.get("language", "unknown"),
                "sourcecountry": article.get("sourcecountry", ""),
            }
            processed.append(processed_article)
        except:
            continue
    
    df = pd.DataFrame(processed)
    
    # Remove duplicates
    if not df.empty:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["url"], keep="first")
        if initial_count != len(df):
            print(f"  Removed {initial_count - len(df)} duplicates")
        
        # Sort by date
        if "seendate" in df.columns:
            df = df.sort_values("seendate", ascending=False)
    
    return df


def main():
    """Main function"""
    print("=" * 70)
    print("Enhanced GDELT Historical Data Fetcher")
    print("=" * 70)
    
    # Check for NewsAPI key in environment variable
    newsapi_key = os.environ.get("NEWSAPI_KEY", None)
    if not newsapi_key:
        print("\n⚠ NEWSAPI_KEY not found in environment variables")
        print("  To use NewsAPI for historical data:")
        print("  1. Get free API key from https://newsapi.org/")
        print("  2. Set environment variable: export NEWSAPI_KEY=your_key")
        print("  Note: Free tier only covers last month of data\n")
    
    # Fetch data
    df = fetch_with_fallback_strategies(
        queries=STOCK_QUERIES,
        start_date=START_DATE,
        end_date=END_DATE,
        newsapi_key=newsapi_key
    )
    
    if df.empty:
        print("\n⚠ No data retrieved. See instructions above for accessing historical data.")
        return
    
    # Save data
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    
    df_dict = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Saved JSON: {json_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total articles: {len(df)}")
    if "seendate" in df.columns:
        print(f"Date range: {df['seendate'].min()} to {df['seendate'].max()}")
    print(f"Unique domains: {df['domain'].nunique() if 'domain' in df.columns else 'N/A'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

