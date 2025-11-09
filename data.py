"""
GDELT Project API - Stock Market Headlines Data Fetcher
========================================================
Fetches headline data related to stock market from GDELT Project API
for the last 20 years (2004-2024).
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os

# API Configuration
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = "gdelt_stock_headlines.csv"
OUTPUT_JSON = "gdelt_stock_headlines.json"

# Stock market related queries
# Note: "S&P 500" needs to be quoted in GDELT API, using "SP 500" or "S and P 500" instead
STOCK_QUERIES = [
    "stock market",
    "stock exchange",
    '"S&P 500"',  # Quoted to handle special characters
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
# Note: GDELT API may have limits on how far back data is available
# Adjust START_DATE if API rejects very old dates
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=20*365)  # Approximately 20 years

# Alternative: Use a more recent start date if API has limits
# START_DATE = datetime(2015, 1, 1)  # Start from 2015 if needed


def format_date_for_api(date: datetime) -> str:
    """Format datetime to GDELT API format (YYYYMMDDHHMMSS)"""
    return date.strftime("%Y%m%d%H%M%S")


def fetch_gdelt_data(
    query: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_records: int = 250,
    mode: str = "artlist",
    format_type: str = "json"
) -> Optional[List[Dict]]:
    """
    Fetch data from GDELT Project API
    
    Args:
        query: Search query string
        start_date: Start datetime
        end_date: End datetime
        max_records: Maximum number of records to fetch
        mode: API mode (artlist for article list)
        format_type: Response format (json)
    
    Returns:
        List of article dictionaries or None if error
    """
    params = {
        "query": query,
        "mode": mode,
        "format": format_type,
        "maxrecords": max_records
    }
    
    # Add date parameters only if provided
    if start_date and end_date:
        params["startdatetime"] = format_date_for_api(start_date)
        params["enddatetime"] = format_date_for_api(end_date)
        date_range = f" ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    else:
        date_range = " (no date filter)"
    
    try:
        print(f"  Fetching: '{query}'{date_range}...")
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        # Check if response is empty
        if not response.text or len(response.text.strip()) == 0:
            print(f"    ⚠ Empty response from API")
            return []
        
        # Try to parse JSON
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON decode error: {e}")
            print(f"    Response preview: {response.text[:200]}")
            return []
        
        # GDELT API returns data in "articles" key (matching working example)
        if "articles" in data:
            articles = data["articles"]
        elif isinstance(data, list):
            articles = data
        elif "docs" in data:
            articles = data["docs"]
        else:
            # Try to extract articles from response
            articles = data.get("results", [])
        
        if articles and len(articles) > 0:
            print(f"    ✓ Retrieved {len(articles)} articles")
            return articles
        else:
            print(f"    ⚠ No articles found in response")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"    ✗ Error fetching data: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"    Response status: {e.response.status_code}")
            print(f"    Response text: {e.response.text[:200]}")
        return []
    except json.JSONDecodeError as e:
        print(f"    ✗ Error parsing JSON: {e}")
        print(f"    Response text: {response.text[:200] if 'response' in locals() else 'N/A'}")
        return []
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        return []


def process_articles(articles: List[Dict]) -> pd.DataFrame:
    """
    Process raw article data into a structured DataFrame
    Matches the structure from the working example: seendate, title, url, etc.
    
    Args:
        articles: List of article dictionaries
    
    Returns:
        DataFrame with processed articles
    """
    processed = []
    
    for article in articles:
        try:
            # Extract fields matching the working example structure
            processed_article = {
                "seendate": article.get("seendate", article.get("date", article.get("datetime", ""))),
                "title": article.get("title", article.get("snippet", "")),
                "url": article.get("url", article.get("sourceurl", "")),
                "url_mobile": article.get("url_mobile", ""),
                "domain": article.get("domain", ""),
                "language": article.get("language", "unknown"),
                "sourcecountry": article.get("sourcecountry", ""),
                "socialimage": article.get("socialimage", ""),
            }
            
            # Try to extract additional fields
            if "snippet" in article:
                processed_article["snippet"] = article["snippet"]
            elif "text" in article:
                processed_article["snippet"] = article["text"][:500]  # First 500 chars
            
            processed.append(processed_article)
            
        except Exception as e:
            print(f"    ⚠ Error processing article: {e}")
            continue
    
    return pd.DataFrame(processed)


def fetch_all_stock_headlines(
    queries: List[str],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    max_records_per_query: int = 250,
    use_date_filter: bool = True,
    chunk_years: int = 1
) -> pd.DataFrame:
    """
    Fetch headlines for all stock market queries
    
    Args:
        queries: List of search queries
        start_date: Start datetime
        end_date: End datetime
        max_records_per_query: Max records per query
        use_date_filter: Whether to use date filtering
        chunk_years: Number of years per chunk (for large date ranges)
    
    Returns:
        Combined DataFrame with all headlines
    """
    all_articles = []
    
    print(f"\nFetching stock market headlines from GDELT Project API")
    if start_date and end_date:
        total_years = (end_date - start_date).days / 365.25
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_years:.1f} years)")
        if total_years > chunk_years:
            print(f"Breaking into {chunk_years}-year chunks to avoid API limits")
    else:
        print("Date range: No filter (fetching recent articles)")
    print(f"Total queries: {len(queries)}\n")
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] Query: '{query}'")
        
        # If using date filter and date range is large, break into chunks
        if use_date_filter and start_date and end_date:
            total_days = (end_date - start_date).days
            chunk_days = chunk_years * 365
            
            if total_days > chunk_days:
                # Fetch in chunks
                current_start = start_date
                chunk_num = 1
                
                while current_start < end_date:
                    current_end = min(current_start + timedelta(days=chunk_days), end_date)
                    print(f"  Chunk {chunk_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                    
                    articles = fetch_gdelt_data(
                        query=query,
                        start_date=current_start,
                        end_date=current_end,
                        max_records=max_records_per_query
                    )
                    
                    if articles:
                        all_articles.extend(articles)
                    
                    current_start = current_end + timedelta(days=1)
                    chunk_num += 1
                    time.sleep(1)  # Rate limiting between chunks
            else:
                # Single fetch for small date ranges
                articles = fetch_gdelt_data(
                    query=query,
                    start_date=start_date,
                    end_date=end_date,
                    max_records=max_records_per_query
                )
                
                if articles:
                    all_articles.extend(articles)
        else:
            # No date filter
            articles = fetch_gdelt_data(
                query=query,
                start_date=None,
                end_date=None,
                max_records=max_records_per_query
            )
            
            if articles:
                all_articles.extend(articles)
        
        # Rate limiting - be respectful to the API
        time.sleep(1)
    
    if not all_articles:
        print("\n⚠ No articles retrieved. Returning empty DataFrame.")
        return pd.DataFrame()
    
    print(f"\n✓ Total articles retrieved: {len(all_articles)}")
    print("Processing articles...")
    
    df = process_articles(all_articles)
    
    # Remove duplicates based on URL
    initial_count = len(df)
    df = df.drop_duplicates(subset=["url"], keep="first")
    duplicates_removed = initial_count - len(df)
    
    if duplicates_removed > 0:
        print(f"  Removed {duplicates_removed} duplicate articles")
    
    # Sort by date (using seendate field from working example)
    if "seendate" in df.columns:
        df = df.sort_values("seendate", ascending=False)
    elif "date" in df.columns:
        df = df.sort_values("date", ascending=False)
    
    print(f"✓ Final dataset: {len(df)} unique articles")
    
    return df


def save_data(df: pd.DataFrame, csv_path: str, json_path: str):
    """
    Save DataFrame to CSV and JSON files
    
    Args:
        df: DataFrame to save
        csv_path: Path for CSV file
        json_path: Path for JSON file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    
    # Save as CSV
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved CSV: {csv_path}")
    
    # Save as JSON
    df_dict = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Saved JSON: {json_path}")


def print_historical_data_instructions():
    """Print instructions for accessing historical data before 2017"""
    print("\n" + "=" * 70)
    print("Alternative Strategies for Historical Data (Before 2017)")
    print("=" * 70)
    print("""
The GDELT API v2 has limitations for data before 2017. Here are alternative
strategies to get historical stock market headlines:

1. GDELT Raw Data Files (Recommended):
   - URL: https://www.gdeltproject.org/data.html
   - Files available from 1979 to present
   - Download daily CSV files using wget/curl
   - Example: wget http://data.gdeltproject.org/gdeltv2/YYYYMMDD.gkg.csv.zip

2. Google BigQuery (Best for Large Datasets):
   - GDELT data is available on Google BigQuery (free tier: 1TB/month)
   - Can query historical data with SQL
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

4. Alternative APIs:
   - NewsAPI (requires API key, paid tier for historical data)
   - Alpha Vantage (financial news API)
   - See data_historical.py for implementation

5. Combine Sources:
   - Use GDELT API for 2017+ (current script)
   - Use BigQuery or raw files for 2005-2016
   - Merge datasets together
    """)
    print("=" * 70 + "\n")


def main():
    """Main function to fetch and save stock market headlines"""
    print("=" * 70)
    print("GDELT Project API - Stock Market Headlines Fetcher")
    print("=" * 70)
    
    # Check if we need historical data
    gdelt_cutoff = datetime(2017, 1, 1)
    needs_historical = START_DATE < gdelt_cutoff
    
    if needs_historical:
        print(f"\n⚠ Note: Requested start date ({START_DATE.strftime('%Y-%m-%d')}) is before")
        print(f"  GDELT API's historical limit (2017-01-01).")
        print(f"  Will fetch available data from 2017+ and provide instructions for older data.\n")
    
    # Fetch all headlines
    # Break 20-year range into 1-year chunks to avoid API date range limits
    print("Attempting to fetch data for last 20 years in 1-year chunks...")
    df = fetch_all_stock_headlines(
        queries=STOCK_QUERIES,
        start_date=START_DATE,
        end_date=END_DATE,
        max_records_per_query=250,
        use_date_filter=True,
        chunk_years=1  # Fetch 1 year at a time
    )
    
    # If no data retrieved with date filter, try without date filter as fallback
    if df.empty:
        print("\n⚠ No data retrieved with date filter. Trying without date filter (recent articles only)...")
        df = fetch_all_stock_headlines(
            queries=STOCK_QUERIES,
            start_date=None,
            end_date=None,
            max_records_per_query=250,
            use_date_filter=False
        )
    
    # Show instructions for historical data if needed
    if needs_historical:
        print_historical_data_instructions()
    
    if df.empty:
        print("\n⚠ No data to save. Exiting.")
        return
    
    # Save data
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON)
    
    save_data(df, csv_path, json_path)
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total articles: {len(df)}")
    
    # Use seendate if available (matching working example)
    date_col = "seendate" if "seendate" in df.columns else "date"
    if date_col in df.columns:
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    else:
        print("Date range: N/A")
    
    print(f"Unique domains: {df['domain'].nunique() if 'domain' in df.columns else 'N/A'}")
    print(f"Languages: {df['language'].value_counts().head(3).to_dict() if 'language' in df.columns else 'N/A'}")
    print("=" * 70)
    
    # Display sample (matching working example format)
    print("\nSample headlines (first 5):")
    print("-" * 70)
    if "title" in df.columns:
        for idx, row in df.head(5).iterrows():
            title = row.get("title", "N/A")[:70]
            date = row.get("seendate", row.get("date", "N/A"))
            print(f"[{date}] {title}")
    print("=" * 70)


if __name__ == "__main__":
    main()
