"""
GDELT BigQuery Helper - Access Historical Data (2005-2017)
============================================================
This script provides SQL queries and instructions for accessing
GDELT historical data via Google BigQuery.

Google BigQuery offers:
- Free tier: 1 TB queries per month
- Historical GDELT data from 1979 to present
- SQL interface for complex queries
"""

def get_bigquery_query(start_date: str, end_date: str, keywords: list) -> str:
    """
    Generate a BigQuery SQL query for GDELT historical data
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        keywords: List of keywords to search for
    
    Returns:
        SQL query string
    """
    # Build keyword conditions
    keyword_conditions = " OR ".join([
        f"LOWER(V2DocumentIdentifier) LIKE '%{kw.lower()}%'"
        for kw in keywords
    ])
    
    query = f"""
-- GDELT BigQuery Query for Historical Data
-- Date range: {start_date} to {end_date}
-- Keywords: {', '.join(keywords)}

SELECT
    DATE(_PARTITIONTIME) as date,
    V2DocumentIdentifier as url,
    V2Tone as tone,
    V2Themes as themes,
    V2Locations as locations,
    V2Persons as persons,
    V2Organizations as organizations,
    V2GCAM as gcam,
    V2SharingImage as sharing_image,
    V2RelatedImages as related_images,
    V2SocialImageEmbeds as social_image_embeds,
    V2SocialVideoEmbeds as social_video_embeds,
    V2Quotations as quotations,
    V2AllNames as all_names,
    V2Amounts as amounts,
    V2TranslationInfo as translation_info,
    V2Extras as extras
FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    DATE(_PARTITIONTIME) BETWEEN '{start_date}' AND '{end_date}'
    AND ({keyword_conditions})
ORDER BY
    DATE(_PARTITIONTIME) DESC
LIMIT
    100000
"""
    return query


def get_simplified_query(start_date: str, end_date: str, keywords: list) -> str:
    """
    Generate a simplified BigQuery query (faster, less columns)
    """
    keyword_conditions = " OR ".join([
        f"LOWER(V2DocumentIdentifier) LIKE '%{kw.lower()}%'"
        for kw in keywords
    ])
    
    query = f"""
-- Simplified query for stock market headlines
SELECT
    DATE(_PARTITIONTIME) as date,
    V2DocumentIdentifier as url,
    V2Tone as tone
FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    DATE(_PARTITIONTIME) BETWEEN '{start_date}' AND '{end_date}'
    AND ({keyword_conditions})
ORDER BY
    DATE(_PARTITIONTIME) DESC
LIMIT
    100000
"""
    return query


def print_instructions():
    """Print instructions for using BigQuery"""
    print("=" * 70)
    print("GDELT BigQuery Access Instructions")
    print("=" * 70)
    print("""
1. SETUP:
   - Go to: https://console.cloud.google.com/
   - Create a new project (or use existing)
   - Enable BigQuery API
   - Set up billing (free tier: 1 TB queries/month)

2. ACCESS BIGQUERY:
   - Go to: https://console.cloud.google.com/bigquery
   - Or use Python client:
     pip install google-cloud-bigquery
     pip install google-cloud-bigquery-storage

3. RUN QUERY:
   - Copy the SQL query below
   - Paste into BigQuery console
   - Click "Run"
   - Export results to CSV/JSON

4. PYTHON CLIENT EXAMPLE:
   ```python
   from google.cloud import bigquery
   
   client = bigquery.Client(project='your-project-id')
   query = \"\"\"[SQL QUERY HERE]\"\"\"
   
   df = client.query(query).to_dataframe()
   df.to_csv('gdelt_historical.csv', index=False)
   ```

5. COST ESTIMATION:
   - Free tier: 1 TB queries/month
   - Typical query: ~100-500 MB
   - Can run ~2000-10000 queries/month on free tier

6. DATA STRUCTURE:
   - V2DocumentIdentifier: Article URL
   - V2Tone: Sentiment score
   - V2Themes: Topics/themes
   - V2Locations: Geographic locations
   - V2Persons: People mentioned
   - V2Organizations: Organizations mentioned
    """)
    print("=" * 70)


def main():
    """Generate queries for historical stock market data"""
    print_instructions()
    
    # Stock market keywords
    keywords = [
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
    
    # Generate queries for 2005-2016 (before API limit)
    print("\n" + "=" * 70)
    print("Generated SQL Queries")
    print("=" * 70)
    
    print("\n--- Full Query (all columns) ---")
    print(get_bigquery_query("2005-01-01", "2016-12-31", keywords))
    
    print("\n--- Simplified Query (faster, fewer columns) ---")
    print(get_simplified_query("2005-01-01", "2016-12-31", keywords))
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("1. Copy one of the queries above")
    print("2. Go to BigQuery console: https://console.cloud.google.com/bigquery")
    print("3. Paste and run the query")
    print("4. Export results to CSV")
    print("5. Merge with data from data.py (2017+)")
    print("=" * 70)


if __name__ == "__main__":
    main()

