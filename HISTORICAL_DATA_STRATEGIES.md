# Strategies for Getting GDELT Historical Data (Before 2017)

## Problem
The GDELT API v2 has limitations and only provides data from **2017 onwards**. For data before 2017, you need alternative strategies.

## Solution Overview

### ✅ Strategy 1: Google BigQuery (Recommended)
**Best for:** Large-scale historical data queries

**Advantages:**
- Free tier: 1 TB queries per month
- Historical data from 1979 to present
- SQL interface for complex queries
- Fast and scalable

**How to use:**
1. Run `python gdelt_bigquery_helper.py` to generate SQL queries
2. Go to [Google BigQuery Console](https://console.cloud.google.com/bigquery)
3. Copy and run the generated query
4. Export results to CSV
5. Merge with data from `data.py` (2017+)

**Example Query:**
```sql
SELECT
    DATE(_PARTITIONTIME) as date,
    V2DocumentIdentifier as url,
    V2Tone as tone
FROM
    `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE
    DATE(_PARTITIONTIME) BETWEEN '2005-01-01' AND '2016-12-31'
    AND (LOWER(V2DocumentIdentifier) LIKE '%stock market%'
         OR LOWER(V2DocumentIdentifier) LIKE '%NASDAQ%')
ORDER BY DATE(_PARTITIONTIME) DESC
LIMIT 100000
```

---

### ✅ Strategy 2: GDELT Raw Data Files
**Best for:** Downloading specific date ranges

**Advantages:**
- Direct access to raw data
- Available from 1979 to present
- No API limits

**How to use:**
1. Visit: https://www.gdeltproject.org/data.html
2. Download daily files:
   ```bash
   # Example: Download files for 2005
   for date in {20050101..20051231}; do
       wget http://data.gdeltproject.org/gdeltv2/${date}.gkg.csv.zip
   done
   ```
3. Extract and process CSV files
4. Filter for stock market keywords

**File Format:**
- Daily files: `YYYYMMDD.gkg.csv.zip`
- Contains all articles for that day
- Need to filter by keywords after download

---

### ✅ Strategy 3: GDELT Analysis Service
**Best for:** Quick exploration and small exports

**Advantages:**
- Web-based interface
- No coding required
- Visual analysis tools

**How to use:**
1. Go to: https://analysis.gdeltproject.org/
2. Set date range: 2005-01-01 to 2016-12-31
3. Enter keywords: "stock market", "NASDAQ", etc.
4. Export results

**Limitations:**
- Limited export sizes
- Less flexible than BigQuery

---

### ✅ Strategy 4: Alternative APIs
**Best for:** Supplementing GDELT data

**Options:**

1. **NewsAPI**
   - Free tier: Last month only
   - Paid tier: Historical data available
   - See `data_historical.py` for implementation

2. **Alpha Vantage**
   - Financial news API
   - Limited historical coverage
   - Requires API key

3. **Yahoo Finance News**
   - Free, no API key needed
   - Historical financial news
   - Can be scraped (check ToS)

---

### ✅ Strategy 5: Combine Multiple Sources
**Best approach for complete dataset:**

1. **2017-2025:** Use `data.py` (GDELT API v2)
   ```bash
   python data.py
   ```

2. **2005-2016:** Use BigQuery or raw files
   ```bash
   python gdelt_bigquery_helper.py  # Generate queries
   # Then run queries in BigQuery console
   ```

3. **Merge datasets:**
   ```python
   import pandas as pd
   
   # Load both datasets
   df_recent = pd.read_csv('data/raw/gdelt_stock_headlines.csv')
   df_historical = pd.read_csv('gdelt_historical_bigquery.csv')
   
   # Standardize columns
   # Merge
   df_combined = pd.concat([df_historical, df_recent], ignore_index=True)
   df_combined.to_csv('gdelt_complete_2005_2025.csv', index=False)
   ```

---

## Quick Start Guide

### Option A: BigQuery (Recommended)
```bash
# 1. Generate SQL queries
python gdelt_bigquery_helper.py

# 2. Copy query to BigQuery console
# 3. Run query and export to CSV
# 4. Merge with recent data
```

### Option B: Raw Data Files
```bash
# 1. Download files for specific dates
wget http://data.gdeltproject.org/gdeltv2/20050101.gkg.csv.zip

# 2. Extract and process
unzip 20050101.gkg.csv.zip
# Filter for stock market keywords in CSV

# 3. Merge with recent data
```

### Option C: Use Enhanced Script
```bash
# Run enhanced script (includes NewsAPI option)
python data_historical.py
# Note: Requires NEWSAPI_KEY for historical data
```

---

## Cost Comparison

| Method | Cost | Data Range | Ease of Use |
|--------|------|------------|-------------|
| GDELT API v2 | Free | 2017+ | ⭐⭐⭐⭐⭐ |
| Google BigQuery | Free (1TB/month) | 1979+ | ⭐⭐⭐⭐ |
| Raw Data Files | Free | 1979+ | ⭐⭐⭐ |
| GDELT Analysis | Free | 1979+ | ⭐⭐⭐⭐⭐ |
| NewsAPI | Free/Paid | Last month/Historical | ⭐⭐⭐ |

---

## Recommended Workflow

1. **Use `data.py`** to get data from 2017-2025 (automatic)
2. **Use BigQuery** to get data from 2005-2016 (run `gdelt_bigquery_helper.py`)
3. **Merge datasets** using pandas
4. **Result:** Complete dataset from 2005-2025

---

## Files in This Project

- `data.py` - Main script for GDELT API (2017+)
- `data_historical.py` - Enhanced script with multiple strategies
- `gdelt_bigquery_helper.py` - Generates BigQuery SQL queries
- `HISTORICAL_DATA_STRATEGIES.md` - This file

---

## Need Help?

- GDELT Documentation: https://www.gdeltproject.org/
- BigQuery Docs: https://cloud.google.com/bigquery/docs
- GDELT Data Portal: https://www.gdeltproject.org/data.html

