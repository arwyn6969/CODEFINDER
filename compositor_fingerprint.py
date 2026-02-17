#!/usr/bin/env python3
"""
Compositor Fingerprinting Tool (Zero Dependency Version)

Analyzes the distribution of Long-s ('ſ') vs Short-s ('s') 
to identify compositor patterns.
"""

import sqlite3
import csv
from pathlib import Path

# Database path
DB_PATH = Path("data/codefinder.db")
OUTPUT_DIR = Path("reports/compositor_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_connection():
    return sqlite3.connect(DB_PATH)

def analyze_long_s_profile():
    """Analyze Long-s vs Short-s distribution per page."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Query counts per page
    cursor.execute("""
    SELECT 
        s.name as source,
        p.page_number,
        SUM(CASE WHEN ci.character = 'ſ' THEN 1 ELSE 0 END) as long_s_count,
        SUM(CASE WHEN ci.character = 's' THEN 1 ELSE 0 END) as short_s_count 
    FROM character_instances ci
    JOIN pages p ON ci.page_id = p.id
    JOIN sources s ON p.source_id = s.id
    WHERE ci.character IN ('ſ', 's')
    GROUP BY s.name, p.page_number
    ORDER BY p.page_number, s.name
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    print(f"{'Page':<6} {'Source':<10} {'Long-s':<8} {'Short-s':<8} {'Ratio':<8}")
    print("-" * 50)
    
    data = []
    
    for row in rows:
        source, page, long_s, short_s = row
        total = long_s + short_s
        ratio = long_s / total if total > 0 else 0
        
        print(f"{page:<6} {source:<10} {long_s:<8} {short_s:<8} {ratio:.2f}")
        
        data.append({
            'page': page,
            'source': source,
            'long_s': long_s,
            'short_s': short_s,
            'ratio': ratio
        })
        
    # Save CSV
    csv_path = OUTPUT_DIR / "long_s_profile.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['page', 'source', 'long_s', 'short_s', 'ratio'])
        writer.writeheader()
        writer.writerows(data)
        
    print(f"\nSaved profile data to {csv_path}")
    return data

def analyze_outliers(data):
    """Find pages with biggest difference between Wright and Aspley."""
    # Organize by page
    page_map = {}
    for item in data:
        if item['page'] not in page_map:
            page_map[item['page']] = {}
        page_map[item['page']][item['source']] = item['ratio']
        
    print("\nTop Divergences (Wright vs Aspley):")
    print(f"{'Page':<6} {'Wright':<8} {'Aspley':<8} {'Delta':<8}")
    print("-" * 40)
    
    diffs = []
    for page, sources in page_map.items():
        if 'wright' in sources and 'aspley' in sources:
            w = sources['wright']
            a = sources['aspley']
            delta = abs(w - a)
            diffs.append((delta, page, w, a))
            
    diffs.sort(reverse=True)
    
    for delta, page, w, a in diffs[:10]:
        print(f"{page:<6} {w:.2f}     {a:.2f}     {delta:.2f}")

def main():
    print("="*60)
    print("COMPOSITOR FINGERPRINTING ANALYSIS")
    print("="*60)
    
    try:
        data = analyze_long_s_profile()
        if not data:
            print("No data found!")
            return
            
        analyze_outliers(data)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
