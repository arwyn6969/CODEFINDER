import csv
import collections
from pathlib import Path

def summarize_anomalies():
    csv_path = "reports/sonnet_print_block_analysis/anomalies_map.csv"
    report_path = "reports/sonnet_print_block_analysis/anomaly_precision_grid_report.txt"
    
    if not Path(csv_path).exists():
        print("CSV not found.")
        return

    pages = collections.defaultdict(int)
    types = collections.defaultdict(int)
    chars = collections.defaultdict(int)
    
    # Store coordinates to calculate bounding box/spread
    page_spreads = collections.defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        total_anomalies = 0
        for row in reader:
            total_anomalies += 1
            p = row['Page']
            pages[p] += 1
            types[row['Type']] += 1
            chars[row['Character']] += 1
            
            try:
                x = float(row['X'])
                y = float(row['Y'])
                page_spreads[p].append((x, y))
            except ValueError:
                pass

    with open(report_path, 'w') as f:
        f.write("# Anomaly Precision Grid Report\n")
        f.write("## Axiom of Intent Verification\n\n")
        f.write(f"Total Anomalies Mapped: {total_anomalies}\n")
        f.write(f"Coordinate Precision: Floating point (e.g., X=1204.00, Y=59.00)\n\n")
        
        f.write("## Anomaly Types (Entropy Signals)\n")
        for t, count in types.items():
            f.write(f"- {t}: {count}\n")
        f.write("\n")
        
        f.write("## Top 10 Pages by Anomaly Density\n")
        sorted_pages = sorted(pages.items(), key=lambda item: item[1], reverse=True)[:10]
        for p, count in sorted_pages:
            f.write(f"- Page {p}: {count} anomalies\n")
        f.write("\n")
        
        f.write("## Top 10 Affected Characters\n")
        sorted_chars = sorted(chars.items(), key=lambda item: item[1], reverse=True)[:10]
        for c, count in sorted_chars:
            f.write(f"- '{c}': {count}\n")
            
    print(f"Report report generated: {report_path}")

if __name__ == "__main__":
    summarize_anomalies()
