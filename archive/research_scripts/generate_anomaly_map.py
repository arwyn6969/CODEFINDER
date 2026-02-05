import csv
import sys
from pathlib import Path
from sonnet_print_block_scanner import SonnetPrintBlockScanner

class FastAnomalyMapper(SonnetPrintBlockScanner):
    """
    Subclass of the scanner that skips image saving for speed.
    Used to generate the coordinate map of anomalies.
    """
    def _save_anomaly_image(self, page_image, instance, page_num, char_label):
        # Override to do nothing, returning a dummy path
        return "skipped_for_map"

    def _save_character_image(self, page_image, instance, page_num, scale=3):
        # Override to do nothing
        return "skipped_for_map"

def generate_map():
    pdf_path = "data/sources/archive/SONNETS_QUARTO_1609_NET.pdf"
    output_dir = "reports/sonnet_print_block_analysis"
    
    print("Starting Fast Anomaly Mapping...")
    scanner = FastAnomalyMapper(pdf_path, output_dir)
    
    # Run scan without image saving
    # save_images=False in scan_all_pages prevents the main image save logic,
    # but our overrides ensure even internal calls don't touch disk.
    scanner.scan_all_pages(save_images=False)
    
    print(f"Scan complete. Detected {len(scanner.anomalies)} anomalies.")
    
    # Export to CSV
    csv_path = Path(output_dir) / "anomalies_map.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Page", "X", "Y", "Width", "Height", "Type", "Description", "Character"])
        
        for anomaly in scanner.anomalies:
            writer.writerow([
                anomaly.page_number,
                f"{anomaly.x:.2f}",
                f"{anomaly.y:.2f}",
                f"{anomaly.width:.2f}",
                f"{anomaly.height:.2f}",
                anomaly.anomaly_type,
                anomaly.description,
                anomaly.related_character
            ])
            
    print(f"Anomaly Map saved to: {csv_path}")

if __name__ == "__main__":
    generate_map()
