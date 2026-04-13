from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Outputs" / "Presentation_Refined"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Premium Modern Color Palette (Aesthetics Refactoring)
PRIMARY = "#4F46E5"    # Indigo 600 - Deep, vibrant primary
SECONDARY = "#0F172A"  # Slate 900 - High contrast text
ACCENT = "#F59E0B"     # Amber 500 - Sophisticated highlight
DANGER = "#F43F5E"     # Rose 500  - Elegant alert
SUCCESS = "#10B981"    # Emerald 500 - Modern success
MUTED = "#64748B"      # Slate 500 - Refined neutral
LIGHT = "#E2E8F0"      # Slate 200 - Borders/Grid lines
BACKGROUND = "#FFFFFF" # Purity context
SURFACE = "#F8FAFC"    # Slate 50  - Elegant box backgrounds

MIN_STATE_ORDER_COUNT = 300
MIN_DELAYED_ORDERS = 50
