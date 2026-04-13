import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
import urllib.request
import os

# 한글 폰트 설정 (macOS 기준 AppleGothic)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# Load world map from reliable geojson URL
url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
world = gpd.read_file(url)

# Define ROIs (Primary + Secondary Candidates)
rois = [
    # Primary (Red, Star)
    {"name": "Natanz", "lat": 33.50, "lon": 51.93, "type": "주요 타겟 (Primary)", "color": "red", "marker": "*", "size": 400},
    {"name": "Isfahan", "lat": 32.66, "lon": 51.67, "type": "주요 타겟 (Primary)", "color": "red", "marker": "*", "size": 400},
    {"name": "Bushehr", "lat": 28.97, "lon": 50.84, "type": "주요 타겟 (Primary)", "color": "red", "marker": "*", "size": 400},
    {"name": "Dimona", "lat": 31.07, "lon": 35.03, "type": "주요 타겟 (Primary)", "color": "red", "marker": "*", "size": 400},
    
    # Secondary (Candidates)
    {"name": "Kharg Island", "lat": 29.21, "lon": 50.31, "type": "관심 구역 후보 (Candidate)", "color": "orange", "marker": "o", "size": 180},
    {"name": "Ras Laffan", "lat": 25.91, "lon": 51.52, "type": "관심 구역 후보 (Candidate)", "color": "orange", "marker": "o", "size": 180},
    {"name": "Minab", "lat": 27.14, "lon": 57.07, "type": "관심 구역 후보 (Candidate)", "color": "orange", "marker": "o", "size": 180},
    {"name": "Beirut", "lat": 33.89, "lon": 35.51, "type": "관심 구역 후보 (Candidate)", "color": "orange", "marker": "o", "size": 180},
]

df = pd.DataFrame(rois)
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Plotting the map - Modern Dark Aesthetic (Antigravity Style)
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_aspect('equal')
fig.patch.set_facecolor('#0B0F19')
ax.set_facecolor('#0B0F19') # Deep dark background

# Bounds for Middle East focus
ax.set_xlim([30, 65])
ax.set_ylim([22, 43])

# Draw land and water
world.plot(ax=ax, color='#1E293B', edgecolor='#334155', linewidth=1.0)

# Defined Colors
PRIMARY_COLOR = '#FF2A54'   # Neon Cherry
SECONDARY_COLOR = '#00D2FF' # Electric Cyan

# 1. Plot Primary targets with glow effect
primary = gdf[gdf['type'] == '주요 타겟 (Primary)']
# Glow layers for Primary
for alpha, size in [(0.05, 2000), (0.1, 1000), (0.25, 400)]:
    primary.plot(ax=ax, marker='o', color=PRIMARY_COLOR, markersize=size, alpha=alpha, zorder=3)
# Core marker for Primary
primary.plot(ax=ax, marker='o', color='white', edgecolor=PRIMARY_COLOR, linewidth=2.5, markersize=140, label='주요 타겟 (Primary)', zorder=5)

# 2. Plot Candidates with glow effect
candidates = gdf[gdf['type'] == '관심 구역 후보 (Candidate)']
# Glow layers for Candidates
for alpha, size in [(0.05, 800), (0.2, 300)]:
    candidates.plot(ax=ax, marker='o', color=SECONDARY_COLOR, markersize=size, alpha=alpha, zorder=3)
# Core marker for Candidates
candidates.plot(ax=ax, marker='o', color='white', edgecolor=SECONDARY_COLOR, linewidth=2, markersize=80, label='관심 구역 후보 (Candidate)', zorder=4)

# Labels with Glassmorphism-like dark boxes
for idx, row in gdf.iterrows():
    # Offset labels slightly based on latitude
    y_offset = 0.5 if row['lat'] > 30 else -0.8
    color = PRIMARY_COLOR if "Primary" in row['type'] else SECONDARY_COLOR
    
    ax.annotate(
        text=row['name'], 
        xy=(row['lon'], row['lat']), 
        xytext=(row['lon'], row['lat'] + y_offset),
        fontsize=12, 
        fontweight='bold', 
        color='white', 
        ha='center',
        bbox=dict(boxstyle="round,pad=0.5", fc="#1E293B", ec=color, alpha=0.85, linewidth=1.5), 
        zorder=6
    )

# Title and Labels formatting
plt.title("지정학적 맥락을 고려한 전략적 관심 구역(ROI) 매핑", fontsize=22, fontweight='bold', pad=25, color='white', loc='left')
plt.xlabel("경도 (Longitude)", fontsize=12, color='#94A3B8')
plt.ylabel("위도 (Latitude)", fontsize=12, color='#94A3B8')

# Tick styling
ax.tick_params(colors='#94A3B8', labelsize=10)
for spine in ax.spines.values():
    spine.set_color('#334155')

# Legend formatting
legend = plt.legend(
    loc='lower left', frameon=True, fontsize=12, framealpha=0.9, 
    facecolor='#0F172A', edgecolor='#334155', title="타겟 유형", labelcolor='white'
)
if legend:
    legend.get_title().set_color('white')
    legend.get_title().set_fontsize(13)
    legend.get_title().set_fontweight('bold')
    
    # Marker sizes in legend
    try:
        legend.legendHandles[0]._sizes = [150]
        legend.legendHandles[1]._sizes = [100]
    except AttributeError:
        pass

plt.grid(True, linestyle=':', color='#334155', alpha=0.6)
plt.tight_layout()

# Save locally to be immediately visible in the active workspace
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'premium_roi_map.png')

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Map successfully generated and saved to: {output_path}")
