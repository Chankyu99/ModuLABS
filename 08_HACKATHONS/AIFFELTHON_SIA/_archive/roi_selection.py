import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_roi_selection(resolution=0.1):
    print(f"Loading focused events dataset (using 0.1 degree resolution for grid)...")
    try:
        # Load the focused data generated in the previous step
        # Note: If it's too large, we can sample it for visualization, but for ROI selection we want the full count.
        df = pd.read_parquet('gdelt_main_final.parquet') # Better to use the full data with filters or read the csv if saved.
        # Let's filter again to be sure if we are starting fresh here
        actors_to_check = ['IRN', 'USA']
        actor_mask = (df['Actor1CountryCode'].isin(actors_to_check)) | (df['Actor2CountryCode'].isin(actors_to_check))
        event_mask = (df['ActionGeo_Type'] == 4) & (df['QuadClass'] == 4) & \
                     (df['EventRootCode'].isin(['18', '19', '20'])) & (df['GoldsteinScale'] < -7)
        df_focused = df[actor_mask & event_mask].copy()

        print(f"Focused Events Count: {len(df_focused):,}")

        # 1. Coordinate Grid Transformation (Simulating DGGS)
        df_focused['lat_grid'] = (df_focused['ActionGeo_Lat'] / resolution).round() * resolution
        df_focused['long_grid'] = (df_focused['ActionGeo_Long'] / resolution).round() * resolution

        # Count events per grid cell
        grid_counts = df_focused.groupby(['lat_grid', 'long_grid']).size().reset_index(name='event_count')
        grid_counts = grid_counts.sort_values(by='event_count', ascending=False)

        print(f"Detected {len(grid_counts):,} unique grid cells.")
        
        # 2. Identify ROI Buffers (Top 20 high-density grids)
        top_rois = grid_counts.head(20).copy()
        print("\nTop 20 Potential ROI Grids:")
        print(top_rois)

        # 3. Visualization on Map
        plt.figure(figsize=(15, 10))
        # Base scatter for all focused events
        plt.scatter(df_focused['ActionGeo_Long'], df_focused['ActionGeo_Lat'], s=1, alpha=0.1, color='gray', label='All Focused Events')
        
        # Overlay Heatmap/Density for High Frequency areas
        sns.kdeplot(data=df_focused, x='ActionGeo_Long', y='ActionGeo_Lat', fill=True, cmap='Reds', alpha=0.5)

        # Highlight top 20 ROIs
        plt.scatter(top_rois['long_grid'], top_rois['lat_grid'], s=top_rois['event_count']/100, 
                    edgecolor='blue', facecolor='none', linewidth=2, label='Top ROI Candidates (Scaled by Count)')
        
        # Add Buffer visualization (Rectangles)
        from matplotlib.patches import Rectangle
        ax = plt.gca()
        for i, row in top_rois.iterrows():
            rect = Rectangle((row['long_grid']-resolution/2, row['lat_grid']-resolution/2), resolution, resolution, 
                             fill=False, edgecolor='green', linestyle='--', alpha=0.5)
            ax.add_patch(rect)

        plt.title(f'Level 2b: ROI Selection & Buffet Grid (Iran-US Focus, Res: {resolution}°)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.xlim(df_focused['ActionGeo_Long'].min()-2, df_focused['ActionGeo_Long'].max()+2)
        plt.ylim(df_focused['ActionGeo_Lat'].min()-2, df_focused['ActionGeo_Lat'].max()+2)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roi_selection_map.png')
        print("\nSaved ROI selection map to roi_selection_map.png")

        # 4. Save ROI List
        top_rois.to_csv('top_roi_candidates.csv', index=False)
        print("Saved Top ROI candidates to top_roi_candidates.csv")

    except Exception as e:
        print(f"Error during ROI selection: {e}")

if __name__ == "__main__":
    run_roi_selection()
