import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_enhanced_map(resolution=0.1):
    print("Loading data for enhanced map...")
    try:
        # Load focused data
        df = pd.read_parquet('gdelt_main_final.parquet')
        actors_to_check = ['IRN', 'USA']
        actor_mask = (df['Actor1CountryCode'].isin(actors_to_check)) | (df['Actor2CountryCode'].isin(actors_to_check))
        event_mask = (df['ActionGeo_Type'] == 4) & (df['QuadClass'] == 4) & \
                     (df['EventRootCode'].isin(['18', '19', '20'])) & (df['GoldsteinScale'] < -7)
        df_focused = df[actor_mask & event_mask].copy()

        # Load ROI candidates
        top_rois = pd.read_csv('top_roi_candidates.csv')

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(df_focused, geometry=gpd.points_from_xy(df_focused.ActionGeo_Long, df_focused.ActionGeo_Lat), crs="EPSG:4326")
        
        # Convert to Web Mercator for contextily
        gdf = gdf.to_crs(epsg=3857)

        # Plot
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot all focused events as small dots
        gdf.plot(ax=ax, markersize=1, alpha=0.3, color='orange', label='Material Conflict Events')

        # Add ROI Buffers (Rectangles)
        from matplotlib.patches import Rectangle
        # We need to transform the grid points to EPSG:3857 for plotting
        top_rois_gdf = gpd.GeoDataFrame(top_rois, geometry=gpd.points_from_xy(top_rois.long_grid, top_rois.lat_grid), crs="EPSG:4326")
        top_rois_gdf = top_rois_gdf.to_crs(epsg=3857)

        for i, row in top_rois_gdf.iterrows():
            # Approximate resolution in web mercator (simplified)
            # Better to use the point and a fixed size or re-calculate the box properly
            ax.scatter(row.geometry.x, row.geometry.y, s=row.event_count/50, 
                       edgecolor='cyan', facecolor='none', linewidth=2, label='ROI Candidate' if i == 0 else "")

        # Add background map
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        
        plt.title('Enhanced Level 2b: Strategic ROI Mapping (with Geographic Context)', fontsize=16)
        plt.legend()
        plt.tight_layout()
        plt.savefig('enhanced_roi_map.png', dpi=150)
        print("\nSaved enhanced map to enhanced_roi_map.png")

    except Exception as e:
        print(f"Error during enhanced mapping: {e}")

if __name__ == "__main__":
    run_enhanced_map()
