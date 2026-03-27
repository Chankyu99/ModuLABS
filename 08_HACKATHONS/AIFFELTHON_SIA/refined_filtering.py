import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_comparative_filtering():
    print("Loading GDELT main parquet...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # Actors: IRN or USA
        actors = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        
        # Baseline Filters common to both
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        
        # 1. Rough Setup (Original)
        rough_root_codes = ['18', '19', '20']
        rough_mask = actor_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(rough_root_codes))
        df_rough = main_df[rough_mask].copy()
        
        # 2. Nuanced Setup (Refined)
        refined_root_codes = ['15', '18', '19', '20']
        # Adding IsRootEvent = 1
        refined_mask = actor_mask & geo_mask & goldstein_mask & \
                       (main_df['EventRootCode'].isin(refined_root_codes)) & \
                       (main_df['IsRootEvent'] == 1)
        df_refined = main_df[refined_mask].copy()
        
        print(f"\n--- Quantitative Comparison ---")
        print(f"Rough Event Count: {len(df_rough):,}")
        print(f"Refined Event Count: {len(df_refined):,}")
        print(f"Data Reduction: {(1 - len(df_refined)/len(df_rough))*100:.1f}%")
        
        # Unique locations
        print(f"Rough Unique Locations: {df_rough['ActionGeo_FullName'].nunique():,}")
        print(f"Refined Unique Locations: {df_refined['ActionGeo_FullName'].nunique():,}")

        # Visualization: Temporal Comparison
        plt.figure(figsize=(15, 6))
        df_rough['dt'] = pd.to_datetime(df_rough['SQLDATE'], format='%Y%m%d')
        df_refined['dt'] = pd.to_datetime(df_refined['SQLDATE'], format='%Y%m%d')
        
        df_rough.set_index('dt').resample('W').size().plot(label='Rough (Original)', color='gray', alpha=0.5)
        df_refined.set_index('dt').resample('W').size().plot(label='Refined (Nuanced)', color='red', linewidth=2)
        
        plt.title('Conflict Intensity Comparison: Rough vs. Refined Filtering (Weekly)')
        plt.legend()
        plt.grid(True)
        plt.savefig('comparison_temporal.png')
        
        # 3. Hotspot Comparison (Top 5 Cities)
        top_rough = df_rough['ActionGeo_FullName'].value_counts().head(5)
        top_refined = df_refined['ActionGeo_FullName'].value_counts().head(5)
        
        print("\n--- Top 5 Locations (Rough) ---")
        print(top_rough)
        print("\n--- Top 5 Locations (Refined) ---")
        print(top_refined)

        # Save datasets for record
        # (We won't save full yet to save space, just summaries)

        print("\nComparative plots generated: comparison_temporal.png")

    except Exception as e:
        print(f"Error during comparative filtering: {e}")

if __name__ == "__main__":
    run_comparative_filtering()
