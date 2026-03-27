import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_refined_deep_eda():
    print("Starting Refined Comparative Deep EDA...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # Base Filters
        actors = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        
        # ROUGH Filter (Old)
        rough_root_codes = ['18', '19', '20']
        df_rough = main_df[actor_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(rough_root_codes))].copy()
        
        # REFINED Filter (New)
        refined_root_codes = ['15', '18', '19', '20']
        df_refined = main_df[actor_mask & geo_mask & goldstein_mask & \
                             (main_df['EventRootCode'].isin(refined_root_codes)) & \
                             (main_df['IsRootEvent'] == 1)].copy()

        df_rough['dt'] = pd.to_datetime(df_rough['SQLDATE'], format='%Y%m%d')
        df_refined['dt'] = pd.to_datetime(df_refined['SQLDATE'], format='%Y%m%d')

        # 1. Lead-Lag Comparison (Verbal vs Material)
        # We need Verbal (10-14) as well for both. Let's fix the dataset to include them for this specific test.
        verbal_codes = ['10', '11', '12', '13', '14']
        search_codes = verbal_codes + ['15', '18', '19', '20']
        
        # Re-extract for Lead-Lag specifically to include Verbal
        ll_mask_bare = actor_mask & geo_mask & (main_df['EventRootCode'].isin(search_codes))
        df_ll_rough = main_df[ll_mask_bare].copy()
        df_ll_refined = main_df[ll_mask_bare & (main_df['IsRootEvent'] == 1)].copy()
        
        def get_daily_cat(df):
            df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
            df['Root'] = df['EventRootCode'].astype(str).str.zfill(2)
            df['EventCategory'] = 'Other'
            df.loc[df['Root'].isin(['10','11','12','13','14']), 'EventCategory'] = 'Verbal'
            df.loc[df['Root'].isin(['18','19','20']), 'EventCategory'] = 'Material'
            return df.groupby(['dt', 'EventCategory']).size().unstack(fill_value=0)

        daily_rough = get_daily_cat(df_ll_rough)
        daily_refined = get_daily_cat(df_ll_refined)

        lags = range(-7, 8)
        corr_rough = [daily_rough['Verbal'].corr(daily_rough['Material'].shift(lag)) for lag in lags]
        corr_refined = [daily_refined['Verbal'].corr(daily_refined['Material'].shift(lag)) for lag in lags]

        plt.figure(figsize=(10, 5))
        plt.plot(lags, corr_rough, label='Rough (Mixed)', marker='o', linestyle='--')
        plt.plot(lags, corr_refined, label='Refined (Nuanced)', marker='s', color='red')
        plt.title('Lead-Lag Correlation: Verbal -> Future Material Conflict')
        plt.xlabel('Lag (Days)')
        plt.ylabel('Correlation')
        plt.legend()
        plt.grid(True)
        plt.savefig('refined_lead_lag_comparison.png')

        # 2. Diffusion Signal-to-Noise Ratio (Tone vs Mentions)
        print("Comparing Diffusion Signal-to-Noise...")
        plt.figure(figsize=(10, 6))
        plt.scatter(df_rough['AvgTone'], df_rough['NumMentions'], alpha=0.1, color='gray', label='Rough')
        plt.scatter(df_refined['AvgTone'], df_refined['NumMentions'], alpha=0.3, color='red', label='Refined')
        plt.yscale('log')
        plt.title('News Diffusion Precision: Tone vs. Mentions (Log Scale)')
        plt.legend()
        plt.grid(True)
        plt.savefig('refined_diffusion_comparison.png')

        # 3. Actor aggression stability
        print("Generating comparison statistics...")
        rough_stats = df_rough.groupby('Actor1Name')['GoldsteinScale'].mean().sort_values().head(10)
        refined_stats = df_refined.groupby('Actor1Name')['GoldsteinScale'].mean().sort_values().head(10)
        
        print("\n--- Top 10 Most Negative Actors (Rough) ---")
        print(rough_stats)
        print("\n--- Top 10 Most Negative Actors (Refined) ---")
        print(refined_stats)

        print("\nComparative EDA analysis finished.")

    except Exception as e:
        print(f"Error during refined deep eda: {e}")

if __name__ == "__main__":
    run_refined_deep_eda()
