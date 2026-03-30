import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_full_comparative_analysis():
    print("Initializing Full Side-by-Side Comparison (Rough vs Refined)...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # Define Filters
        actors = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        
        # 1. ROUGH (Original)
        rough_codes = ['18', '19', '20']
        df_rough = main_df[actor_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(rough_codes))].copy()
        
        # 2. REFINED (Nuanced)
        refined_codes = ['15', '18', '19', '20']
        df_refined = main_df[actor_mask & geo_mask & goldstein_mask & \
                             (main_df['EventRootCode'].isin(refined_codes)) & \
                             (main_df['IsRootEvent'] == 1)].copy()

        print(f"Datasets ready. Rough: {len(df_rough):,}, Refined: {len(df_refined):,}")

        # Helper for common plots
        def plot_compare(data_rough, data_refined, title, filename, plot_type='bar', columns=None):
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            if plot_type == 'bar':
                sns.barplot(x=data_rough.index, y=data_rough.values, ax=axes[0], palette='Blues_r')
                sns.barplot(x=data_refined.index, y=data_refined.values, ax=axes[1], palette='Reds_r')
            elif plot_type == 'scatter':
                axes[0].scatter(data_rough[columns[0]], data_rough[columns[1]], alpha=0.1, color='blue')
                axes[0].set_yscale('log')
                axes[1].scatter(data_refined[columns[0]], data_refined[columns[1]], alpha=0.3, color='red')
                axes[1].set_yscale('log')
            
            axes[0].set_title(f"{title} (Rough)")
            axes[1].set_title(f"{title} (Refined)")
            plt.tight_layout()
            plt.savefig(filename)
            print(f"Generated: {filename}")

        # A. Actor Aggression Profile Comparison
        aggr_rough = df_rough.groupby('Actor1Name')['GoldsteinScale'].mean().sort_values().head(10)
        aggr_refined = df_refined.groupby('Actor1Name')['GoldsteinScale'].mean().sort_values().head(10)
        plot_compare(aggr_rough, aggr_refined, 'Actor Aggression Profile', 'final_compare_actor_aggr.png')

        # B. News Diffusion Comparison
        plot_compare(df_rough, df_refined, 'News Diffusion (Tone vs Mentions)', 'final_compare_diffusion.png', 
                     plot_type='scatter', columns=['AvgTone', 'NumMentions'])

        # C. Regional Volatility Comparison (Weekly Std Dev)
        df_rough['dt'] = pd.to_datetime(df_rough['SQLDATE'], format='%Y%m%d')
        df_refined['dt'] = pd.to_datetime(df_refined['SQLDATE'], format='%Y%m%d')
        
        top_city = 'Tehran' # Focus on the main hotspot for clarity
        vol_rough = df_rough[df_rough['ActionGeo_FullName'] == top_city].groupby(pd.Grouper(key='dt', freq='W'))['AvgTone'].std()
        vol_refined = df_refined[df_refined['ActionGeo_FullName'] == top_city].groupby(pd.Grouper(key='dt', freq='W'))['AvgTone'].std()
        
        plt.figure(figsize=(15, 6))
        plt.plot(vol_rough.index, vol_rough.values, label='Rough (Volatility)', color='blue', alpha=0.5, linestyle='--')
        plt.plot(vol_refined.index, vol_refined.values, label='Refined (Volatility)', color='red', linewidth=2)
        plt.title(f'Sentiment Volatility Comparison in {top_city} (Weekly)')
        plt.legend()
        plt.grid(True)
        plt.savefig('final_compare_volatility.png')
        print("Generated: final_compare_volatility.png")

        # D. Summary Table Generation (Metrics)
        comparison_stats = pd.DataFrame({
            'Metric': ['Event Count', 'Unique Geo Points', 'Avg News Mentions'],
            'Rough': [len(df_rough), df_rough['ActionGeo_FullName'].nunique(), df_rough['NumMentions'].mean()],
            'Refined': [len(df_refined), df_refined['ActionGeo_FullName'].nunique(), df_refined['NumMentions'].mean()]
        })
        print("\n--- Summary Statistics ---")
        print(comparison_stats)
        comparison_stats.to_csv('final_comparison_stats.csv', index=False)

        print("\nAll comparative analytic plots and data generated successfully.")

    except Exception as e:
        print(f"Error during full comparative analysis: {e}")

if __name__ == "__main__":
    run_full_comparative_analysis()
