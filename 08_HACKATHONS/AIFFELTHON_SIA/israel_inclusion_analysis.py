import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_israel_inclusion_analysis():
    print("Launching Israel Inclusion Comparative Analysis...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # Base Filters
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        root_codes = ['15', '18', '19', '20']
        is_root_mask = (main_df['IsRootEvent'] == 1)
        
        # Group A: IRN-USA
        a_actors = ['IRN', 'USA']
        a_mask = (main_df['Actor1CountryCode'].isin(a_actors)) | (main_df['Actor2CountryCode'].isin(a_actors))
        df_a = main_df[a_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(root_codes)) & is_root_mask].copy()
        
        # Group B: IRN-USA-ISR
        b_actors = ['IRN', 'USA', 'ISR']
        b_mask = (main_df['Actor1CountryCode'].isin(b_actors)) | (main_df['Actor2CountryCode'].isin(b_actors))
        df_b = main_df[b_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(root_codes)) & is_root_mask].copy()

        for df in [df_a, df_b]:
            df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

        print(f"Datasets ready. IRN-USA: {len(df_a):,}, IRN-USA-ISR: {len(df_b):,}")

        # 1. Temporal Comparison
        fig, ax = plt.subplots(figsize=(15, 6))
        df_a.set_index('dt').resample('W').size().plot(ax=ax, label='IRN-USA only', color='blue', alpha=0.5)
        df_b.set_index('dt').resample('W').size().plot(ax=ax, label='IRN-USA-ISR (Expanded)', color='green', linewidth=2)
        plt.title('Conflict Intensity Comparison: Adding Israel to Target Actors')
        plt.legend()
        plt.grid(True)
        plt.savefig('isr_compare_temporal.png')

        # 2. Golden Time Lead-Lag Comparison (Simplified)
        def get_lead_time_stats(df_local):
            df_local['Root'] = df_local['EventRootCode'].astype(str).str.zfill(2)
            # Rough proxy for sign vs incident within nuanced set
            df_local['IsSign'] = df_local['Root'] == '15'
            # (Note: In nuanced set, we only have 15, 18, 19, 20. So 15 is the sign.)
            signs = df_local[df_local['IsSign']]
            incidents = df_local[~df_local['IsSign']]
            
            leads = []
            for city, group in df_local.groupby('ActionGeo_FullName'):
                city_signs = signs[signs['ActionGeo_FullName'] == city]
                city_incs = incidents[incidents['ActionGeo_FullName'] == city]
                for _, s in city_signs.iterrows():
                    f_inc = city_incs[(city_incs['dt'] > s['dt']) & (city_incs['dt'] <= s['dt'] + pd.Timedelta(days=14))]
                    if not f_inc.empty:
                        leads.append((f_inc.iloc[0]['dt'] - s['dt']).days)
            return leads

        leads_a = get_lead_time_stats(df_a)
        leads_b = get_lead_time_stats(df_b)
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(leads_a, label='IRN-USA', fill=True, color='blue', alpha=0.3)
        sns.kdeplot(leads_b, label='IRN-USA-ISR', fill=True, color='green', alpha=0.3)
        plt.title('Lead-Time Distribution Shift (Israel Inclusion)')
        plt.xlabel('Lead Time (Days)')
        plt.legend()
        plt.savefig('isr_compare_leadtime.png')

        # 3. Hotspot Shift
        top_a = df_a['ActionGeo_FullName'].value_counts().head(10)
        top_b = df_b['ActionGeo_FullName'].value_counts().head(10)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        sns.barplot(x=top_a.values, y=top_a.index, ax=axes[0], palette='Blues_r')
        sns.barplot(x=top_b.values, y=top_b.index, ax=axes[1], palette='Greens_r')
        axes[0].set_title('Top ROI (IRN-USA)')
        axes[1].set_title('Top ROI (IRN-USA-ISR)')
        plt.tight_layout()
        plt.savefig('isr_compare_hotspots.png')

        # Quantitative Summary
        isr_impact_stats = pd.DataFrame({
            'Category': ['Total Events', 'Unique Locations', 'Avg Lead Time (Days)'],
            'IRN-USA': [len(df_a), df_a['ActionGeo_FullName'].nunique(), np.mean(leads_a) if leads_a else 0],
            'Expanded (ISR)': [len(df_b), df_b['ActionGeo_FullName'].nunique(), np.mean(leads_b) if leads_b else 0]
        })
        print("\n--- Israel Inclusion Impact ---")
        print(isr_impact_stats)
        isr_impact_stats.to_csv('israel_inclusion_stats.csv', index=False)

        print("\nIsrael inclusion comparison plots generated.")

    except Exception as e:
        print(f"Error during Israel inclusion analysis: {e}")

if __name__ == "__main__":
    run_israel_inclusion_analysis()
