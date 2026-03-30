import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_refined_isr_analysis():
    print("Refining ISR analysis metrics and terminology...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # 0. Data Prep (Nuanced ISR)
        actors = ['IRN', 'USA', 'ISR']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        root_codes = ['15', '18', '19', '20']
        is_root_mask = (main_df['IsRootEvent'] == 1)
        
        df = main_df[actor_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(root_codes)) & is_root_mask].copy()
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
        df['Root'] = df['EventRootCode'].astype(str).str.zfill(2)

        # 1. Refined Volatility (Tehran/Teheran) - Remove 0 outliers and low-count weeks
        target_city = 'Tehran|Teheran'
        city_df = df[df['ActionGeo_FullName'].str.contains(target_city, case=False, na=False)]
        if not city_df.empty:
            # Calculate both standard deviation and count per week
            vol_stats = city_df.groupby(pd.Grouper(key='dt', freq='W'))['AvgTone'].agg(['std', 'count'])
            
            # Filter: 1) Must have at least 3 events to make 'volatility' meaningful
            #         2) Standard deviation must be > 0 (avoids single-value artifacts)
            refined_vol = vol_stats[(vol_stats['count'] >= 3) & (vol_stats['std'] > 0.1)]['std']
            
            plt.figure(figsize=(15, 5))
            refined_vol.plot(color='#d32f2f', linewidth=2, marker='.', linestyle='-')
            plt.title(f'Refined Sentiment Volatility: Tehran Region (Cleaned)')
            plt.ylabel('Volatility (Std Dev of Tone)')
            plt.xlabel('Date (Weekly)')
            plt.grid(True, alpha=0.3)
            plt.savefig('final_isr_volatility_tehran.png')
            print(f"Refined volatility plot for Tehran saved to 'final_isr_volatility_tehran.png'")

        # 2. Refined Golden Time (Capped/Focused on 2-3 days)
        # Re-run sequence logic
        signs = df[df['Root'] == '15']
        incidents = df[df['Root'].isin(['18','19','20'])]
        leads = []
        for city, group in df.groupby('ActionGeo_FullName'):
            city_signs = group[group['Root'] == '15']
            city_incs = group[group['Root'].isin(['18','19','20'])]
            for _, s in city_signs.iterrows():
                f_inc = city_incs[(city_incs['dt'] > s['dt']) & (city_incs['dt'] <= s['dt'] + pd.Timedelta(days=14))]
                if not f_inc.empty:
                    leads.append((f_inc.iloc[0]['dt'] - s['dt']).days)
        
        if leads:
            leads_arr = np.array(leads)
            # Find the proportion of events within 3 days
            within_3 = (leads_arr <= 3).mean() * 100
            
            plt.figure(figsize=(10, 6))
            sns.histplot(leads, bins=range(1, 16), kde=True, color='green')
            plt.axvspan(1, 3, alpha=0.2, color='orange', label=f'Target Window (1-3 days): {within_3:.1f}%')
            plt.title(f'Optimized Golden Time Window (Target 2-3 days)')
            plt.xlabel('Lead Time (Days until Tension High)')
            plt.legend()
            plt.savefig('final_isr_golden_refined.png')

        # 3. Refined Markov (Term: 갈등 심화)
        df_sorted = df.sort_values(['dt', 'GLOBALEVENTID'])
        df_sorted['prev_root'] = df_sorted['Root'].shift(1)
        transition_matrix = pd.crosstab(df_sorted['prev_root'], df_sorted['Root'], normalize='index')
        plt.figure(figsize=(10, 6))
        sns.heatmap(transition_matrix, cmap='YlGn', annot=True)
        plt.title('Conflict Intensification Matrix (기존 에스컬레이션)')
        plt.savefig('final_isr_markov_refined.png')

        print("\nRefined plots generated successfully.")

    except Exception as e:
        print(f"Error during metric refinement: {e}")

if __name__ == "__main__":
    run_refined_isr_analysis()
