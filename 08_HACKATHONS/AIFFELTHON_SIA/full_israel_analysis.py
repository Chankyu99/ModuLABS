import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_full_israel_analysis():
    print("Launching Full Israel-Inclusive Analysis Suite...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # 0. Data Preparation (ISR Expanded Nuanced Set)
        actors = ['IRN', 'USA', 'ISR']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        root_codes = ['15', '18', '19', '20']
        is_root_mask = (main_df['IsRootEvent'] == 1)
        
        df = main_df[actor_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(root_codes)) & is_root_mask].copy()
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
        df['Root'] = df['EventRootCode'].astype(str).str.zfill(2)

        print(f"Dataset ready: {len(df):,} events.")

        # 1. Temporal Intensity
        plt.figure(figsize=(15, 6))
        df.set_index('dt').resample('W').size().plot(color='#2e7d32', linewidth=2)
        plt.title('ISR-USA-IRN Conflict Intensity (Weekly)')
        plt.grid(True)
        plt.savefig('isr_full_temporal.png')

        # 2. Lead-Lag Correlation (Verbal 10-14 -> Material 15,18,19,20)
        search_codes = ['10','11','12','13','14','15','18','19','20']
        df_ll = main_df[actor_mask & geo_mask & (main_df['EventRootCode'].isin(search_codes)) & (main_df['IsRootEvent'] == 1)].copy()
        df_ll['dt'] = pd.to_datetime(df_ll['SQLDATE'], format='%Y%m%d')
        df_ll['Cat'] = 'Other'
        df_ll.loc[df_ll['EventRootCode'].astype(str).str.zfill(2).isin(['10','11','12','13','14']), 'Cat'] = 'Verbal'
        df_ll.loc[df_ll['EventRootCode'].astype(str).str.zfill(2).isin(['15','18','19','20']), 'Cat'] = 'Material'
        
        daily = df_ll.groupby(['dt', 'Cat']).size().unstack(fill_value=0)
        if 'Verbal' in daily.columns and 'Material' in daily.columns:
            lags = range(-7, 8)
            corrs = [daily['Verbal'].corr(daily['Material'].shift(lag)) for lag in lags]
            plt.figure(figsize=(10, 5))
            plt.stem(lags, corrs, linefmt='#2e7d32', markerfmt='go')
            plt.title('ISR-triad Lead-Lag: Verbal -> Material')
            plt.savefig('isr_full_leadlag.png')

        # 3. News Diffusion (Tone vs Mentions)
        plt.figure(figsize=(10, 6))
        plt.scatter(df['AvgTone'], df['NumMentions'], alpha=0.3, color='#2e7d32')
        plt.yscale('log')
        plt.title('News Diffusion (Tone vs Mentions) - ISR triad')
        plt.savefig('isr_full_diffusion.png')

        # 4. Actor Aggression Profile
        stats = df.groupby('Actor1Name')['GoldsteinScale'].mean().sort_values().head(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=stats.index, y=stats.values, palette='Greens_r')
        plt.xticks(rotation=45, ha='right')
        plt.title('Top Aggressive Actors (ISR Expanded)')
        plt.savefig('isr_full_actor.png')

        # 5. Regional Volatility (Jerusalem/Gaza focus)
        target_city = 'Jerusalem'
        city_df = df[df['ActionGeo_FullName'].str.contains(target_city, case=False, na=False)]
        if not city_df.empty:
            vol = city_df.groupby(pd.Grouper(key='dt', freq='W'))['AvgTone'].std()
            plt.figure(figsize=(15, 5))
            vol.plot(color='#d32f2f', linewidth=2)
            plt.title(f'Sentiment Volatility in {target_city}')
            plt.savefig('isr_full_volatility.png')

        # 6. Markov Transition (Escalation Path)
        df_sorted = df.sort_values(['dt', 'GLOBALEVENTID'])
        df_sorted['prev_root'] = df_sorted['Root'].shift(1)
        transition_matrix = pd.crosstab(df_sorted['prev_root'], df_sorted['Root'], normalize='index')
        plt.figure(figsize=(10, 6))
        sns.heatmap(transition_matrix, cmap='YlGn', annot=True)
        plt.title('ISR triad Escalation Markov Matrix')
        plt.savefig('isr_full_markov.png')

        # 7. Event News Decay (Half-Life)
        top_date = df.groupby('dt')['NumMentions'].sum().idxmax()
        decay_window = df[(df['dt'] >= top_date) & (df['dt'] <= top_date + pd.Timedelta(days=14))]
        daily_decay = decay_window.groupby('dt')['NumMentions'].sum()
        daily_decay = daily_decay / daily_decay.iloc[0]
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(daily_decay)), daily_decay, marker='o', color='#2e7d32')
        plt.title('ISR triad Event News Decay')
        plt.savefig('isr_full_decay.png')

        # 8. Golden Time Lead-Time Audit
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
            plt.figure(figsize=(10, 6))
            sns.histplot(leads, bins=range(1, 16), kde=True, color='green')
            plt.title(f'Golden Time Window (ISR Exp): Avg {np.mean(leads):.1f} days')
            plt.savefig('isr_full_golden.png')

        # 9. Hotspot Strategic Audit
        top_cities = df['ActionGeo_FullName'].value_counts().head(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_cities.values, y=top_cities.index, palette='Greens_r')
        plt.title('Top 10 Hotspots (ISR triad)')
        plt.savefig('isr_full_hotspots.png')

        print("\nAll ISR-inclusive analysis plots generated.")

    except Exception as e:
        print(f"Error during full Israel analysis: {e}")

if __name__ == "__main__":
    run_full_israel_analysis()
