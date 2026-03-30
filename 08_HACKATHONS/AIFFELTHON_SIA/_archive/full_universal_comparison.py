import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_universal_comparison():
    print("Launching Universal Side-by-Side Comparison (Rough vs Refined)...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # 0. Data Preparation
        actors = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        
        # ROUGH (Old Policy)
        rough_codes = ['18', '19', '20']
        df_rough = main_df[actor_mask & geo_mask & goldstein_mask & (main_df['EventRootCode'].isin(rough_codes))].copy()
        
        # REFINED (New Policy)
        refined_codes = ['15', '18', '19', '20']
        df_refined = main_df[actor_mask & geo_mask & goldstein_mask & \
                             (main_df['EventRootCode'].isin(refined_codes)) & \
                             (main_df['IsRootEvent'] == 1)].copy()

        for df in [df_rough, df_refined]:
            df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

        print(f"Datasets: Rough={len(df_rough):,}, Refined={len(df_refined):,}")

        # Helper for side-by-side plots
        def save_sb_plot(func_rough, func_refined, title, filename):
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            func_rough(axes[0])
            func_refined(axes[1])
            axes[0].set_title(f"{title} (Rough)")
            axes[1].set_title(f"{title} (Refined)")
            plt.tight_layout()
            plt.savefig(filename)
            print(f"Generated: {filename}")

        # 1. Temporal (Intensity)
        def plot_temporal(df, ax):
            df.set_index('dt').resample('W').size().plot(ax=ax, color='blue' if df.equals(df_rough) else 'red')
            ax.set_ylabel('Weekly Event Count')
        save_sb_plot(lambda ax: plot_temporal(df_rough, ax), 
                     lambda ax: plot_temporal(df_refined, ax), 
                     'Temporal Intensity', 'comp_temporal.png')

        # 2. News Diffusion (Tone vs Mentions)
        def plot_diffusion(df, ax):
            ax.scatter(df['AvgTone'], df['NumMentions'], alpha=0.3, color='blue' if df.equals(df_rough) else 'red')
            ax.set_yscale('log')
            ax.set_xlabel('AvgTone')
            ax.set_ylabel('NumMentions (Log)')
        save_sb_plot(lambda ax: plot_diffusion(df_rough, ax), 
                     lambda ax: plot_diffusion(df_refined, ax), 
                     'News Diffusion', 'comp_diffusion.png')

        # 3. Actor Aggression Profile
        def plot_actor_aggr(df, ax):
            stats = df.groupby('Actor1Name')['GoldsteinScale'].mean().sort_values().head(10)
            sns.barplot(x=stats.index, y=stats.values, ax=ax, palette='coolwarm')
            ax.set_xticklabels(stats.index, rotation=45, ha='right')
        save_sb_plot(lambda ax: plot_actor_aggr(df_rough, ax), 
                     lambda ax: plot_actor_aggr(df_refined, ax), 
                     'Top Aggressive Actors', 'comp_actor_aggr.png')

        # 4. Regional Volatility (Tehran)
        def plot_volatility(df, ax):
            city_df = df[df['ActionGeo_FullName'] == 'Tehran']
            if not city_df.empty:
                v = city_df.groupby(pd.Grouper(key='dt', freq='W'))['AvgTone'].std()
                v.plot(ax=ax, color='blue' if df.equals(df_rough) else 'red')
                ax.set_ylabel('Sentiment Volatility')
        save_sb_plot(lambda ax: plot_volatility(df_rough, ax), 
                     lambda ax: plot_volatility(df_refined, ax), 
                     'Tehran Sentiment Volatility', 'comp_volatility.png')

        # 5. Lead-Lag Correlation (Verbal vs Material)
        # Note: Needs broader data for verbal
        search_codes = ['10','11','12','13','14','18','19','20','15']
        ll_df_rough = main_df[actor_mask & geo_mask & (main_df['EventRootCode'].isin(search_codes))].copy()
        ll_df_refined = ll_df_rough[ll_df_rough['IsRootEvent'] == 1].copy()
        for d in [ll_df_rough, ll_df_refined]:
            d['dt'] = pd.to_datetime(d['SQLDATE'], format='%Y%m%d')
            d['Cat'] = 'Other'
            d.loc[d['EventRootCode'].astype(str).str.zfill(2).isin(['10','11','12','13','14']), 'Cat'] = 'Verbal'
            d.loc[d['EventRootCode'].astype(str).str.zfill(2).isin(['18','19','20']), 'Cat'] = 'Material'
        
        def plot_lead_lag(df_ll, ax):
            daily = df_ll.groupby(['dt', 'Cat']).size().unstack(fill_value=0)
            if 'Verbal' in daily.columns and 'Material' in daily.columns:
                lags = range(-7, 8)
                corrs = [daily['Verbal'].corr(daily['Material'].shift(lag)) for lag in lags]
                ax.stem(lags, corrs)
                ax.set_xlabel('Lag (Days)')
                ax.set_ylabel('Correlation')
        save_sb_plot(lambda ax: plot_lead_lag(ll_df_rough, ax), 
                     lambda ax: plot_lead_lag(ll_df_refined, ax), 
                     'Lead-Lag: Verbal -> Material', 'comp_lead_lag.png')

        # 6. Markov Transition (Escalation Path)
        def plot_markov(df, ax):
            df_sorted = df.sort_values(['dt', 'GLOBALEVENTID'])
            df_sorted['Root'] = df_sorted['EventRootCode'].astype(str).str.zfill(2)
            df_sorted['prev_root'] = df_sorted['Root'].shift(1)
            transition_matrix = pd.crosstab(df_sorted['prev_root'], df_sorted['Root'], normalize='index')
            sns.heatmap(transition_matrix, ax=ax, cmap='YlOrRd', cbar=False)
        save_sb_plot(lambda ax: plot_markov(df_rough, ax), 
                     lambda ax: plot_markov(df_refined, ax), 
                     'Markov Escalation Matrix', 'comp_markov.png')

        # 7. Event Decay (News Half-Life)
        def plot_decay(df, ax):
            top_date = df.groupby('dt')['NumMentions'].sum().idxmax()
            decay_window = df[(df['dt'] >= top_date) & (df['dt'] <= top_date + pd.Timedelta(days=14))]
            daily_decay = decay_window.groupby('dt')['NumMentions'].sum()
            daily_decay = daily_decay / daily_decay.iloc[0]
            ax.plot(range(len(daily_decay)), daily_decay, marker='o', color='blue' if df.equals(df_rough) else 'red')
            ax.set_xlabel('Days since Peak')
            ax.set_ylabel('Relative Volume')
        save_sb_plot(lambda ax: plot_decay(df_rough, ax), 
                     lambda ax: plot_decay(df_refined, ax), 
                     'Event News Decay', 'comp_decay.png')

        print("\nAll comparative analytic plots (7 categories) generated successfully.")

    except Exception as e:
        print(f"Error during universal comparison: {e}")

if __name__ == "__main__":
    run_universal_comparison()
