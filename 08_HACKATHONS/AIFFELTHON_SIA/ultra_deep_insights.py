import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_ultra_deep_insights():
    print("Starting Ultra Deep Insight Mining (Level 1)...")
    try:
        # Load the main dataset for the richest insights
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        actors_to_check = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors_to_check)) | (main_df['Actor2CountryCode'].isin(actors_to_check))
        df = main_df[actor_mask].copy()
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
        df['Root'] = df['EventRootCode'].astype(str).str.zfill(2)

        # 1. Escalation Path Analysis (Markov Transition)
        print("\n1. Modeling Escalation Path (Matrix)...")
        # Shift the Root codes to find the next event in the same day or consecutive events
        # This is a simplified transition matrix over daily aggregates or sequences
        df_sorted = df.sort_values(['dt', 'GLOBALEVENTID'])
        df_sorted['prev_root'] = df_sorted['Root'].shift(1)
        transition_matrix = pd.crosstab(df_sorted['prev_root'], df_sorted['Root'], normalize='index')
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(transition_matrix, annot=False, cmap='YlOrRd')
        plt.title('Event Root Code Transition Matrix (Escalation Path)')
        plt.xlabel('Next Event Type')
        plt.ylabel('Current Event Type')
        plt.savefig('ultra_transition_matrix.png')

        # 2. News Elasticity Analysis
        print("2. Analyzing News Elasticity by Region...")
        top_cities = df['ActionGeo_FullName'].value_counts().head(5).index
        elasticity_data = []
        for city in top_cities:
            city_df = df[df['ActionGeo_FullName'] == city]
            # Slope of NumMentions / abs(GoldsteinScale) for negative events
            neg_city_df = city_df[city_df['GoldsteinScale'] < 0].copy()
            if len(neg_city_df) > 100:
                slope, intercept = np.polyfit(np.abs(neg_city_df['GoldsteinScale']), neg_city_df['NumMentions'], 1)
                elasticity_data.append({'Location': city, 'Elasticity': slope})
        
        elastic_df = pd.DataFrame(elasticity_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=elastic_df, x='Location', y='Elasticity', palette='viridis')
        plt.title('News Elasticity: Mention Increase per Incident Severity')
        plt.xticks(rotation=45, ha='right')
        plt.savefig('ultra_news_elasticity.png')

        # 3. Source Concentration Analysis (Proxy: Mentions Scale)
        print("3. Analyzing Source Concentration Pattern...")
        # Since NumSources is missing, we check Mentions distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['NumMentions'], bins=100, kde=True, log_scale=True, color='purple')
        plt.title('Distribution of News Mentions (Focusing on Extreme Outliers)')
        plt.savefig('ultra_mentions_distribution.png')

        # 4. Event Persistence Analysis (Decay)
        print("4. Analyzing Event Persistence (News Decay)...")
        # Identify high-impact dates
        top_dates = df.groupby('dt')['NumMentions'].sum().sort_values(ascending=False).head(5).index
        plt.figure(figsize=(12, 6))
        for t_date in top_dates:
            decay_window = df[(df['dt'] >= t_date) & (df['dt'] <= t_date + pd.Timedelta(days=14))]
            daily_decay = decay_window.groupby('dt')['NumMentions'].sum()
            # Normalize to peak
            daily_decay = daily_decay / daily_decay.iloc[0]
            plt.plot(range(len(daily_decay)), daily_decay, label=str(t_date.date()))
        
        plt.title('Event Interest Decay (Half-Life Analysis)')
        plt.xlabel('Days since Peak Incident')
        plt.ylabel('Relative News Volume (Peak = 1.0)')
        plt.legend()
        plt.grid(True)
        plt.savefig('ultra_event_decay.png')

        print("\nAll Ultra Deep Insight plots generated successfully.")

    except Exception as e:
        print(f"Error during Ultra Deep Mining: {e}")

if __name__ == "__main__":
    run_ultra_deep_insights()
