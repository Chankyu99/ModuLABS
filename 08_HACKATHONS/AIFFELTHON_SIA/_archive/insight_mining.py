import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_insight_mining():
    print("Starting Advanced Insight Mining (Level 1)...")
    try:
        # Load the focused dataset (material conflicts + verbal for full context if possible)
        # We'll reload the main parquet but only necessary columns to save memory
        cols = ['dt', 'Actor1Name', 'Actor2Name', 'GoldsteinScale', 'NumMentions', 'AvgTone', 'ActionGeo_FullName']
        
        # We'll use the combined verbal/material logic from deep_eda.py
        # But for speed, let's use the focused sample if it's enough, 
        # however for Actor profiling we want the richest set.
        
        print("Loading data for profiling...")
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        actors_to_check = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors_to_check)) | (main_df['Actor2CountryCode'].isin(actors_to_check))
        
        # Include a broader range for insight mining
        event_mask = (main_df['GoldsteinScale'] < 0) & (main_df['ActionGeo_Type'] == 4)
        df = main_df[actor_mask & event_mask].copy()
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

        # 1. Actor Aggression Profile
        print("\n1. Generating Actor Aggression Profile...")
        # Focus on top 20 most active Actor1Names (excluding generic country codes if possible)
        top_actors = df['Actor1Name'].value_counts().head(20).index
        actor1_stats = df[df['Actor1Name'].isin(top_actors)].groupby('Actor1Name').agg({
            'GoldsteinScale': 'mean',
            'NumMentions': 'sum'
        }).sort_values('GoldsteinScale')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=actor1_stats.index, y=actor1_stats['GoldsteinScale'], palette='Reds_r')
        plt.title('Actor Aggression Profile (Average Goldstein Scale)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('insight_actor_aggression.png')

        # 2. Regional Volatility Analysis
        print("2. Analyzing Regional Volatility...")
        top_cities = df['ActionGeo_FullName'].value_counts().head(5).index
        volatility = df[df['ActionGeo_FullName'].isin(top_cities)].groupby(['ActionGeo_FullName', pd.Grouper(key='dt', freq='W')])['AvgTone'].std().unstack(level=0)
        
        plt.figure(figsize=(12, 6))
        volatility.plot(marker='s', alpha=0.7)
        plt.title('Weekly Sentiment Volatility (Std Dev of AvgTone) by Location')
        plt.ylabel('Volatility (Std Dev)')
        plt.grid(True)
        plt.savefig('insight_regional_volatility.png')

        # 3. Sentiment Lead-Lag Analysis (Pre-spike Sentiment)
        print("3. Performing Sentiment Lead-Lag Analysis...")
        daily_stats = df.groupby('dt').agg({'AvgTone': 'mean', 'NumMentions': 'sum'})
        lags = range(-7, 8)
        # Relationship between today's Sentiment and future Mentions
        correlations = [daily_stats['AvgTone'].corr(daily_stats['NumMentions'].shift(-lag)) for lag in lags]
        
        plt.figure(figsize=(10, 5))
        plt.stem(lags, correlations)
        plt.title('Correlation: Today\'s Tone vs. Future News Volume (NumMentions)')
        plt.xlabel('Lag (Days) - Positive means Tone leads Mentions')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        plt.savefig('insight_sentiment_lag.png')

        # 4. Mediator Influence Analysis
        print("4. Analyzing Mediator/Third-Party Influence...")
        # Actors that are involved but ARE NOT USA/IRN directly (as Actor 2)
        # (Assuming Actor1 is USA or IRN)
        direct_actors = ['UNITED STATES', 'IRAN', 'THE US', 'IRANIAN', 'AMERICAN', 'TEHRAN', 'WASHINGTON']
        mediator_df = df[(df['Actor1Name'].isin(direct_actors)) & (~df['Actor2Name'].isin(direct_actors)) & (df['Actor2Name'].notnull())]
        mediator_stats = mediator_df.groupby('Actor2Name').agg({
            'GoldsteinScale': 'mean',
            'NumMentions': 'sum'
        }).sort_values('NumMentions', ascending=False).head(15)
        
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=mediator_stats, x='GoldsteinScale', y='NumMentions', size='NumMentions', hue='GoldsteinScale', palette='coolwarm', sizes=(100, 1000))
        plt.title('Third-Party Influence: News Attention vs. Conflict Scale')
        plt.axvline(0, color='gray', linestyle='--')
        for i, txt in enumerate(mediator_stats.index):
            plt.annotate(txt, (mediator_stats.GoldsteinScale[i], mediator_stats.NumMentions[i]))
        plt.savefig('insight_mediator_influence.png')

        print("\nAll Insight Mining plots generated successfully.")

    except Exception as e:
        print(f"Error during Insight Mining: {e}")

if __name__ == "__main__":
    run_insight_mining()
