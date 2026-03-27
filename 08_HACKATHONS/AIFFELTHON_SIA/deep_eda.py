import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_deep_eda():
    print("Starting Deep EDA (Level 1)...")
    try:
        # We need both Verbal (10-14) and Material (18-20) for lead-lag analysis
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        actors_to_check = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors_to_check)) | (main_df['Actor2CountryCode'].isin(actors_to_check))
        
        verbal_codes = ['10', '11', '12', '13', '14']
        material_codes = ['18', '19', '20']
        combined_codes = verbal_codes + material_codes
        
        event_mask = (main_df['ActionGeo_Type'] == 4) & \
                     (main_df['EventRootCode'].isin(combined_codes)) & \
                     (main_df['GoldsteinScale'] < 0) # Loosened for verbal
        
        df = main_df[actor_mask & event_mask].copy()
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

        print("1. Event Sequence Analysis (Lead-Lag)")
        # We want to see if "Verbal Disagreement" (EventRootCode 10-14) leads to "Material Conflict" (18-20)
        # Note: EventRootCode is object in parquet but might be int in CSV. Let's force object.
        df['Root'] = df['EventRootCode'].astype(str).str.zfill(2)
        
        verbal_codes = ['10', '11', '12', '13', '14']
        material_codes = ['18', '19', '20']

        print(f"Unique Root Codes in Data: {df['Root'].unique()}")

        df['EventCategory'] = 'Other'
        df.loc[df['Root'].isin(verbal_codes), 'EventCategory'] = 'Verbal_Conflict'
        df.loc[df['Root'].isin(material_codes), 'EventCategory'] = 'Material_Conflict'
        
        print(f"Event Categories found: {df['EventCategory'].unique()}")

        # Daily aggregations
        daily_cat = df.groupby(['dt', 'EventCategory']).size().unstack(fill_value=0)
        
        if 'Verbal_Conflict' in daily_cat.columns and 'Material_Conflict' in daily_cat.columns:
            # Cross-correlation between Verbal and Material
            lags = range(-10, 11)
            correlations = [daily_cat['Verbal_Conflict'].corr(daily_cat['Material_Conflict'].shift(lag)) for lag in lags]
            
            plt.figure(figsize=(10, 5))
            plt.stem(lags, correlations)
            plt.title('Cross-Correlation: Verbal Conflict vs. Delayed Material Conflict')
            plt.xlabel('Lag (Days) - Positive means Verbal precedes Material')
            plt.ylabel('Correlation Coefficient')
            plt.grid(True)
            plt.savefig('deep_eda_correlation.png')
            print("Saved cross-correlation plot to deep_eda_correlation.png")

        print("\n2. Actor Profile Impact")
        # Checking if the 'Actor1Type1Code' exists (e.g. MIL, GOV)
        # We need to reload the parquet since CSV might not have all columns
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        # Re-apply filters for full column set
        actors_to_check = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors_to_check)) | (main_df['Actor2CountryCode'].isin(actors_to_check))
        event_mask = (main_df['ActionGeo_Type'] == 4) & (main_df['QuadClass'] == 4) & \
                     (main_df['EventRootCode'].isin(['18', '19', '20'])) & (main_df['GoldsteinScale'] < -7)
        full_focused = main_df[actor_mask & event_mask].copy()

        # Add Actor1Type1Code analysis if exists (it's not in the 19 columns identified previously)
        # Let's check columns again
        print(f"Main Columns: {main_df.columns.tolist()}")
        
        # If Actor1Code is what we have:
        # Actually in GDELT v1, Actor1Code often contains Type codes. 
        # But based on our previous df.info(), we have 19 columns.

        print("\n3. News Diffusion: AvgTone vs. NumMentions")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=full_focused, x='AvgTone', y='NumMentions', alpha=0.1, color='blue')
        plt.title('Propensity of News Diffusion: Tone vs. Mention Frequency')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig('deep_eda_diffusion.png')
        print("Saved news diffusion plot to deep_eda_diffusion.png")

        # 4. Critical Escalation Marker (High Mentions + Deep Negative Tone)
        extreme_events = full_focused[(full_focused['NumMentions'] > full_focused['NumMentions'].quantile(0.99)) & 
                                  (full_focused['AvgTone'] < full_focused['AvgTone'].quantile(0.01))]
        print(f"\nDetected {len(extreme_events)} Extreme Escalation Markers.")
        extreme_events.to_csv('extreme_escalation_events.csv', index=False)
        print("Saved extreme markers to extreme_escalation_events.csv")

    except Exception as e:
        print(f"Error during Deep EDA: {e}")

if __name__ == "__main__":
    run_deep_eda()
