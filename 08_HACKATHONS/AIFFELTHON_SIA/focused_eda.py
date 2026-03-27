import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_focused_eda():
    print("Loading datasets...")
    try:
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        url_df = pd.read_parquet('gdelt_url_final.parquet')
        
        print(f"Original Row Count: {len(main_df):,}")

        # 1. Apply Actor Filter (Iran and USA)
        # Using Actor1CountryCode and Actor2CountryCode
        actors_to_check = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors_to_check)) | \
                     (main_df['Actor2CountryCode'].isin(actors_to_check))
        
        # 2. Apply CAMEO Event Filters
        # ActionGeo_Type == 4 (City level)
        # QuadClass == 4 (Material Conflict)
        # EventRootCode in ['18', '19', '20']
        # GoldsteinScale < -7
        event_mask = (main_df['ActionGeo_Type'] == 4) & \
                     (main_df['QuadClass'] == 4) & \
                     (main_df['EventRootCode'].isin(['18', '19', '20'])) & \
                     (main_df['GoldsteinScale'] < -7)
        
        focused_df = main_df[actor_mask & event_mask].copy()
        print(f"Focused (Filtered) Row Count: {len(focused_df):,}")

        if len(focused_df) == 0:
            print("No events found matching the criteria.")
            return

        # 3. Join with Source URLs
        focused_df = focused_df.merge(url_df, on='GLOBALEVENTID', how='left')
        print("Joined with Source URLs.")

        # 4. Spatial Analysis
        print("\nTop 10 Cities for Focused Events:")
        top_cities = focused_df['ActionGeo_FullName'].value_counts().head(10)
        print(top_cities)

        plt.figure(figsize=(12, 6))
        top_cities.plot(kind='bar', color='salmon')
        plt.title('Top 10 Conflict Cities (Iran-US Focus)')
        plt.tight_layout()
        plt.savefig('focused_top_cities.png')

        # 5. Temporal Analysis
        focused_df['dt'] = pd.to_datetime(focused_df['SQLDATE'], format='%Y%m%d')
        daily_counts = focused_df.set_index('dt').resample('D').size()
        
        plt.figure(figsize=(15, 6))
        daily_counts.plot()
        plt.title('Daily Event Frequency (Iran-US Material Conflict)')
        plt.xlabel('Date')
        plt.ylabel('Event Count')
        plt.tight_layout()
        plt.savefig('focused_timeline.png')

        # 6. Sentiment/Tone Summary
        print("\nSentiment Summary (AvgTone):")
        print(focused_df['AvgTone'].describe())

        # 7. Sample Events with URL
        print("\nSample Filtered Events with URLs:")
        cols_to_show = ['SQLDATE', 'Actor1Name', 'Actor2Name', 'EventCode', 'ActionGeo_FullName', 'SOURCEURL']
        print(focused_df[cols_to_show].head(5))

        # Save the focused dataframe for further Level 2b/3 analysis
        focused_df.to_csv('focused_events_sample.csv', index=False)
        print("\nSaved focused events sample to focused_events_sample.csv")

    except Exception as e:
        print(f"Error during Focused EDA: {e}")

if __name__ == "__main__":
    run_focused_eda()
