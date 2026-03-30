import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set working directory to the project folder
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_eda():
    print("Loading data... this might take a moment due to the large file size (2.6GB RAM).")
    try:
        # Load data
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        url_df = pd.read_parquet('gdelt_url_final.parquet')
        print("Data loaded successfully.")

        # Basic Info
        print("\n--- Main Data Info ---")
        print(main_df.info())
        print("\n--- URL Data Info ---")
        print(url_df.info())

        # Descriptive Statistics
        print("\n--- Descriptive Statistics ---")
        print(main_df.describe())

        # Check for missing values
        print("\n--- Missing Values ---")
        print(main_df.isnull().sum())

        # Top 20 locations
        print("\nTop 20 Action Locations:")
        top_locations = main_df['ActionGeo_FullName'].value_counts().head(20)
        print(top_locations)

        # Plotting
        plt.figure(figsize=(12, 6))
        top_locations.plot(kind='bar')
        plt.title('Top 20 Action Locations')
        plt.tight_layout()
        plt.savefig('top_locations.png')
        print("Saved plot to top_locations.png")

        # Distribution of GoldsteinScale and AvgTone
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        sns.histplot(main_df['GoldsteinScale'], bins=50, ax=ax[0])
        ax[0].set_title('Distribution of GoldsteinScale')
        sns.histplot(main_df['AvgTone'], bins=50, ax=ax[1])
        ax[1].set_title('Distribution of AvgTone')
        plt.tight_layout()
        plt.savefig('sentiment_distributions.png')
        print("Saved plot to sentiment_distributions.png")

        # Time series of event counts
        print("\nGenerating time series analysis...")
        main_df['dt'] = pd.to_datetime(main_df['SQLDATE'], format='%Y%m%d')
        time_series = main_df.set_index('dt').resample('M').size()
        plt.figure(figsize=(12, 6))
        time_series.plot()
        plt.title('Monthly Event Count Over Time')
        plt.tight_layout()
        plt.savefig('event_timeline.png')
        print("Saved plot to event_timeline.png")

    except Exception as e:
        print(f"Error during EDA: {e}")

if __name__ == "__main__":
    run_eda()
