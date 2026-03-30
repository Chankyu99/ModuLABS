import pandas as pd
import matplotlib.pyplot as plt
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def analyze_recent_issues():
    print("Loading focused events (842k rows)...")
    try:
        # Load the focused events saved previously
        df = pd.read_csv('focused_events_sample.csv')
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')

        # 1. Filter for Recent Period (2025-2026)
        recent_df = df[df['dt'] >= '2025-01-01'].copy()
        print(f"Recent (2025-2026) Events: {len(recent_df):,}")

        # 2. Key Actor Interaction (Beyond just IRN/USA)
        # Who are we interacting with the most in this focused subset?
        print("\nTop Involved Actors (Actor2) in Recent IRN/USA Conflicts:")
        top_actors = recent_df['Actor2Name'].value_counts().head(10)
        print(top_actors)

        # 3. Intensity Search (Top Mentions)
        # Events with high mentions are likely more "verified" or "critical"
        critical_events = recent_df.sort_values(by='NumMentions', ascending=False).head(50)
        
        # 4. Visualization: Recent Intensity
        plt.figure(figsize=(15, 6))
        recent_df.set_index('dt').resample('W').size().plot(color='red', marker='o')
        plt.title('Weekly Significant Conflict Intensity (2025-2026)')
        plt.ylabel('Event Count')
        plt.grid(True)
        plt.savefig('recent_intensity_trend.png')
        print("\nSaved recent intensity trend to recent_intensity_trend.png")

        # 5. Top 10 Critical Events Report
        print("\n--- Top 10 Recent Critical Events (by Mentions) ---")
        report_cols = ['SQLDATE', 'Actor1Name', 'Actor2Name', 'EventCode', 'ActionGeo_FullName', 'NumMentions', 'AvgTone', 'SOURCEURL']
        print(critical_events[report_cols].head(10))

        # Save for Level 3 LLM verification
        critical_events[report_cols].to_csv('critical_events_for_llm.csv', index=False)
        print("\nFinalized critical event list for Level 3: critical_events_for_llm.csv")

    except Exception as e:
        print(f"Error during recent issue analysis: {e}")

if __name__ == "__main__":
    analyze_recent_issues()
