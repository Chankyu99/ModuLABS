import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set working directory
os.chdir('/Users/chankyulee/Desktop/ModuLABS/08_HACKATHONS/AIFFELTHON_SIA')

def run_golden_time_analysis():
    print("Quantifying Golden Time Window & Geopolitical Hotspots...")
    try:
        # Load Refined Dataset (Nuanced Policy)
        main_df = pd.read_parquet('gdelt_main_final.parquet')
        
        # Apply Refined Filters
        actors = ['IRN', 'USA']
        actor_mask = (main_df['Actor1CountryCode'].isin(actors)) | (main_df['Actor2CountryCode'].isin(actors))
        geo_mask = (main_df['ActionGeo_Type'] == 4)
        goldstein_mask = (main_df['GoldsteinScale'] < -7)
        
        # We need ALL codes to find signs (10-15) and events (18-20)
        # Note: Root 15 is our 'military posture' sign.
        sign_codes = ['10', '11', '12', '13', '14', '15']
        event_codes = ['18', '19', '20']
        
        all_relevant_mask = actor_mask & geo_mask & (main_df['IsRootEvent'] == 1)
        df = main_df[all_relevant_mask].copy()
        df['dt'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
        df['Root'] = df['EventRootCode'].astype(str).str.zfill(2)
        
        # Categorize
        df['Type'] = 'Other'
        df.loc[df['Root'].isin(sign_codes), 'Type'] = 'Sign'
        df.loc[df['Root'].isin(event_codes), 'Type'] = 'Incident'
        
        # 1. Golden Time Sequence Analysis
        df = df[df['Type'] != 'Other'].sort_values(['ActionGeo_FullName', 'dt'])
        
        sequences = []
        for city, group in df.groupby('ActionGeo_FullName'):
            if len(group) < 5: continue # Skip minor locations
            
            signs = group[group['Type'] == 'Sign']
            incidents = group[group['Type'] == 'Incident']
            
            for _, s in signs.iterrows():
                # Find the first incident that occurs AFTER this sign within 14 days
                future_incidents = incidents[(incidents['dt'] > s['dt']) & (incidents['dt'] <= s['dt'] + pd.Timedelta(days=14))]
                if not future_incidents.empty:
                    first_incident = future_incidents.iloc[0]
                    lead_time = (first_incident['dt'] - s['dt']).days
                    sequences.append({
                        'Location': city,
                        'Sign_ID': s['GLOBALEVENTID'],
                        'Incident_ID': first_incident['GLOBALEVENTID'],
                        'LeadTime': lead_time
                    })
        
        seq_df = pd.DataFrame(sequences)
        
        if seq_df.empty:
            print("No escalation sequences found with 14-day window.")
            return

        # 2. Golden Time Distribution
        print(f"Total escalation sequences 포착: {len(seq_df):,}")
        avg_lead = seq_df['LeadTime'].mean()
        median_lead = seq_df['LeadTime'].median()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(seq_df['LeadTime'], bins=range(1, 16), kde=True, color='gold')
        plt.axvline(avg_lead, color='red', linestyle='--', label=f'Avg: {avg_lead:.1f} days')
        plt.axvline(median_lead, color='green', linestyle='-', label=f'Median: {median_lead:.1f} days')
        plt.title('Golden Time Window: Duration from Sign to Incident')
        plt.xlabel('Lead Time (Days until escalation)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.savefig('golden_time_distribution.png')
        
        # 3. Geopolitical Hotspot Relevance
        top_escalation_cities = seq_df['Location'].value_counts().head(10)
        print("\n--- Top 10 Escalation Hotspots (Sign -> Incident) ---")
        print(top_escalation_cities)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_escalation_cities.values, y=top_escalation_cities.index, palette='magma')
        plt.title('Top 10 Geopolitical Hotspots by Escalation Frequency')
        plt.xlabel('Number of Escalation Sequences')
        plt.tight_layout()
        plt.savefig('hotspot_relevance.png')

        # 4. Strategic Match Check (Simple heuristic)
        strategic_targets = ['Tehran', 'Hormuz', 'Baghdad', 'Gaza', 'Jerusalem', 'Damascus', 'Isfahan', 'Bandar Abbas', 'Beirut']
        found_strategic = [c for c in top_escalation_cities.index if any(t in c for t in strategic_targets)]
        
        print("\n--- Geopolitical Relevance Audit ---")
        print(f"Strategic Targets Identified: {found_strategic}")
        
        summary_stats = {
            'Total_Sequences': len(seq_df),
            'Avg_LeadTime': avg_lead,
            'Strategic_Match_Count': len(found_strategic)
        }
        
        print(f"\nFinal Summary: Average Golden Time Window is {avg_lead:.1f} days.")

    except Exception as e:
        print(f"Error during golden time analysis: {e}")

if __name__ == "__main__":
    run_golden_time_analysis()
