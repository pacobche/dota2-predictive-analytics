import os
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost.plotting import plot_importance
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()

# --- CONSTANTS ---
OPENDOTA_URL = "https://api.opendota.com/api/publicMatches"
STRATZ_URL = "https://api.stratz.com/graphql"
MIN_RANK = 80  # Immortal Rank roughly
TARGET_GAME_VERSION = 181 # Manual override for patch consistency
MATCH_LIMIT = 4000

def get_weighted_win_rate(team_hero_ids, hero_stats_df):
    """
    Calculates the average Win Rate of a team weighted by the pick frequency 
    of each hero in the current meta.
    """
    # Filter stats for the specific heroes in the team
    team_data = hero_stats_df[hero_stats_df['id'].isin(team_hero_ids)]
    
    if team_data.empty:
        return 0.5 # Fallback to neutral probability if no data found
        
    win_rates = team_data['wr'].tolist()
    pick_rates = team_data['pick_count'].tolist()
    
    # Return weighted average
    return np.average(win_rates, weights=pick_rates)

def fetch_public_matches(limit=1000):
    """
    ETL Function: Extracts public matches from OpenDota with pagination.
    Includes duplicate handling to prevent data contamination.
    """
    print(f"--- Starting Data Ingestion (Target: {limit} matches) ---")
    
    all_matches = []
    last_match_id = None
    params = {'min_rank': MIN_RANK}

    while len(all_matches) < limit:
        if last_match_id:
            params['less_than_match_id'] = last_match_id
            
        try:
            response = requests.get(OPENDOTA_URL, params=params, timeout=10)
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                break
                
            batch = response.json()
            
            if not batch:
                print("No more matches returned by API.")
                break
                
            # Convert to DataFrame to filter
            df_batch = pd.DataFrame(batch)
            
            # Apply Filters: Ranked (7), All Pick (22), Min Duration (15m)
            mask = (df_batch['lobby_type'] == 7) & \
                   (df_batch['game_mode'] == 22) & \
                   (df_batch['duration'] >= 900)
            
            valid_batch = df_batch[mask].copy()
            
            # Append as list of dicts (faster than repeated concat)
            all_matches.extend(valid_batch.to_dict('records'))
            
            # Update cursor
            last_match_id = df_batch.iloc[-1]['match_id']
            print(f"Ingested {len(all_matches)} matches...")
            
        except Exception as e:
            print(f"Critical Loop Error: {e}")
            break

    # Final DataFrame Construction
    df = pd.DataFrame(all_matches)
    
    # --- ENGINEERING AUDIT FIX: Deduplication ---
    initial_len = len(df)
    df.drop_duplicates(subset=['match_id'], inplace=True)
    if len(df) < initial_len:
        print(f"⚠️ AUDIT WARNING: Removed {initial_len - len(df)} duplicate matches found during ingestion.")
    
    return df.head(limit) # Return strict limit

def fetch_hero_stats():
    """
    Fetches Hero Win Rates via Stratz GraphQL API.
    """
    token = os.getenv("STRATZ_API_KEY")
    if not token:
        raise ValueError("STRATZ_API_KEY not found in environment variables.")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "STRATZ_API_PYTHON",
        "Content-Type": "application/json"
    }

    query = """
    {
      heroStats {
        winGameVersion(bracketIds: [IMMORTAL]) {
          gameVersionId
          heroId
          matchCount
          winCount
        }
      }
    }
    """
    
    print("--- Fetching Hero Meta Stats (GraphQL) ---")
    response = requests.post(STRATZ_URL, json={'query': query}, headers=headers)
    
    if response.status_code != 200:
        raise ConnectionError(f"Stratz API Failed: {response.status_code}")
        
    data = response.json()['data']['heroStats']['winGameVersion']
    df = pd.DataFrame(data)
    
    # Filter by Game Version
    df = df[df['gameVersionId'] == TARGET_GAME_VERSION].copy()
    
    # Feature Engineering on Stats
    df['wr'] = df['winCount'] / df['matchCount']
    df.rename(columns={'heroId': 'id', 'matchCount': 'pick_count'}, inplace=True)
    
    return df

def train_and_evaluate(df):
    """
    Splits data temporally, trains XGBoost, and reports metrics.
    """
    print("--- Training Model (Temporal Split) ---")
    
    # Split Point (80% Newest is Test, 20% Oldest is Train - Correct logic for OpenDota order)
    # OpenDota returns Newest -> Oldest. 
    # Index 0 is NOW. Index N is PAST.
    # Train: Past (Tail). Test: Future (Head).
    split_point = int(len(df) * 0.2) # Adjusted logic: Top 20% is Test (New), Bottom 80% is Train (Old)
    
    # Let's verify sort order just to be robust
    df = df.sort_values('start_time', ascending=True) # Oldest first now
    split_index = int(len(df) * 0.8)
    
    X = df[['radiant_wwr', 'dire_wwr']]
    y = df['radiant_win'].astype(int)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Training Set: {len(X_train)} matches (Oldest)")
    print(f"Testing Set: {len(X_test)} matches (Newest)")

    # Model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model Results:")
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Plotting
    plot_importance(model)
    plt.title("XGBoost Feature Importance")
    plt.show()

def main():
    # 1. Ingestion
    df_matches = fetch_public_matches(limit=MATCH_LIMIT)
    
    if df_matches.empty:
        print("No matches fetched. Exiting.")
        return

    # 2. Meta Stats
    try:
        df_stats = fetch_hero_stats()
    except Exception as e:
        print(f"Error fetching stats: {e}")
        return

    # 3. Feature Engineering
    print("--- Calculating Team Weighted Win Rates ---")
    # Apply is slow, but acceptable for <10k rows
    df_matches['radiant_wwr'] = df_matches['radiant_team'].apply(lambda x: get_weighted_win_rate(x, df_stats))
    df_matches['dire_wwr'] = df_matches['dire_team'].apply(lambda x: get_weighted_win_rate(x, df_stats))

    # 4. Modeling
    train_and_evaluate(df_matches)

if __name__ == "__main__":
    main()