# Dota 2 Match Predictor 

📋 Executive Summary

This project implements an end-to-end Machine Learning pipeline to predict the outcome of Dota 2 ranked matches in the Immortal Bracket.

Leveraging the Stratz GraphQL API, I engineered a feature set based on Team Weighted Win Rates (WWR) and trained an XGBoost Classifier. The project evolved from a simple predictive model into a forensic audit of Data Engineering pipelines, as initial theoretical accuracy (96%) was flagged and traced back to a critical fault in the data ingestion logic.

🔍 The Forensic Audit: Why 96% was Wrong

During the initial validation phase, the model exhibited unusually high performance metrics (96.72% Accuracy). Instead of accepting this result, I conducted a rigorous audit of the pipeline.
1. The Anomaly (Pipeline Fault)

A code review of the ETL module revealed a logic error in the Pagination Loop (Data Ingestion).

    The Bug: The while loop used to fetch historical matches failed to correctly update the cursor (endDateTime).

    The Consequence: This caused Data Contamination, where specific matches were duplicated across both the Training and Testing datasets. The model was not "predicting" the winner; it was "memorizing" the duplicates.

2. The Correction (Engineering Fix)

I refactored the pagination logic to ensure strict Temporal Isolation (Train sets and Test sets representing distinct, non-overlapping time periods).

3. The Result (Real Baseline)

After fixing the ingestion pipeline, the accuracy stabilized to a realistic baseline of ~53%. This confirms that raw draft composition alone (without considering Player Skill/MMR, Lane Matchups, or Roles) provides only a marginal edge over a coin flip in the current meta.

⚙️ Technical Architecture
1. Data Acquisition (GraphQL ETL)

    Source: Stratz API (GraphQL).

    Query Logic: Optimized GraphQL queries to extract high-fidelity match data, filtering specifically for the Immortal Bracket to reduce noise from lower-skill variance.

    Volume: Processed 4,000+ matches with full hero composition details.

2. Feature Engineering

    Vectorization: Transformed JSON-nested hero IDs into team-based feature vectors.

    Weighted Win Rate (WWR): Calculated the aggregate win probability of the 5 selected heroes based on global meta statistics.

        Features: radiant_wwr, dire_wwr.

3. Modeling

    Algorithm: XGBoost Classifier (objective='binary:logistic').

    Validation: Implemented Chronological Splitting (Train on Past -> Predict Future) to simulate production constraints.

📊 Performance Comparative Matrix
Metric	Phase 1 (Ingestion Fault)	Phase 2 (Corrected Pipeline)
Accuracy	96.72% (Contaminated)	53.10% (Valid Baseline)
Precision	0.97	0.53
Recall	0.97	0.54
ROC-AUC	0.99	0.54

    Strategic Conclusion: The drop to ~53% accuracy serves as a validation of the corrected pipeline. It demonstrates that Team Weighted Win Rate (WWR) is insufficient as a standalone predictor for high-level play. Future iterations must incorporate Positional Logic (Pos 1-5) and Player MMR to improve predictive power.

🚀 Usage

1. Clone the repository
Bash

git clone https://github.com/yourusername/dota2-predictive-analytics.git
cd dota2-predictive-analytics

2. Install Dependencies
Bash

pip install -r requirements.txt

3. API Configuration Create a .env file in the root directory and add your Stratz API token:
Bash

STRATZ_API_KEY=your_token_here

4. Run the Pipeline
Bash

python main.py

🛠️ Roadmap & Improvements

    [x] ETL Pipeline: Migration from REST to GraphQL (Stratz).

    [x] Pipeline Audit: Fix pagination logic and remove data duplication.

    [ ] Advanced Logic: Add support for Hero Roles (Hard Carry vs. Support) to evaluate team balance.

    [ ] Deployment: Create a Streamlit web interface for real-time drafting.

### 👤 Author

**Francisco García**
*Operations Research Analyst & Data Practitioner*
[[LinkedIn Profile](https://www.linkedin.com/in/francisco-garcia-886195201)
