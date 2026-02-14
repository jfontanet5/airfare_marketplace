Airfare Marketplace
A modular airfare search and pricing intelligence system built to explore:
Provider abstraction
Real-time API integration (OAuth2)
Historical price tracking
Price-drop prediction using machine learning
Clean system design for extensibility
This project serves as a systems design and applied ML portfolio piece.

Problem
Most airfare search tools:
Personalize prices based on cookies and location
Obscure price variability over time
Do not expose historical trends
Offer no predictive guidance on when to buy
This project explores how a transparent airfare engine could be architected with:
Multiple interchangeable data providers
Historical fare tracking
ML-driven price-drop prediction
Clear separation between UI, business logic, and integrations

System Architecture
airfare_marketplace/
│
├── src/
│ ├── streamlit_app.py # UI layer
│ ├── core/ # Domain models
│ ├── providers/ # Data source abstraction
│ ├── services/ # External integrations
│ ├── ml_price_model.py # ML training + inference
│ └── engine.py # Scoring + recommendation logic
│
├── notebooks/ # ML experimentation
├── data/ # Local data (ignored)
└── requirements.txt
Architectural Principles
Provider pattern for data source abstraction
Dataclass-based domain models for normalized offers
OAuth2 client credentials flow for secure API integration
Environment-based secret management
Separation of concerns between UI, engine, and external services

Providers
The system supports interchangeable providers:

1. MockProvider
   Offline demonstration mode for deterministic testing.
2. CSVProvider
   Local dataset-driven provider for development and reproducibility.
3. AmadeusProvider
   Live integration with Amadeus Flight Offers API:
   OAuth2 token exchange
   Token caching
   Flexible-date multi-query logic
   Post-filtering for stop constraints
   Providers return normalized Offer objects.
   A bridge layer converts them into legacy dictionaries for scoring compatibility.

Machine Learning
The ML module predicts the probability of a fare dropping within the next 7 days.
Current implementation:
Synthetic training dataset generator
Random Forest classifier
Real-time inference helper
UI integration for buy vs wait suggestion
Planned upgrade:
Train on real collected historical fare data

Key Engineering Decisions
Decoupled provider abstraction enables future integration with additional APIs or scraping pipelines without modifying UI logic.
Token handling is centralized in amadeus_client.py.
Flexible-date search is implemented as controlled multi-call expansion.
Raw API payloads are intentionally removed from UI rendering to prevent data leakage and maintain clarity.
.env secrets are excluded from version control.

Running Locally
git clone https://github.com/jfontanet5/airfare_marketplace.git
cd airfare_marketplace

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run src/streamlit_app.py

Enabling Live API
Create a .env file in the project root:
AMADEUS_ENV=test
AMADEUS_CLIENT_ID=your_api_key
AMADEUS_CLIENT_SECRET=your_api_secret
If credentials are not configured, the app can still run using Mock or CSV providers.

Roadmap
Persist live searches into structured historical storage
Replace synthetic ML data with real collected fare data
Introduce caching layer for live API calls
Expand multi-region request simulation
Containerize deployment
Add model retraining pipeline

Author
Julio Fontanet
Data Scientist
