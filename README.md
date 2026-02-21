Airfare Marketplace
A modular airfare intelligence engine with historical storage, FX normalization, and ML-driven price forecasting.
This project demonstrates production-style architecture across:
Provider abstraction & external API integration
Canonical domain modeling
Deterministic deduplication
Historical data persistence
FX normalization with caching
ML inference integration
Clean separation of UI, business logic, and infrastructure
The goal is not just to search flights — but to design a scalable airfare intelligence system.

Problem
Modern airfare platforms:
Obscure historical pricing behavior
Personalize results without transparency
Provide no predictive guidance
Mix currency representations
Lack architectural modularity
This project explores how a transparent airfare engine can be built with:
Clear domain contracts
Interchangeable providers
Historical data persistence
Currency normalization
Leakage-safe ML integration
Extensible system boundaries

System Architecture
airfare_marketplace/
│
├── src/
│ ├── streamlit_app.py # UI layer
│ ├── core/
│ │ ├── models.py # Canonical Offer → Itinerary → Segment model
│ │ └── scoring.py # Recommendation scoring engine
│ │
│ ├── providers/
│ │ ├── base.py
│ │ ├── mock_provider.py
│ │ ├── csv_provider.py
│ │ └── amadeus_provider.py # Live OAuth2 integration
│ │
│ ├── services/
│ │ ├── amadeus_client.py # OAuth2 + API client
│ │ └── fx_rate_services.py # Daily FX conversion (EUR → USD)
│ │
│ ├── sqlite_history_store.py # Historical storage layer
│ ├── data_access.py
│ └── ml_price_model.py # ML inference utilities
│
├── data/
│ └── price_history.sqlite # Historical observations (local)
│
├── models/
│ └── price_drop_model.pkl # Serialized ML model
│
├── notebooks/ # ML experimentation
└── requirements.txt

Canonical Domain Model
Offer
├── origin / destination
├── departure_date / return_date
├── airline summary
├── price + currency
├── offer_signature (deterministic)
└── itineraries[]
└── segments[]

Each segment includes:
origin / destination
departure / arrival timestamps
carrier codes
flight number
aircraft code
This normalized structure enables:
Deterministic itinerary-level deduplication
Stable offer_signature
Clean UI rendering
ML-ready feature extraction
Storage abstraction

Providers
The system supports interchangeable providers:
1️⃣ MockProvider
Offline deterministic provider for testing.
2️⃣ CSVProvider
Local reproducible dataset provider.
3️⃣ AmadeusProvider (Live)
OAuth2 Client Credentials flow
Token caching
Flexible date expansion (±3 days)
Post-filtering for stop constraints
Itinerary-level deduplication
Stable itinerary signatures
All providers return normalized Offer objects.

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
