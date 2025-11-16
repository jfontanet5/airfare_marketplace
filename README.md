Airfare Marketplace (MVP)
A transparent and intelligent airfare search engine designed to uncover the actual lowest prices—without the inflation caused by cookies, repeated searches, or location-based pricing.
The project combines clean-session multi-region search, dynamic itinerary generation, and an upcoming ML price-prediction model to help travelers make smarter booking decisions.

Project Vision
Most flight tools quietly personalize prices. Frequent searches increase fare quotes. IP location changes what deals you are shown. Browsers store cookies that anchor your “willingness to pay.” As a result, travelers rarely see the true lowest fare.
This project takes a different approach.
Every search is executed as if it came from a fresh device, in multiple geographic regions, and without any stored history. The long-term goal is to build a system that not only compares one-way and roundtrip options across regions, but also learns fare behavior over time to predict price drops and recommend the best time to buy.

Current Capabilities
The app is built with Streamlit and already supports a flexible, data-driven search flow. Users can configure:
• One-way or roundtrip travel
• Departure and return dates
• Search mode: Optimal (mix of one-ways and roundtrips) or classic roundtrip
• Maximum stops
• Preferred airlines
• Multicity allowance
• Flexible dates (±3-day simulated window)
Behind the scenes, the engine generates simulated offers from multiple global regions and providers. It then evaluates the cheapest US fare versus the cheapest global fare, giving users a preview of how multi-region price scanning will behave when connected to real data.

Architecture Overview
The system is structured as a lightweight app with a modular core:
airfare_marketplace/
src/
streamlit_app.py # Main UI and search logic
notebooks/ # ML exploration and data analysis
data/ # Local, ignored dataset storage
requirements.txt
README.md

The backend is intentionally decoupled. As real data sources are added, the engine functions can be replaced with API calls, scrapers, or ML-driven pricing components without changing the UI contract.

Technology Stack
The app runs on Python with Streamlit as the user interface layer.
Pandas handles the table structures.
Future iterations will integrate external APIs, structured flight datasets, and machine learning models for prediction and recommendation.

Machine Learning Module
This project now includes a dedicated ML pipeline to predict airfare price drops.
Current ML Features

- Synthetic dataset generator (`make_synthetic_training_data`)
- Baseline classifier using scikit-learn (`train_baseline_model`)
- Inference helper for real-time predictions
- Notebook for experimentation (`notebooks/price_model_exploration.ipynb`)
- Future integration: “Chance of price drop in next 7 days” metric in UI

ML functionality is modular and lives in: src/ml_price_model.py, src/ml_price_model.py

Running the Project Locally
git clone https://github.com/jfontanet5/airfare_marketplace.git
cd airfare_marketplace

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run src/streamlit_app.py

Roadmap
The next steps focus on grounding the engine in real data and adding intelligence to itinerary generation:
• Integrate initial real flight datasets (API or scraped CSV)
• Build the first version of the multi-region request pipeline
• Implement true one-way combination logic for “Optimal” mode
• Compare classic roundtrip vs paired one-way pricing
• Track historical fares for trend analysis
• Train ML models for price-drop likelihood and booking window recommendations
• Deploy the application for public use

About the Project
Created by Julio Fontanet as a long-term initiative to rethink airfare transparency using data engineering, user-centric design, and applied machine learning.
This project is actively evolving and serves as both a technical portfolio piece and a foundation for more advanced airfare intelligence tools.
