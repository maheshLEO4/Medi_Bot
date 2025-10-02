#!/bin/bash
# .streamlit/app.sh

echo "📦 Building database..."
python build_db.py

echo "🚀 Starting Streamlit..."
streamlit run app.py
