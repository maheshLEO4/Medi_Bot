#!/bin/bash
echo "🚀 Running MediBot setup script..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if database needs to be built
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db)" ]; then
    echo "🏗️ Building Chroma database..."
    if python build_db.py; then
        echo "✅ Database built successfully!"
    else
        echo "❌ Database build failed!"
        exit 1
    fi
else
    echo "✅ Database already exists - skipping build"
fi

echo "🎉 Setup completed successfully!"