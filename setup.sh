#!/bin/bash
echo "🚀 Running MediBot setup script..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install Pinecone specific packages
echo "📦 Installing Pinecone dependencies..."
pip install pinecone-client langchain-pinecone

# Check if Pinecone API key is set
if [ -z "$PINECONE_API_KEY" ]; then
    echo "❌ ERROR: PINECONE_API_KEY environment variable not set!"
    echo "   Please set your Pinecone API key:"
    echo "   export PINECONE_API_KEY='your-api-key-here'"
    exit 1
fi

# Check if database needs to be built
echo "🔍 Checking Pinecone index status..."
if python -c "
import os
from pinecone import Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
indexes = [index.name for index in pc.list_indexes()]
print('existing' if 'medi-bot-medical' in indexes else 'missing')
" | grep -q "missing"; then
    echo "🏗️ Building Pinecone database..."
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