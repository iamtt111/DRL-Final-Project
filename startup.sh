#!/bin/bash

# Startup script for dev workflow
# Fetches latest changes and loads handover context

echo "🚀 Starting development session..."
echo ""

# Pull latest changes from main branch
echo "📥 Fetching latest changes from main..."
git pull origin main

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if handover.md exists and display it
if [ -f "openspec/handover.md" ]; then
    echo "📋 Loading previous handover context:"
    echo ""
    cat openspec/handover.md
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
else
    echo "ℹ️  No previous handover context found. Starting fresh!"
    echo ""
fi

# Friendly instruction
echo "✨ Ready to code!"
echo ""
echo "💡 PRO TIP:"
echo "   Copy the handover context above and paste it to your AI assistant to plan next actions."
echo "   Usage: npx copilot chat --install-mcp && copilot dev:ending when done!"
echo ""
