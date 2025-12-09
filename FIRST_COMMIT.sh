#!/bin/bash
# ForgeML v0.1.0 - First Commit and Push to GitHub
# Repository: https://github.com/firechair/forge-ml

set -e  # Exit on error

echo "🚀 ForgeML v0.1.0 - First Release"
echo "=================================="
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "⚠️  Not a git repository. Initializing..."
    git init
    git branch -M master
fi

# Check if remote exists
if ! git remote get-url origin &> /dev/null; then
    echo "➕ Adding remote origin..."
    git remote add origin https://github.com/firechair/forge-ml.git
fi

echo "📋 Git status:"
git status --short
echo ""

# Add all files
echo "➕ Adding all files to git..."
git add .

echo ""
echo "📝 Creating initial commit..."
git commit -m "feat: Initial release of ForgeML v0.1.0

🎉 First public release of ForgeML - ML Project Factory

ForgeML is a comprehensive ML project scaffolding tool that helps developers
create production-ready machine learning projects in minutes.

## Features

### Core Functionality
- CLI tool with init, train, and serve commands
- Two production-ready templates (sentiment analysis + time series)
- MLflow integration for experiment tracking
- FastAPI model serving with REST API
- DVC integration for data versioning
- Docker deployment setup

### Templates
- **Sentiment Analysis**: Fine-tune BERT/DistilBERT/RoBERTa for text classification
- **Time Series Forecasting**: LSTM-based sequence-to-sequence predictions

### Cross-Platform Support
- Linux, macOS, and Windows fully supported
- Automated setup scripts for one-command installation
- Platform-specific documentation and guides

### Developer Experience
- Comprehensive documentation (2,800+ lines)
- 4 example workflows
- 33+ test functions across templates
- Pre-commit hooks for code quality
- GitHub Actions CI/CD pipelines
- Makefile with 30+ development commands

### Team Collaboration
- Shared MLflow server setup (PostgreSQL backend)
- DVC remote storage configuration
- Team workflow documentation
- Multi-developer examples

## What's Included

- \`cli/\` - Command-line interface implementation
- \`templates/sentiment/\` - NLP text classification template
- \`templates/timeseries/\` - Time series forecasting template
- \`infra/\` - Docker Compose infrastructure (local + team)
- \`examples/\` - 4 comprehensive example workflows
- \`docs/\` - Installation, quickstart, FAQ, team guides
- \`scripts/\` - Automated setup and verification scripts
- \`tests/\` - Integration test suite
- \`.github/\` - CI/CD pipelines and issue templates

## Quick Start

\`\`\`bash
# Clone and setup
git clone https://github.com/firechair/forge-ml.git
cd forge-ml
./scripts/setup.sh

# Create a project
mlfactory init sentiment --name my-project
cd my-project

# Train and serve
pip install -r requirements.txt
python train.py
python serve.py
\`\`\`

## Technical Stack

- Python 3.10+
- PyTorch with HuggingFace Transformers
- MLflow for experiment tracking
- FastAPI for API serving
- DVC for data versioning
- Docker for deployment
- pytest for testing

## Documentation

- README.md - Complete project overview
- CONTRIBUTING.md - Contributor guide
- CHANGELOG.md - Release history
- docs/ - Comprehensive guides
- PROJECT_REVIEW.md - Audit and verification
- QUICKSTART_VERIFICATION.md - Setup verification

## License

MIT License - See LICENSE file

---

Built with ❤️ for the ML community

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

echo ""
echo "✅ Commit created successfully!"
echo ""

# Show commit details
echo "📄 Commit details:"
git log -1 --stat
echo ""

# Ask for confirmation before pushing
echo "⚠️  Ready to push to GitHub?"
echo "   Repository: https://github.com/firechair/forge-ml"
echo "   Branch: master"
echo ""
read -p "Push to GitHub? (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Pushing to GitHub..."
    git push -u origin master

    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "🎉 ForgeML v0.1.0 is now live!"
    echo ""
    echo "Next steps:"
    echo "1. Visit: https://github.com/firechair/forge-ml"
    echo "2. Create a release: https://github.com/firechair/forge-ml/releases/new"
    echo "   - Tag: v0.1.0"
    echo "   - Title: ForgeML v0.1.0 - Initial Release"
    echo "   - Description: Copy from CHANGELOG.md"
    echo "3. Share your project with the community! 🚀"
    echo ""
else
    echo ""
    echo "⏸️  Push cancelled. You can push later with:"
    echo "   git push -u origin master"
    echo ""
fi
