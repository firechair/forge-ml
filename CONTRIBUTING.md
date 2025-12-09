# Contributing to ForgeML

Thank you for considering contributing to ForgeML! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Creating New Templates](#creating-new-templates)

---

## 📜 Code of Conduct

Be respectful, inclusive, and constructive. We're all here to learn and build together.

---

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/forge-ml.git
   cd forge-ml
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/firechair/forge-ml.git
   ```

---

## 💻 Development Setup

### Prerequisites

- Python 3.10 or higher
- Docker (optional, for running MLflow)
- Git

### Quick Setup (Recommended)

Use our automated setup script for one-command installation:

```bash
# Linux/macOS
./scripts/setup.sh

# Windows PowerShell
.\scripts\setup.ps1

# Or use Make
make setup
```

This will:
- Create a virtual environment
- Install all dependencies
- Initialize pre-commit hooks
- Start Docker infrastructure
- Run verification tests

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
mlfactory --help
```

### Using Make Commands

We provide a Makefile with common development tasks:

```bash
make help          # Show all available commands
make install       # Install ForgeML in dev mode
make test          # Run all tests
make test-coverage # Run tests with coverage report
make format        # Format code with black
make lint          # Run linters (flake8, mypy)
make check         # Run format check + lint + tests
make docker-up     # Start MLflow infrastructure
make clean         # Clean build artifacts
make verify        # Verify installation
```

### Verifying Your Setup

Before starting development, verify everything is working:

```bash
# Run verification script
python scripts/verify.py

# Or use Make
make verify

# Test the CLI
mlfactory --help
mlfactory init sentiment --name test-project

# Start infrastructure
make docker-up

# Check MLflow is accessible
curl http://localhost:5000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cli --cov=templates

# Run specific test file
pytest tests/test_cli.py

# Using Make
make test              # Run all tests
make test-coverage     # Run with coverage report
```

---

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/firechair/forge-ml/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (OS, Python version, etc.)
   - Relevant logs or screenshots

### Suggesting Features

1. Check [existing feature requests](https://github.com/firechair/forge-ml/issues?q=is%3Aissue+label%3Aenhancement)
2. Open a new issue with:
   - Clear description of the feature
   - Use case / problem it solves
   - Proposed implementation (if you have ideas)

### Contributing Code

1. **Find or create an issue** describing what you'll work on
2. **Comment on the issue** to let others know you're working on it
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** following our [coding standards](#coding-standards)
5. **Write tests** for your changes
6. **Run tests** to ensure everything works
7. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add feature: description of what you did"
   ```
8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Open a Pull Request** on GitHub

---

## 🔄 Pull Request Process

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features or bug fixes
3. **Run linters and formatters**:
   ```bash
   black .
   flake8 .
   mypy cli/
   ```
4. **Ensure all tests pass**:
   ```bash
   pytest
   ```
5. **Update CHANGELOG.md** (if applicable)
6. **Reference related issues** in your PR description
7. **Wait for review** - maintainers will review and provide feedback
8. **Make requested changes** if needed
9. **Celebrate** when your PR is merged! 🎉

---

## 📝 Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Formatter**: Black
- **Linter**: Flake8
- **Type checker**: MyPy

### Code Organization

- Keep functions small and focused
- Use descriptive variable names
- Add docstrings to all public functions and classes
- Use type hints for function signatures

### Example

```python
from typing import Optional

def create_project(
    template: str,
    project_name: str,
    output_dir: Optional[str] = None
) -> bool:
    """
    Create a new ML project from a template.
    
    Args:
        template: Name of the template to use
        project_name: Name for the new project
        output_dir: Directory to create project in (default: current directory)
        
    Returns:
        True if project created successfully, False otherwise
        
    Raises:
        ValueError: If template doesn't exist
    """
    # Implementation here
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

Examples:
```
feat: add image classification template
fix: resolve MLflow connection timeout
docs: update quick start guide
test: add tests for CLI init command
```

---

## 🧪 Testing

### Writing Tests

- Use `pytest` for all tests
- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`

### Test Structure

```python
import pytest
from cli.main import create_project

def test_create_project_success():
    """Test successful project creation."""
    result = create_project("sentiment", "test-proj")
    assert result is True

def test_create_project_invalid_template():
    """Test error handling for invalid template."""
    with pytest.raises(ValueError):
        create_project("nonexistent", "test-proj")
```

### Running Specific Tests

```bash
# Run tests matching a keyword
pytest -k "test_init"

# Run tests in a specific file
pytest tests/test_cli.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=cli --cov-report=html
```

---

## 🎨 Creating New Templates

Templates are one of the most valuable contributions! Here's how to create one:

### Template Structure

```
templates/your_template/
├── config.yaml          # Configuration file
├── train.py            # Training script
├── model.py            # Model definition
├── serve.py            # Serving endpoint
├── requirements.txt    # Dependencies
├── Dockerfile         # Docker configuration
├── README.md          # Template documentation
└── tests/             # Template-specific tests
    └── test_model.py
```

### Template Checklist

- [ ] All core files implemented (train.py, model.py, serve.py)
- [ ] config.yaml with sensible defaults
- [ ] requirements.txt with all dependencies
- [ ] README.md explaining how to use the template
- [ ] Tests covering main functionality
- [ ] Dockerfile that works
- [ ] Example dataset or data loading code
- [ ] MLflow integration for tracking
- [ ] Clear error messages

### Template Guidelines

1. **Use clear, descriptive variable names**
2. **Add comments explaining non-obvious code**
3. **Include example usage in README**
4. **Make it easy to customize** (via config.yaml)
5. **Follow the same structure** as existing templates
6. **Test end-to-end** before submitting

### Submitting a Template

1. Create the template in `templates/your_template/`
2. Add documentation in the template's README.md
3. Add entry to main README.md under "Available Templates"
4. Test it end-to-end
5. Open a PR with description of what the template does

---

## 🐛 Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test CLI Commands Locally

```bash
# Run CLI directly
python -m cli.main init sentiment --name test

# Or use the installed command
mlfactory init sentiment --name test
```

### Check MLflow Connection

```bash
# Verify MLflow is running
curl http://localhost:5000

# Check MLflow UI
open http://localhost:5000
```

---

## 📚 Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Code Style](https://black.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 💬 Questions?

- Open an issue for questions
- Join our discussions on GitHub
- Reach out to maintainers

---

## 🙏 Thank You!

Every contribution, no matter how small, makes ForgeML better for everyone. Thank you for being part of this community! ❤️

---

*Happy coding!* 🚀
