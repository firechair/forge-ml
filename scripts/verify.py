"""
ForgeML Installation Verification Script

Checks that all components are properly installed and working.
"""

import sys
import subprocess
from pathlib import Path


def check(name, test_func):
    """Run a check and print result."""
    try:
        test_func()
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False


def check_python_version():
    """Check Python version is 3.10+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        raise RuntimeError(f"Python {version.major}.{version.minor} found, need 3.10+")


def check_imports():
    """Check core dependencies can be imported."""
    import torch  # noqa: F401
    import yaml  # noqa: F401
    import mlflow  # noqa: F401
    import typer  # noqa: F401
    import fastapi  # noqa: F401


def check_cli():
    """Check mlfactory CLI is available."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.main", "--help"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("CLI not working")


def check_templates():
    """Check templates exist."""
    templates_dir = Path(__file__).parent.parent / "templates"
    templates = list(templates_dir.glob("*/"))

    if not templates:
        raise RuntimeError("No templates found")

    template_names = [t.name for t in templates if t.is_dir()]
    if not template_names:
        raise RuntimeError("No templates found")


def check_docker():
    """Check if Docker is running (optional)."""
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except Exception:
        pass
    return False


def main():
    """Run all verification checks."""
    print("ForgeML Installation Verification")
    print("=" * 40)
    print()

    checks = [
        ("Python 3.10+", check_python_version),
        ("Core dependencies", check_imports),
        ("CLI command", check_cli),
        ("Templates", check_templates),
    ]

    passed = 0
    failed = 0

    for name, func in checks:
        if check(name, func):
            passed += 1
        else:
            failed += 1

    # Optional checks
    print()
    print("Optional Components:")
    print("-" * 40)

    docker_running = check_docker()
    if docker_running:
        print("✓ Docker (running)")
    else:
        print("⚠ Docker (not running or not installed)")

    # Summary
    print()
    print("=" * 40)
    if failed == 0:
        print(f"✅ All checks passed ({passed}/{len(checks)})")
        print()
        print("ForgeML is ready to use!")
        return 0
    else:
        print(f"❌ Some checks failed ({failed}/{len(checks)} failed)")
        print()
        print("Please fix the issues above before using ForgeML.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
