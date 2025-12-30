import pytest
from click.testing import CliRunner
from cli.main import app
from typer.testing import CliRunner as TyperRunner
from pathlib import Path
import shutil
import os


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    original_dir = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_dir)


class TestInitCommand:
    """Tests for the init command."""

    def test_init_creates_project(self, temp_dir):
        """Test that init creates a project successfully."""
        runner = TyperRunner()
        result = runner.invoke(app, ["init", "sentiment", "--name", "test-proj"])

        assert result.exit_code == 0
        assert "Created project" in result.output
        assert (temp_dir / "test-proj").exists()

    def test_init_creates_correct_structure(self, temp_dir):
        """Test that init creates all necessary files."""
        runner = TyperRunner()
        result = runner.invoke(app, ["init", "sentiment", "--name", "test-proj"])

        project_path = temp_dir / "test-proj"
        assert (project_path / "train.py").exists()
        assert (project_path / "model.py").exists()
        assert (project_path / "config.yaml").exists()
        assert (project_path / "requirements.txt").exists()
        assert (project_path / "README.md").exists()

    def test_init_fails_for_invalid_template(self, temp_dir):
        """Test that init fails gracefully for invalid template."""
        runner = TyperRunner()
        result = runner.invoke(app, ["init", "nonexistent", "--name", "test-proj"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_init_fails_if_directory_exists(self, temp_dir):
        """Test that init fails if project directory already exists."""
        # Create the directory first
        (temp_dir / "test-proj").mkdir()

        runner = TyperRunner()
        result = runner.invoke(app, ["init", "sentiment", "--name", "test-proj"])

        assert result.exit_code != 0
        assert "already exists" in result.output


class TestTrainCommand:
    """Tests for the train command."""

    def test_train_fails_without_config(self, temp_dir):
        """Test that train fails when config.yaml is missing."""
        runner = TyperRunner()
        result = runner.invoke(app, ["train"])

        assert result.exit_code != 0
        assert "not found" in result.output

    def test_train_fails_without_train_script(self, temp_dir):
        """Test that train fails when train.py is missing."""
        # Create a config file but no train.py
        (temp_dir / "config.yaml").write_text("test: true")

        runner = TyperRunner()
        result = runner.invoke(app, ["train"])

        assert result.exit_code != 0
        assert "train.py not found" in result.output


class TestServeCommand:
    """Tests for the serve command."""

    def test_serve_fails_without_serve_script(self, temp_dir):
        """Test that serve fails when serve.py is missing."""
        runner = TyperRunner()
        result = runner.invoke(app, ["serve"])

        assert result.exit_code != 0
        assert "serve.py not found" in result.output

    def test_serve_fails_without_model(self, temp_dir):
        """Test that serve fails when model is missing."""
        # Create serve.py but no model
        (temp_dir / "serve.py").write_text("# placeholder")

        runner = TyperRunner()
        result = runner.invoke(app, ["serve"])

        assert result.exit_code != 0
        assert "No trained model found" in result.output
