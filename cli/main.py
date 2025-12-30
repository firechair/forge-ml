import typer
from pathlib import Path
import shutil

app = typer.Typer()

@app.command()
def init(
    template: str,
    name: str = typer.Option("my-project", help="Project name")
):
    """
    Create a new ML project from a template.
    
    Example: mlfactory init sentiment --name my-sentiment-project
    """
    # 1. Check if template exists
    template_path = Path(__file__).parent.parent / "templates" / template
    
    if not template_path.exists():
        typer.echo(f"❌ Template '{template}' not found!")
        raise typer.Exit(1)
    
    # 2. Create new project folder
    project_path = Path.cwd() / name
    if project_path.exists():
        typer.echo(f"❌ Folder '{name}' already exists!")
        raise typer.Exit(1)
    
    # 3. Copy template files to new project
    shutil.copytree(template_path, project_path)
    
    typer.echo(f"✅ Created project '{name}' from template '{template}'")
    typer.echo(f"📁 Location: {project_path}")
    typer.echo(f"\nNext steps:")
    typer.echo(f"  cd {name}")
    typer.echo(f"  mlfactory train")

@app.command()
def train(
    config: str = typer.Option("config.yaml", help="Config file path"),
    experiment: str = typer.Option("default", help="Experiment name")
):
    """
    Train a model using the configuration file.
    
    Example: mlfactory train --config config.yaml --experiment baseline
    """
    import subprocess
    import sys
    
    # Check if config file exists
    config_path = Path(config)
    if not config_path.exists():
        typer.echo(f"❌ Config file '{config}' not found!")
        typer.echo(f"Make sure you're in a project directory with a config.yaml file.")
        raise typer.Exit(1)
    
    # Check if train.py exists
    train_script = Path.cwd() / "train.py"
    if not train_script.exists():
        typer.echo(f"❌ train.py not found in current directory!")
        typer.echo(f"Make sure you're in a project directory created with 'mlfactory init'")
        raise typer.Exit(1)
    
    typer.echo(f"🚀 Starting training...")
    typer.echo(f"📋 Config: {config}")
    typer.echo(f"🧪 Experiment: {experiment}")
    typer.echo("")
    
    # Run the training script
    try:
        result = subprocess.run(
            [sys.executable, "train.py", "--config", config, "--experiment", experiment],
            check=True
        )
        typer.echo(f"\n✅ Training completed successfully!")
    except subprocess.CalledProcessError as e:
        typer.echo(f"\n❌ Training failed with error code {e.returncode}")
        raise typer.Exit(e.returncode)
    except KeyboardInterrupt:
        typer.echo(f"\n⚠️  Training interrupted by user")
        raise typer.Exit(130)


@app.command()
def serve(
    model_uri: str = typer.Option(None, help="MLflow model URI (optional if best_model/ exists)"),
    port: int = typer.Option(8000, help="Port to serve on")
):
    """
    Serve a trained model via FastAPI.
    
    Example: mlfactory serve --port 8000
    Example: mlfactory serve --model-uri runs:/abc123def/model --port 8000
    """
    import subprocess
    import sys
    
    # Check if serve.py exists
    serve_script = Path.cwd() / "serve.py"
    if not serve_script.exists():
        typer.echo(f"❌ serve.py not found in current directory!")
        typer.echo(f"Make sure you're in a project directory created with 'mlfactory init'")
        raise typer.Exit(1)
    
    # Check if model exists (either best_model or MLflow URI)
    model_path = Path.cwd() / "best_model"
    if not model_path.exists() and not model_uri:
        typer.echo(f"❌ No trained model found!")
        typer.echo(f"")
        typer.echo(f"Options:")
        typer.echo(f"  1. Train a model first: python train.py")
        typer.echo(f"  2. Or specify an MLflow model URI: mlfactory serve --model-uri runs:/abc123/model")
        raise typer.Exit(1)
    
    if model_uri:
        typer.echo(f"📥 Loading model from MLflow URI: {model_uri}")
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            import os

            # Download model artifacts from MLflow to best_model directory
            typer.echo(f"🔄 Downloading model artifacts from MLflow...")

            # Remove existing best_model if it exists
            if model_path.exists():
                shutil.rmtree(model_path)

            # Parse the model URI
            # Supports: runs:/<run_id>/model or models:/<name>/<version>
            try:
                if model_uri.startswith("runs:/"):
                    # Extract run_id and path
                    parts = model_uri.replace("runs:/", "").split("/", 1)
                    run_id = parts[0]
                    artifact_path = parts[1] if len(parts) > 1 else "model"

                    # Download artifacts
                    client = MlflowClient()
                    model_artifacts_path = client.download_artifacts(run_id, artifact_path)

                    # Copy to best_model directory
                    shutil.copytree(model_artifacts_path, model_path)

                elif model_uri.startswith("models:/"):
                    # Use MLflow's model loading which handles registry URIs
                    model_artifacts_path = mlflow.artifacts.download_artifacts(model_uri)
                    shutil.copytree(model_artifacts_path, model_path)

                else:
                    raise ValueError(f"Unsupported model URI format. Use runs:/<run_id>/model or models:/<name>/<version>")

                typer.echo(f"✅ Model downloaded successfully to {model_path}")

            except Exception as uri_error:
                raise RuntimeError(f"Failed to parse or download model URI: {uri_error}")

        except ImportError:
            typer.echo(f"❌ MLflow not installed. Install with: pip install mlflow")
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"❌ Failed to load model from MLflow: {e}")
            typer.echo(f"")
            typer.echo(f"Troubleshooting:")
            typer.echo(f"  1. Verify the MLflow tracking server is running")
            typer.echo(f"  2. Check that the model URI is correct (runs:/<run_id>/model or models:/<name>/<version>)")
            typer.echo(f"  3. Ensure MLFLOW_TRACKING_URI environment variable is set if using remote server")
            typer.echo(f"  4. Check that the run exists and has model artifacts")
            raise typer.Exit(1)

    typer.echo(f"🌐 Starting FastAPI server on port {port}...")
    typer.echo(f"📦 Model: {model_path}")
    typer.echo(f"")
    typer.echo(f"API will be available at:")
    typer.echo(f"  • http://localhost:{port}")
    typer.echo(f"  • Docs: http://localhost:{port}/docs")
    typer.echo(f"  • Health: http://localhost:{port}/health")
    typer.echo(f"")
    typer.echo(f"Press CTRL+C to stop the server")
    typer.echo(f"")
    
    # Run the serving script
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "serve:app", 
             "--host", "0.0.0.0", "--port", str(port), "--reload"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        typer.echo(f"\n❌ Server failed with error code {e.returncode}")
        raise typer.Exit(e.returncode)
    except KeyboardInterrupt:
        typer.echo(f"\n✅ Server stopped")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()
