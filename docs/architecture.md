# ForgeML Architecture

This document provides a comprehensive overview of the ForgeML system architecture, component design, and data flow.

---

## Table of Contents

- [System Overview](#system-overview)
- [High-Level Architecture](#high-level-architecture)
- [Core Components](#core-components)
- [Template Architecture](#template-architecture)
- [Infrastructure Layer](#infrastructure-layer)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Patterns](#design-patterns)

---

## System Overview

ForgeML is a modular ML project scaffolding system that follows a layered architecture pattern. It separates concerns between CLI orchestration, project generation, infrastructure management, and runtime execution.

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                          │
│                    (CLI Interface)                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ForgeML CLI Engine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     init     │  │    train     │  │    serve     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Template Repository                       │
│  ┌───────────────────┐         ┌───────────────────┐       │
│  │   Sentiment       │         │   Time Series     │       │
│  │   Analysis        │         │   Forecasting     │       │
│  └───────────────────┘         └───────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Infrastructure Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  MLflow  │  │   DVC    │  │  Docker  │  │ FastAPI  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## High-Level Architecture

### Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        Developer Workflow                        │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   ForgeML CLI Tool     │
                    │   (cli/main.py)        │
                    └────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │   init   │    │  train   │    │  serve   │
        └──────────┘    └──────────┘    └──────────┘
                │               │               │
                ▼               ▼               ▼
        ┌──────────────────────────────────────────┐
        │      Template Rendering Engine           │
        │  - Jinja2 templating                     │
        │  - Configuration management              │
        │  - File generation                       │
        └──────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  Generated    │  │   Training    │  │   Serving     │
    │  Project      │  │   Pipeline    │  │   API         │
    │  Structure    │  │   (train.py)  │  │   (serve.py)  │
    └───────────────┘  └───────────────┘  └───────────────┘
                                │               │
                ┌───────────────┼───────────────┘
                ▼               ▼
        ┌────────────────────────────────┐
        │   Infrastructure Services      │
        │  - MLflow (tracking)           │
        │  - DVC (data versioning)       │
        │  - Docker (containerization)   │
        └────────────────────────────────┘
```

---

## Core Components

### 1. CLI Engine

**Location:** `cli/`

The CLI engine is the entry point for all user interactions, built using Click framework.

```
cli/
├── __init__.py
├── main.py              # CLI entry point
├── commands/
│   ├── init.py         # Project initialization
│   ├── train.py        # Training orchestration
│   └── serve.py        # Model serving
├── utils/
│   ├── template.py     # Template rendering
│   ├── config.py       # Configuration management
│   └── validators.py   # Input validation
└── exceptions.py       # Custom exceptions
```

**Responsibilities:**
- Parse user commands and arguments
- Validate inputs
- Orchestrate template rendering
- Manage project lifecycle

**Key Design Decisions:**
- Click framework for intuitive CLI interface
- Plugin architecture for extensibility
- Command pattern for operation isolation

### 2. Template System

**Location:** `templates/`

Templates are self-contained ML project blueprints with standardized structure.

```
templates/
├── sentiment/
│   ├── config.yaml          # Configuration schema
│   ├── train.py            # Training entry point
│   ├── model.py            # Model definition
│   ├── serve.py            # API endpoint
│   ├── requirements.txt    # Dependencies
│   ├── Dockerfile          # Container config
│   ├── tests/              # Test suite
│   └── README.md           # Documentation
└── timeseries/
    └── (same structure)
```

**Template Architecture Pattern:**

```
┌─────────────────────────────────────────────────────┐
│              Template Structure                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐         ┌──────────────┐        │
│  │ config.yaml  │────────▶│   train.py   │        │
│  │              │         │              │        │
│  │ - Model cfg  │         │ - Data load  │        │
│  │ - Training   │         │ - Training   │        │
│  │ - MLflow     │         │ - Evaluation │        │
│  └──────────────┘         └──────────────┘        │
│                                  │                 │
│                                  ▼                 │
│  ┌──────────────┐         ┌──────────────┐        │
│  │   model.py   │◀────────│   MLflow     │        │
│  │              │         │   Tracking   │        │
│  │ - Model def  │         │              │        │
│  │ - Forward    │         │ - Metrics    │        │
│  │ - Inference  │         │ - Artifacts  │        │
│  └──────────────┘         └──────────────┘        │
│        │                                           │
│        ▼                                           │
│  ┌──────────────┐         ┌──────────────┐        │
│  │   serve.py   │────────▶│   FastAPI    │        │
│  │              │         │   Endpoints  │        │
│  │ - Load model │         │              │        │
│  │ - Predict    │         │ - /predict   │        │
│  │ - Validate   │         │ - /health    │        │
│  └──────────────┘         └──────────────┘        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 3. Infrastructure Layer

**Components:**

#### MLflow (Experiment Tracking)

```
┌────────────────────────────────────────────────┐
│            MLflow Architecture                 │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────────────────────────────────┐    │
│  │         MLflow Server                │    │
│  │      (Docker Container)              │    │
│  │                                      │    │
│  │  ┌────────────┐  ┌────────────┐    │    │
│  │  │ Tracking   │  │   Model    │    │    │
│  │  │  Server    │  │  Registry  │    │    │
│  │  └────────────┘  └────────────┘    │    │
│  │         │               │           │    │
│  │         ▼               ▼           │    │
│  │  ┌────────────────────────────┐    │    │
│  │  │   Backend Store            │    │    │
│  │  │   (PostgreSQL for teams)   │    │    │
│  │  │   (SQLite for local)       │    │    │
│  │  └────────────────────────────┘    │    │
│  │         │                           │    │
│  │         ▼                           │    │
│  │  ┌────────────────────────────┐    │    │
│  │  │   Artifact Store           │    │    │
│  │  │   (Local filesystem)       │    │    │
│  │  └────────────────────────────┘    │    │
│  └──────────────────────────────────────┘    │
│                   ▲                           │
│                   │                           │
│         ┌─────────┴─────────┐                │
│         │                   │                │
│    ┌────────┐          ┌────────┐           │
│    │Training│          │Serving │           │
│    │Scripts │          │  API   │           │
│    └────────┘          └────────┘           │
│                                              │
└────────────────────────────────────────────────┘
```

#### Docker Infrastructure

```
docker-compose.yml
├── MLflow Server (port 5000)
│   ├── Backend: SQLite
│   └── Artifacts: ./mlartifacts
│
docker-compose-team.yml
├── MLflow Server (port 5000)
├── PostgreSQL (port 5432)
│   ├── User: mlflow
│   ├── Database: mlflow
│   └── Persistent volume
└── Network: mlflow-network
```

#### DVC (Data Version Control)

```
┌────────────────────────────────────────────┐
│          DVC Architecture                  │
├────────────────────────────────────────────┤
│                                            │
│  Project Root                              │
│  ├── .dvc/                                │
│  │   ├── config         # DVC config      │
│  │   └── cache/         # Local cache     │
│  │                                        │
│  ├── data.dvc           # Data pointers   │
│  └── models.dvc         # Model pointers  │
│                                            │
│              │                             │
│              ▼                             │
│  ┌─────────────────────────────┐         │
│  │    Remote Storage           │         │
│  │  - S3 / GCS / Azure         │         │
│  │  - Shared team access       │         │
│  └─────────────────────────────┘         │
│                                            │
└────────────────────────────────────────────┘
```

---

## Template Architecture

### Sentiment Analysis Template

```
┌───────────────────────────────────────────────────────────┐
│         Sentiment Analysis Pipeline Architecture         │
└───────────────────────────────────────────────────────────┘

Input: Text Data
     │
     ▼
┌─────────────────────┐
│  Data Preprocessing │
│  - Tokenization     │
│  - Truncation       │
│  - Padding          │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Model (Transformer)│
│  - BERT/DistilBERT  │
│  - RoBERTa          │
│  - Fine-tuning      │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Classification     │
│  - Softmax layer    │
│  - Binary/Multi     │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Output             │
│  - Sentiment label  │
│  - Confidence score │
└─────────────────────┘
```

**Model Architecture:**

```python
TransformerModel
├── Embedding Layer (from pretrained)
├── Transformer Layers (12-24 layers)
│   ├── Multi-head Self-Attention
│   ├── Feed Forward Network
│   └── Layer Normalization
├── Pooling Layer (CLS token)
└── Classification Head
    ├── Linear Layer
    ├── Dropout (0.1)
    └── Softmax Activation
```

### Time Series Forecasting Template

```
┌───────────────────────────────────────────────────────────┐
│        Time Series Forecasting Architecture              │
└───────────────────────────────────────────────────────────┘

Input: Historical Sequence
     │
     ▼
┌─────────────────────┐
│  Preprocessing      │
│  - Normalization    │
│  - Windowing        │
│  - Feature scaling  │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  LSTM Encoder       │
│  - Sequence input   │
│  - Hidden states    │
│  - Context vector   │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  LSTM Decoder       │
│  - Future prediction│
│  - Multi-step ahead │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Output Layer       │
│  - Linear mapping   │
│  - Denormalization  │
└─────────────────────┘
     │
     ▼
Output: Forecast Values
```

**LSTM Architecture:**

```
Input Shape: (batch_size, sequence_length, n_features)
     │
     ▼
┌──────────────────────┐
│  LSTM Layer 1        │
│  - Hidden: 128       │
│  - Dropout: 0.2      │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  LSTM Layer 2        │
│  - Hidden: 64        │
│  - Dropout: 0.2      │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Dense Layer         │
│  - Units: 32         │
│  - Activation: ReLU  │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│  Output Layer        │
│  - Units: horizon    │
└──────────────────────┘
```

---

## Data Flow

### Project Creation Flow

```
User: mlfactory init sentiment --name my-project
                        │
                        ▼
            ┌───────────────────────┐
            │  CLI Parser (Click)   │
            │  - Parse arguments    │
            │  - Validate inputs    │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Template Loader      │
            │  - Find template dir  │
            │  - Load config schema │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Project Generator    │
            │  - Create directory   │
            │  - Copy template      │
            │  - Render configs     │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Post-processing      │
            │  - Initialize git     │
            │  - Setup DVC          │
            │  - Create README      │
            └───────────────────────┘
                        │
                        ▼
                  Project Ready!
```

### Training Flow

```
User: python train.py
            │
            ▼
┌───────────────────────┐
│  Load Configuration   │
│  - config.yaml        │
│  - MLflow URI         │
│  - Model params       │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Initialize MLflow    │
│  - Start experiment   │
│  - Log parameters     │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Load Dataset         │
│  - HuggingFace        │
│  - Custom CSV         │
│  - Preprocessing      │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Initialize Model     │
│  - Load pretrained    │
│  - Configure layers   │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Training Loop        │
│  For each epoch:      │
│    - Forward pass     │
│    - Loss calculation │
│    - Backward pass    │
│    - Optimizer step   │
│    - Log metrics      │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Evaluation           │
│  - Test set metrics   │
│  - Confusion matrix   │
│  - Log to MLflow      │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Save Model           │
│  - Best checkpoint    │
│  - MLflow artifact    │
│  - Local directory    │
└───────────────────────┘
```

### Serving Flow

```
User: mlfactory serve
            │
            ▼
┌───────────────────────┐
│  FastAPI Application  │
│  - Load serve.py      │
│  - Initialize app     │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Load Model           │
│  - From best_model/   │
│  - Or from MLflow     │
│  - Initialize weights │
└───────────────────────┘
            │
            ▼
┌───────────────────────┐
│  Start Server         │
│  - Bind to port       │
│  - Uvicorn ASGI       │
│  - Listen requests    │
└───────────────────────┘
            │
            ▼
┌─────────────────────────────────────┐
│  API Endpoints Ready                │
│  ┌──────────────────────────────┐  │
│  │ POST /predict                │  │
│  │ - Receive input              │  │
│  │ - Preprocess                 │  │
│  │ - Model inference            │  │
│  │ - Return prediction          │  │
│  └──────────────────────────────┘  │
│  ┌──────────────────────────────┐  │
│  │ GET /health                  │  │
│  │ - Check model loaded         │  │
│  │ - Return status              │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## Technology Stack

### Core Technologies

```
┌─────────────────────────────────────────────────────────┐
│                 Technology Stack                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  CLI & Orchestration                                    │
│  ├── Python 3.10+           (Runtime)                  │
│  ├── Click                  (CLI framework)            │
│  └── Jinja2                 (Templating)               │
│                                                         │
│  Machine Learning                                       │
│  ├── PyTorch                (Deep learning framework)  │
│  ├── Transformers           (HuggingFace models)       │
│  ├── scikit-learn           (Metrics, preprocessing)   │
│  └── NumPy/Pandas           (Data manipulation)        │
│                                                         │
│  Experiment Tracking                                    │
│  ├── MLflow                 (Tracking & registry)      │
│  └── TensorBoard            (Visualization)            │
│                                                         │
│  Model Serving                                          │
│  ├── FastAPI                (Web framework)            │
│  ├── Uvicorn                (ASGI server)              │
│  └── Pydantic               (Data validation)          │
│                                                         │
│  Data Management                                        │
│  ├── DVC                    (Data version control)     │
│  └── HuggingFace Datasets   (Dataset hub)              │
│                                                         │
│  Infrastructure                                         │
│  ├── Docker                 (Containerization)         │
│  ├── Docker Compose         (Multi-container)          │
│  └── PostgreSQL             (MLflow backend)           │
│                                                         │
│  Development Tools                                      │
│  ├── pytest                 (Testing)                  │
│  ├── black                  (Code formatting)          │
│  ├── flake8                 (Linting)                  │
│  └── mypy                   (Type checking)            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Design Patterns

### 1. Template Pattern

ForgeML uses the Template Method pattern for project generation:

```python
class BaseTemplate:
    def generate_project(self, config):
        self.validate_config(config)      # Hook
        self.create_structure()           # Template method
        self.render_files(config)         # Hook
        self.post_process()               # Hook

class SentimentTemplate(BaseTemplate):
    def validate_config(self, config):
        # Sentiment-specific validation
        pass
```

### 2. Strategy Pattern

Model serving uses Strategy pattern for different model types:

```python
class ModelServer:
    def __init__(self, model_loader: ModelLoaderStrategy):
        self.loader = model_loader

    def load_model(self):
        return self.loader.load()

class TransformerLoader(ModelLoaderStrategy):
    def load(self):
        return AutoModelForSequenceClassification.from_pretrained(...)
```

### 3. Factory Pattern

Template creation uses Factory pattern:

```python
class TemplateFactory:
    @staticmethod
    def create_template(template_type: str) -> BaseTemplate:
        if template_type == "sentiment":
            return SentimentTemplate()
        elif template_type == "timeseries":
            return TimeSeriesTemplate()
        raise ValueError(f"Unknown template: {template_type}")
```

### 4. Observer Pattern

MLflow tracking implements Observer pattern for logging:

```python
class MLflowObserver:
    def on_epoch_end(self, epoch, metrics):
        mlflow.log_metrics(metrics, step=epoch)

    def on_training_end(self, model, metrics):
        mlflow.log_artifact(model_path)
```

---

## Scalability Considerations

### Horizontal Scaling

```
┌────────────────────────────────────────────────────┐
│          Production Deployment Architecture        │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────┐        │
│  │         Load Balancer                │        │
│  │         (nginx/ALB)                  │        │
│  └──────────────────────────────────────┘        │
│                    │                              │
│       ┌────────────┼────────────┐                │
│       ▼            ▼            ▼                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │ API     │  │ API     │  │ API     │          │
│  │ Server 1│  │ Server 2│  │ Server N│          │
│  └─────────┘  └─────────┘  └─────────┘          │
│       │            │            │                │
│       └────────────┼────────────┘                │
│                    ▼                              │
│       ┌─────────────────────────┐                │
│       │   Shared Model Storage  │                │
│       │   (S3/GCS/Azure Blob)   │                │
│       └─────────────────────────┘                │
│                                                    │
└────────────────────────────────────────────────────┘
```

### Monitoring Architecture

```
Generated Project
       │
       ├── FastAPI Metrics
       │   └── Prometheus endpoint (/metrics)
       │
       ├── MLflow Tracking
       │   ├── Training metrics
       │   └── Model performance
       │
       └── Application Logs
           └── Structured JSON logging
                   │
                   ▼
           ┌──────────────┐
           │  Monitoring  │
           │   Stack      │
           │              │
           │ - Prometheus │
           │ - Grafana    │
           │ - ELK Stack  │
           └──────────────┘
```

---

## Security Architecture

```
┌────────────────────────────────────────────┐
│        Security Layers                     │
├────────────────────────────────────────────┤
│                                            │
│  API Layer                                 │
│  ├── HTTPS/TLS encryption                 │
│  ├── API key authentication (optional)    │
│  └── Rate limiting                         │
│                                            │
│  Application Layer                         │
│  ├── Input validation (Pydantic)          │
│  ├── SQL injection prevention             │
│  └── XSS protection                        │
│                                            │
│  Infrastructure Layer                      │
│  ├── Docker isolation                      │
│  ├── Network segmentation                  │
│  └── Secret management (.env)              │
│                                            │
│  Data Layer                                │
│  ├── Encrypted storage (at rest)          │
│  └── Access control (IAM)                  │
│                                            │
└────────────────────────────────────────────┘
```

---

## Extension Points

ForgeML is designed to be extensible:

### 1. Custom Templates

```
templates/
└── your_custom_template/
    ├── config.yaml
    ├── train.py
    ├── model.py
    └── serve.py
```

### 2. Custom CLI Commands

```python
# cli/commands/custom.py
@click.command()
def custom_command():
    """Your custom command."""
    pass
```

### 3. Custom Model Architectures

```python
# template/model.py
class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your architecture
```

### 4. Custom Data Loaders

```python
# template/data.py
class CustomDataLoader:
    def load_data(self):
        # Your data loading logic
```

---

## Performance Optimization

### Training Optimization

```
┌────────────────────────────────────────┐
│   Training Performance Features        │
├────────────────────────────────────────┤
│                                        │
│  ✓ Mixed precision training (AMP)     │
│  ✓ Gradient accumulation              │
│  ✓ DataLoader num_workers tuning      │
│  ✓ Pin memory for GPU                 │
│  ✓ Gradient checkpointing              │
│  ✓ Model parallelism support           │
│                                        │
└────────────────────────────────────────┘
```

### Inference Optimization

```
┌────────────────────────────────────────┐
│   Serving Performance Features         │
├────────────────────────────────────────┤
│                                        │
│  ✓ Model quantization (INT8)          │
│  ✓ ONNX runtime support                │
│  ✓ Batch prediction API                │
│  ✓ Caching layer (Redis)               │
│  ✓ Async request handling              │
│  ✓ Connection pooling                  │
│                                        │
└────────────────────────────────────────┘
```

---

## Future Architecture Enhancements

### Planned Features

1. **Distributed Training**
   - Multi-GPU support via PyTorch DDP
   - Kubernetes job orchestration
   - Ray integration for distributed training

2. **AutoML Integration**
   - Hyperparameter optimization (Optuna)
   - Neural architecture search
   - Automated feature engineering

3. **Model Serving Enhancements**
   - TorchServe integration
   - TensorFlow Serving support
   - Triton Inference Server compatibility

4. **Observability**
   - OpenTelemetry integration
   - Distributed tracing
   - Performance profiling

---

## Conclusion

ForgeML's architecture is designed with the following principles:

- **Modularity**: Each component is independent and replaceable
- **Extensibility**: Easy to add new templates and features
- **Scalability**: Production-ready from day one
- **Maintainability**: Clear separation of concerns
- **Best Practices**: Industry-standard patterns and tools

This architecture enables rapid ML project development while maintaining production-grade quality and reliability.

---

**For questions or contributions, see [CONTRIBUTING.md](../CONTRIBUTING.md)**
