#!/bin/bash

# TPU Trillium Training Demo - Deployment Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-"us-east5"}
CLUSTER_NAME=${CLUSTER_NAME:-"merves-tpu-training-demo"}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if required tools are installed
    for tool in gcloud terraform kubectl podman; do
        if ! command -v $tool &> /dev/null; then
            print_error "$tool is not installed. Please install it first."
            exit 1
        fi
    done
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        print_error "PROJECT_ID environment variable is not set."
        print_error "Please set it with: export PROJECT_ID=your-gcp-project-id"
        exit 1
    fi
    
    # Check if HUGGINGFACE_TOKEN is set
    if [ -z "$HUGGINGFACE_TOKEN" ]; then
        print_error "HUGGINGFACE_TOKEN environment variable is not set."
        print_error "Please set it with: export HUGGINGFACE_TOKEN=your-hf-token"
        print_error "Get your token from: https://huggingface.co/settings/tokens"
        exit 1
    fi
    
    print_status "Prerequisites check passed!"
}

setup_gcp() {
    print_status "Setting up Google Cloud Project..."
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    print_status "Enabling required APIs..."
    gcloud services enable container.googleapis.com
    gcloud services enable tpu.googleapis.com
    gcloud services enable compute.googleapis.com
    gcloud services enable storage.googleapis.com
    gcloud services enable artifactregistry.googleapis.com
    
    print_status "Google Cloud setup completed!"
}

deploy_infrastructure() {
    print_status "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f terraform.tfvars ]; then
        print_status "Creating terraform.tfvars..."
        cat > terraform.tfvars << EOF
project_id   = "$PROJECT_ID"
region       = "$REGION"
cluster_name = "$CLUSTER_NAME"
EOF
    fi
    
    # Initialize and apply Terraform
    terraform init
    terraform plan
    terraform apply -auto-approve
    
    # Get outputs
    export GCS_BUCKET=$(terraform output -raw model_artifacts_bucket)
    export DATASETS_BUCKET=$(terraform output -raw datasets_bucket)
    export GKE_SA_EMAIL=$(terraform output -raw service_account_email)
    
    cd ..
    
    print_status "Infrastructure deployment completed!"
    print_status "GCS Bucket: $GCS_BUCKET"
    print_status "Service Account: $GKE_SA_EMAIL"
}

build_docker_images() {
    print_status "Building Docker images..."
    
    # Get Artifact Registry repository name from Terraform
    cd terraform
    export ARTIFACT_REGISTRY_REPO=$(terraform output -raw artifact_registry_repository)
    cd ..
    
    # Configure Docker authentication
    gcloud auth configure-docker $REGION-docker.pkg.dev
    gcloud auth configure-docker gcr.io
    
    # Build MaxText image
    print_status "Building MaxText training image..."
    podman build -f docker/Dockerfile.maxtext -t $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/maxtext-trainer:latest .
    podman push $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/maxtext-trainer:latest
    
    # Build vLLM image
    print_status "Building vLLM serving image..."
    podman build -f docker/Dockerfile.vllm -t $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/vllm-server:latest .
    podman push $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/vllm-server:latest
    
    print_status "Docker images built and pushed!"
}

configure_kubernetes() {
    print_status "Configuring Kubernetes..."
    
    # Get cluster credentials
    gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION
    
    # Get Terraform outputs for environment variables
    cd terraform
    
    # Check if terraform outputs exist
    if ! terraform output -raw model_artifacts_bucket >/dev/null 2>&1; then
        print_status "ERROR: Terraform outputs not found. Please run './scripts/deploy.sh infra' first."
        exit 1
    fi
    
    export GCS_BUCKET=$(terraform output -raw model_artifacts_bucket)
    export DATASETS_BUCKET=$(terraform output -raw datasets_bucket)
    export GKE_SA_EMAIL=$(terraform output -raw service_account_email)
    export ARTIFACT_REGISTRY_REPO=$(terraform output -raw artifact_registry_repository)
    
    # Validate that outputs are not empty
    if [ -z "$GCS_BUCKET" ] || [ -z "$DATASETS_BUCKET" ] || [ -z "$GKE_SA_EMAIL" ] || [ -z "$ARTIFACT_REGISTRY_REPO" ]; then
        print_status "ERROR: Terraform outputs are empty. Please run './scripts/deploy.sh infra' first."
        exit 1
    fi
    
    cd ..
    
    # Update Kubernetes manifests with actual values
    # Use a more robust approach with temporary files
    cp kubernetes/maxtext-training-job.yaml kubernetes/maxtext-training-job.yaml.tmp
    cp kubernetes/vllm-serving.yaml kubernetes/vllm-serving.yaml.tmp
    
    # Debug: Print variable values to identify problematic ones
    echo "Debug: PROJECT_ID='$PROJECT_ID'"
    echo "Debug: GCS_BUCKET='$GCS_BUCKET'"
    echo "Debug: DATASETS_BUCKET='$DATASETS_BUCKET'"
    echo "Debug: HUGGINGFACE_TOKEN='$HUGGINGFACE_TOKEN'"
    echo "Debug: GKE_SA_EMAIL='$GKE_SA_EMAIL'"
    echo "Debug: REGION='$REGION'"
    
    # Replace placeholders using a safer method with @ delimiter
    echo "Replacing PROJECT_ID_PLACEHOLDER..."
    sed -i.tmp "s#PROJECT_ID_PLACEHOLDER#$PROJECT_ID#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#GCS_BUCKET_PLACEHOLDER#$GCS_BUCKET#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#DATASETS_BUCKET_PLACEHOLDER#$DATASETS_BUCKET#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#HF_TOKEN_PLACEHOLDER#$HUGGINGFACE_TOKEN#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#GKE_SA_EMAIL_PLACEHOLDER#$GKE_SA_EMAIL#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#gcr.io/PROJECT_ID/maxtext-trainer:latest#$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/maxtext-kube:latest#g" kubernetes/maxtext-training-job.yaml.tmp
    
    sed -i.tmp "s#GCS_BUCKET_PLACEHOLDER#$GCS_BUCKET#g" kubernetes/vllm-serving.yaml.tmp
    sed -i.tmp "s#GKE_SA_EMAIL_PLACEHOLDER#$GKE_SA_EMAIL#g" kubernetes/vllm-serving.yaml.tmp
    sed -i.tmp "s#vllm/vllm-openai:latest#$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_REPO/vllm-server:latest#g" kubernetes/vllm-serving.yaml.tmp
    
    # Move the processed files back
    mv kubernetes/maxtext-training-job.yaml.tmp kubernetes/maxtext-training-job.yaml
    mv kubernetes/vllm-serving.yaml.tmp kubernetes/vllm-serving.yaml
    
    print_status "Kubernetes configuration completed!"
}

prepare_dataset() {
    print_status "Preparing Alpaca dataset..."
    
    # Get Terraform outputs
    cd terraform
    export DATASETS_BUCKET=$(terraform output -raw datasets_bucket)
    cd ..
    
    # Validate that outputs are not empty
    if [ -z "$DATASETS_BUCKET" ]; then
        print_status "ERROR: Terraform outputs are empty. Please run './scripts/deploy.sh infra' first."
        exit 1
    fi
    
    # Create and use a virtual environment for Python dependencies
    VENV_DIR="/tmp/tpu-demo-venv"
    if [ ! -d "$VENV_DIR" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv $VENV_DIR
    fi
    
    # Activate virtual environment and install dependencies
    print_status "Installing required Python packages..."
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install datasets google-cloud-storage tensorflow transformers array-record grain
    
    # Use a reasonable number of samples for demo
    NUM_SAMPLES=${NUM_SAMPLES:-500}
    
    print_status "Using ${NUM_SAMPLES} Alpaca examples for training"
    print_status "Uploading to GCS bucket: $DATASETS_BUCKET"
    
    # Run ArrayRecord data preparation locally
    python scripts/prepare_arrayrecord_data.py \
        --gcs_bucket $DATASETS_BUCKET \
        --gcs_path train_data.array_record \
        --num_samples $NUM_SAMPLES \
        --tokenizer_name google/gemma-2-27b
    
    # Deactivate virtual environment
    deactivate
    
    print_status "Dataset preparation completed!"
}

deploy_training_job() {
    print_status "Deploying training job..."
    
    # Get Terraform outputs
    cd terraform
    # Use PROJECT_ID from environment if available, otherwise get from terraform.tfvars
    if [ -z "$PROJECT_ID" ]; then
        export PROJECT_ID=$(grep '^project_id' terraform.tfvars | cut -d'"' -f2)
    fi
    export GCS_BUCKET=$(terraform output -raw model_artifacts_bucket)
    export DATASETS_BUCKET=$(terraform output -raw datasets_bucket)
    export GKE_SA_EMAIL=$(terraform output -raw service_account_email)
    export ARTIFACT_REGISTRY_REPO=$(terraform output -raw artifact_registry_repository)
    cd ..
    
    # Validate that outputs are not empty
    if [ -z "$PROJECT_ID" ] || [ -z "$GCS_BUCKET" ] || [ -z "$DATASETS_BUCKET" ] || [ -z "$GKE_SA_EMAIL" ] || [ -z "$ARTIFACT_REGISTRY_REPO" ]; then
        print_status "ERROR: Terraform outputs are empty. Please run './scripts/deploy.sh infra' first."
        exit 1
    fi
    
    print_status "Configuring training job with:"
    echo "Debug: PROJECT_ID='$PROJECT_ID'"
    echo "Debug: GCS_BUCKET='$GCS_BUCKET'"
    echo "Debug: DATASETS_BUCKET='$DATASETS_BUCKET'"
    echo "Debug: HUGGINGFACE_TOKEN='$HUGGINGFACE_TOKEN'"
    echo "Debug: GKE_SA_EMAIL='$GKE_SA_EMAIL'"
    echo "Debug: ARTIFACT_REGISTRY_REPO='$ARTIFACT_REGISTRY_REPO'"
    
    # Create a temporary file for the training job with substituted variables
    cp kubernetes/maxtext-training-job.yaml kubernetes/maxtext-training-job.yaml.tmp
    
    # Substitute variables in the training job YAML
    sed -i.tmp "s#PROJECT_ID_PLACEHOLDER#$PROJECT_ID#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#GCS_BUCKET_PLACEHOLDER#$GCS_BUCKET#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#DATASETS_BUCKET_PLACEHOLDER#$DATASETS_BUCKET#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#HF_TOKEN_PLACEHOLDER#$HUGGINGFACE_TOKEN#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#GKE_SA_EMAIL_PLACEHOLDER#$GKE_SA_EMAIL#g" kubernetes/maxtext-training-job.yaml.tmp
    sed -i.tmp "s#ARTIFACT_REGISTRY_REPO_PLACEHOLDER#$ARTIFACT_REGISTRY_REPO#g" kubernetes/maxtext-training-job.yaml.tmp
    
    # Apply the training job with substituted variables
    kubectl apply -f kubernetes/maxtext-training-job.yaml.tmp
    
    # Clean up temporary files
    rm kubernetes/maxtext-training-job.yaml.tmp kubernetes/maxtext-training-job.yaml.tmp.tmp
    
    print_status "Training job deployed!"
    print_status "Monitor with: kubectl logs -f job/gemma-2-27b-finetune"
}

deploy_serving() {
    print_status "Deploying vLLM serving..."
    
    # Apply the serving deployment
    kubectl apply -f kubernetes/vllm-serving.yaml
    
    print_status "vLLM serving deployed!"
    print_status "Check status with: kubectl get pods -l app=vllm-serving"
}

main() {
    print_status "Starting TPU Trillium Training Demo deployment..."
    
    case "${1:-all}" in
        "prereq")
            check_prerequisites
            ;;
        "gcp")
            check_prerequisites
            setup_gcp
            ;;
        "infra")
            check_prerequisites
            deploy_infrastructure
            ;;
        "images")
            check_prerequisites
            build_docker_images
            ;;
        "k8s")
            check_prerequisites
            configure_kubernetes
            ;;
        "data")
            check_prerequisites
            prepare_dataset
            ;;
        "train")
            check_prerequisites
            configure_kubernetes
            deploy_training_job
            ;;
        "serve")
            check_prerequisites
            deploy_serving
            ;;
        "all")
            check_prerequisites
            setup_gcp
            deploy_infrastructure
            build_docker_images
            configure_kubernetes
            prepare_dataset
            deploy_training_job
            print_status "Waiting for training to complete before deploying serving..."
            print_status "Monitor training with: kubectl logs -f job/gemma-2-27b-finetune"
            print_status "Once training is complete, run: ./scripts/deploy.sh serve"
            ;;
        *)
            echo "Usage: $0 [prereq|gcp|infra|images|k8s|data|train|serve|all]"
            echo "  prereq: Check prerequisites"
            echo "  gcp:    Setup Google Cloud Project"
            echo "  infra:  Deploy infrastructure with Terraform"
            echo "  images: Build and push Docker images"
            echo "  k8s:    Configure Kubernetes"
            echo "  data:   Prepare sample dataset"
            echo "  train:  Deploy training job"
            echo "  serve:  Deploy serving"
            echo "  all:    Run all steps (except serving)"
            exit 1
            ;;
    esac
    
    print_status "Deployment step completed successfully!"
}

main "$@"
