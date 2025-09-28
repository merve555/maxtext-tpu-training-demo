#!/bin/bash

# MaxText TPU Training Demo Configuration Script
# This script updates all Kubernetes YAML files with your project-specific values

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if config file exists
CONFIG_FILE="config/config.env"
EXAMPLE_CONFIG="config/config.env.example"

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    print_status "Please copy the example configuration file:"
    echo "  cp $EXAMPLE_CONFIG $CONFIG_FILE"
    echo "  # Edit $CONFIG_FILE with your actual values"
    exit 1
fi

# Load configuration
print_status "Loading configuration from $CONFIG_FILE"
source "$CONFIG_FILE"

# Validate required variables
REQUIRED_VARS=(
    "PROJECT_ID"
    "REGION" 
    "REPO_NAME"
    "HUGGINGFACE_TOKEN"
    "DOCKER_REGISTRY_URL"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        print_error "Required variable $var is not set in $CONFIG_FILE"
        exit 1
    fi
done

# Validate HuggingFace token format
if [[ ! "$HUGGINGFACE_TOKEN" =~ ^hf_[A-Za-z0-9_]{34,}$ ]]; then
    print_warning "HuggingFace token format looks incorrect. Expected format: hf_..."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Generate derived values
ARTIFACTS_BUCKET="${PROJECT_ID}-${ARTIFACTS_BUCKET_SUFFIX}"
DATASETS_BUCKET="${PROJECT_ID}-${DATASETS_BUCKET_SUFFIX}"
MAXTEXT_IMAGE="${DOCKER_REGISTRY_URL}/${PROJECT_ID}/${REPO_NAME}/maxtext-trainer:${MAXTEXT_IMAGE_TAG}"
VLLM_IMAGE="${DOCKER_REGISTRY_URL}/${PROJECT_ID}/${REPO_NAME}/vllm-server:${VLLM_IMAGE_TAG}"

print_status "Configuration Summary:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Artifacts Bucket: $ARTIFACTS_BUCKET"
echo "  Datasets Bucket: $DATASETS_BUCKET"
echo "  MaxText Image: $MAXTEXT_IMAGE"
echo "  vLLM Image: $VLLM_IMAGE"
echo "  HuggingFace Token: ${HUGGINGFACE_TOKEN:0:10}..."
echo

# Confirm before proceeding
read -p "Update all Kubernetes YAML files with these values? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Configuration cancelled."
    exit 0
fi

# Find all step YAML files
YAML_FILES=($(find kubernetes/ -name "step*.yaml" -type f))

if [ ${#YAML_FILES[@]} -eq 0 ]; then
    print_error "No step*.yaml files found in kubernetes/ directory"
    exit 1
fi

print_status "Found ${#YAML_FILES[@]} YAML files to update"

# Create backup directory
BACKUP_DIR="kubernetes/backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup original files
print_status "Creating backup in $BACKUP_DIR"
for file in "${YAML_FILES[@]}"; do
    cp "$file" "$BACKUP_DIR/"
done

# Update each YAML file
for file in "${YAML_FILES[@]}"; do
    print_status "Updating $(basename "$file")"
    
    # Create temporary file
    temp_file=$(mktemp)
    
    # Perform replacements
    sed -e "s/YOUR_PROJECT_ID/$PROJECT_ID/g" \
        -e "s/YOUR_REPO_NAME/$REPO_NAME/g" \
        -e "s/YOUR_HUGGINGFACE_TOKEN/$HUGGINGFACE_TOKEN/g" \
        -e "s|us-east5-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/maxtext-trainer:jax0.7.2-rev1|$MAXTEXT_IMAGE|g" \
        -e "s|us-east5-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/vllm-server:latest|$VLLM_IMAGE|g" \
        -e "s/YOUR_PROJECT_ID-maxtext-tpu-training-demo-artifacts/$ARTIFACTS_BUCKET/g" \
        -e "s/YOUR_PROJECT_ID-maxtext-tpu-training-demo-datasets/$DATASETS_BUCKET/g" \
        "$file" > "$temp_file"
    
    # Replace original file
    mv "$temp_file" "$file"
done

print_success "All YAML files updated successfully!"
print_status "Backup created in: $BACKUP_DIR"

# Verify updates
print_status "Verifying updates..."
VERIFICATION_FAILED=false

for file in "${YAML_FILES[@]}"; do
    if grep -q "YOUR_PROJECT_ID\|YOUR_REPO_NAME\|YOUR_HUGGINGFACE_TOKEN" "$file"; then
        print_warning "$(basename "$file") still contains placeholder values"
        VERIFICATION_FAILED=true
    fi
done

if [ "$VERIFICATION_FAILED" = true ]; then
    print_warning "Some files may not have been updated completely. Please check manually."
else
    print_success "All placeholder values have been replaced!"
fi

# Show next steps
echo
print_status "Next Steps:"
echo "1. Review the updated YAML files"
echo "2. Build and push your Docker images:"
echo "   docker build -f docker/Dockerfile.maxtext-clean -t $MAXTEXT_IMAGE ."
echo "   docker push $MAXTEXT_IMAGE"
echo "   docker build -f docker/Dockerfile.vllm -t $VLLM_IMAGE ."
echo "   docker push $VLLM_IMAGE"
echo "3. Deploy infrastructure with Terraform"
echo "4. Run the pipeline: ./scripts/deploy-full-pipeline.sh"
echo
print_success "Configuration complete!"
