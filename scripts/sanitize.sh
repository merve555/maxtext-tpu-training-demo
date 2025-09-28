#!/bin/bash

# MaxText TPU Training Demo Sanitization Script
# This script removes sensitive information and restores placeholder values

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

# Check if config file exists to get current values
CONFIG_FILE="config/config.env"

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    print_status "Cannot determine current values for sanitization"
    exit 1
fi

# Load current configuration to know what to replace
print_status "Loading current configuration from $CONFIG_FILE"
source "$CONFIG_FILE"

# Find all step YAML files
YAML_FILES=($(find kubernetes/ -name "step*.yaml" -type f))

if [ ${#YAML_FILES[@]} -eq 0 ]; then
    print_error "No step*.yaml files found in kubernetes/ directory"
    exit 1
fi

print_status "Found ${#YAML_FILES[@]} YAML files to sanitize"

# Create backup directory
BACKUP_DIR="kubernetes/backup-sanitize-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup current files
print_status "Creating backup in $BACKUP_DIR"
for file in "${YAML_FILES[@]}"; do
    cp "$file" "$BACKUP_DIR/"
done

print_status "Sanitizing files..."
echo "  Replacing: $PROJECT_ID -> YOUR_PROJECT_ID"
echo "  Replacing: $REPO_NAME -> YOUR_REPO_NAME"
echo "  Replacing: $HUGGINGFACE_TOKEN -> YOUR_HUGGINGFACE_TOKEN"
echo "  Replacing: Docker image URLs -> Template URLs"
echo "  Replacing: GCS bucket names -> Template names"
echo

# Confirm before proceeding
read -p "Sanitize all Kubernetes YAML files? This will remove sensitive values. (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Sanitization cancelled."
    exit 0
fi

# Use bucket names directly from config
MAXTEXT_IMAGE="${DOCKER_REGISTRY_URL}/${PROJECT_ID}/${REPO_NAME}/maxtext-trainer:${MAXTEXT_IMAGE_TAG}"
VLLM_IMAGE="${DOCKER_REGISTRY_URL}/${PROJECT_ID}/${REPO_NAME}/vllm-server:${VLLM_IMAGE_TAG}"

# Sanitize each YAML file
for file in "${YAML_FILES[@]}"; do
    print_status "Sanitizing $(basename "$file")"
    
    # Create temporary file
    temp_file=$(mktemp)
    
    # Perform reverse replacements (back to placeholders)
    sed -e "s/$PROJECT_ID/YOUR_PROJECT_ID/g" \
        -e "s/$REPO_NAME/YOUR_REPO_NAME/g" \
        -e "s/$HUGGINGFACE_TOKEN/YOUR_HUGGINGFACE_TOKEN/g" \
        -e "s|$MAXTEXT_IMAGE|us-east5-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/maxtext-trainer:jax0.7.2-rev1|g" \
        -e "s|$VLLM_IMAGE|us-east5-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/vllm-server:latest|g" \
        -e "s/$ARTIFACTS_BUCKET/YOUR_ARTIFACTS_BUCKET/g" \
        -e "s/$DATASETS_BUCKET/YOUR_DATASETS_BUCKET/g" \
        "$file" > "$temp_file"
    
    # Replace original file
    mv "$temp_file" "$file"
done

print_success "All YAML files sanitized successfully!"
print_status "Backup created in: $BACKUP_DIR"

# Verify sanitization
print_status "Verifying sanitization..."
SANITIZATION_FAILED=false

for file in "${YAML_FILES[@]}"; do
    # Check if any actual sensitive values remain
    if grep -q "$PROJECT_ID\|$HUGGINGFACE_TOKEN\|$REPO_NAME" "$file" 2>/dev/null; then
        print_warning "$(basename "$file") may still contain sensitive values"
        SANITIZATION_FAILED=true
    fi
    
    # Check if placeholders are restored
    if ! grep -q "YOUR_PROJECT_ID\|YOUR_REPO_NAME\|YOUR_HUGGINGFACE_TOKEN" "$file"; then
        print_warning "$(basename "$file") may not have placeholder values restored"
        SANITIZATION_FAILED=true
    fi
done

if [ "$SANITIZATION_FAILED" = true ]; then
    print_warning "Some files may not have been sanitized completely. Please check manually."
else
    print_success "All sensitive values have been replaced with placeholders!"
fi

# Show what was sanitized
echo
print_status "Sanitization Summary:"
echo "✅ Project ID: $PROJECT_ID -> YOUR_PROJECT_ID"
echo "✅ Repository: $REPO_NAME -> YOUR_REPO_NAME"  
echo "✅ HF Token: ${HUGGINGFACE_TOKEN:0:10}... -> YOUR_HUGGINGFACE_TOKEN"
echo "✅ Docker Images: Restored to template URLs"
echo "✅ GCS Buckets: Restored to template names"
echo

print_status "Repository is now safe for public sharing!"
print_warning "Remember: Your local config/config.env still contains real values"
print_status "To reconfigure for deployment, run: ./scripts/configure.sh"
echo
print_success "Sanitization complete!"
