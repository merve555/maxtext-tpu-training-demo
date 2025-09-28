#!/bin/bash

# TPU Trillium Training Demo - Cleanup Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
PROJECT_ID=${PROJECT_ID:-""}
REGION=${REGION:-"us-central1"}
CLUSTER_NAME=${CLUSTER_NAME:-"tpu-training-demo"}

cleanup_kubernetes() {
    print_status "Cleaning up Kubernetes resources..."
    
    # Delete training job
    kubectl delete job gemma-2-27b-finetune --ignore-not-found=true
    
    # Delete serving deployment
    kubectl delete deployment vllm-gemma-serving --ignore-not-found=true
    kubectl delete service vllm-gemma-service --ignore-not-found=true
    kubectl delete ingress vllm-ingress --ignore-not-found=true
    
    # Delete service accounts
    kubectl delete serviceaccount maxtext-sa --ignore-not-found=true
    kubectl delete serviceaccount vllm-sa --ignore-not-found=true
    
    # Delete RBAC resources
    kubectl delete role maxtext-role --ignore-not-found=true
    kubectl delete rolebinding maxtext-rolebinding --ignore-not-found=true
    
    print_status "Kubernetes cleanup completed!"
}

cleanup_docker_images() {
    print_status "Cleaning up Docker images..."
    
    if [ ! -z "$PROJECT_ID" ]; then
        # Delete Artifact Registry repository
        gcloud artifacts repositories delete tpu-demo \
            --location=$REGION \
            --quiet || print_warning "Could not delete Artifact Registry repository"
    else
        print_warning "PROJECT_ID not set, skipping Docker cleanup"
    fi
}

cleanup_infrastructure() {
    print_status "Cleaning up infrastructure with Terraform..."
    
    cd terraform
    
    if [ -f terraform.tfvars ]; then
        terraform destroy -auto-approve
    else
        print_warning "terraform.tfvars not found, skipping Terraform cleanup"
    fi
    
    cd ..
    
    print_status "Infrastructure cleanup completed!"
}

cleanup_local() {
    print_status "Cleaning up local files..."
    
    # Remove terraform state files (optional)
    read -p "Remove Terraform state files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf terraform/.terraform
        rm -f terraform/terraform.tfstate*
        rm -f terraform/terraform.tfvars
        print_status "Terraform state files removed"
    fi
    
    # Remove Docker build cache (optional)
    read -p "Remove Docker build cache? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        podman system prune -f
        print_status "Docker build cache removed"
    fi
}

main() {
    print_warning "This will delete all resources created by the TPU Training Demo!"
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleanup cancelled"
        exit 0
    fi
    
    case "${1:-all}" in
        "k8s")
            cleanup_kubernetes
            ;;
        "images")
            cleanup_docker_images
            ;;
        "infra")
            cleanup_infrastructure
            ;;
        "local")
            cleanup_local
            ;;
        "all")
            cleanup_kubernetes
            cleanup_docker_images
            cleanup_infrastructure
            cleanup_local
            ;;
        *)
            echo "Usage: $0 [k8s|images|infra|local|all]"
            echo "  k8s:    Clean up Kubernetes resources"
            echo "  images: Clean up Docker images"
            echo "  infra:  Clean up infrastructure (Terraform)"
            echo "  local:  Clean up local files"
            echo "  all:    Clean up everything (default)"
            exit 1
            ;;
    esac
    
    print_status "Cleanup completed!"
}

main "$@"
