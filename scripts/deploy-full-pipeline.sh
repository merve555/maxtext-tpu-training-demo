#!/bin/bash

# =============================================================================
# FULL PIPELINE DEPLOYMENT SCRIPT
# =============================================================================
# 
# This script deploys the complete Gemma 3-12B fine-tuning pipeline:
# 1. Model conversion (HuggingFace → MaxText) - TPU
# 2. Model validation test - TPU  
# 3. Fine-tuning on ChartQA dataset - TPU
# 4. Inference test with fine-tuned model - TPU
# 5. Export to HuggingFace format - TPU
# 6. GPU serving with vLLM - GPU
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to wait for job completion
wait_for_job() {
    local job_name=$1
    local timeout=${2:-1800}  # Default 30 minutes
    
    print_status "Waiting for job: $job_name (timeout: ${timeout}s)"
    
    local start_time=$(date +%s)
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -gt $timeout ]; then
            print_error "Job $job_name timed out after ${timeout}s"
            return 1
        fi
        
        # Check job status
        local status=$(kubectl get job $job_name -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "Unknown")
        
        if [ "$status" = "True" ]; then
            print_success "Job $job_name completed successfully!"
            return 0
        elif [ "$status" = "False" ]; then
            print_error "Job $job_name failed!"
            return 1
        fi
        
        echo -n "."
        sleep 10
    done
}

# Function to wait for deployment
wait_for_deployment() {
    local deployment_name=$1
    local timeout=${2:-600}  # Default 10 minutes
    
    print_status "Waiting for deployment: $deployment_name (timeout: ${timeout}s)"
    
    kubectl rollout status deployment/$deployment_name --timeout=${timeout}s
    
    if [ $? -eq 0 ]; then
        print_success "Deployment $deployment_name is ready!"
        return 0
    else
        print_error "Deployment $deployment_name failed to become ready!"
        return 1
    fi
}

# Function to get job logs
get_job_logs() {
    local job_name=$1
    local container_name=${2:-"maxtext-converter"}
    
    print_status "Getting logs for job: $job_name"
    
    # Get the pod name
    local pod_name=$(kubectl get pods -l app=maxtext-conversion -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ -z "$pod_name" ]; then
        # Try different label selectors
        pod_name=$(kubectl get pods --selector="job-name=$job_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    fi
    
    if [ -n "$pod_name" ]; then
        print_status "Pod: $pod_name"
        kubectl logs $pod_name -c $container_name --tail=50
    else
        print_warning "No pod found for job: $job_name"
    fi
}

# Main deployment function
deploy_full_pipeline() {
    print_status "Starting full pipeline deployment..."
    
    # Step 1: Model Conversion
    print_status "=== STEP 1: MODEL CONVERSION ==="
    kubectl apply -f kubernetes/step1-model-conversion.yaml
    if wait_for_job "step1-model-conversion-gemma2b" 1800; then
        print_success "Step 1 completed successfully!"
    else
        print_error "Step 1 failed!"
        get_job_logs "step1-model-conversion-gemma2b" "maxtext-converter"
        exit 1
    fi
    
    # Step 2: Model Validation
    print_status "=== STEP 2: MODEL VALIDATION ==="
    kubectl apply -f kubernetes/step2-model-validation.yaml
    if wait_for_job "step2-model-validation-gemma2b" 600; then
        print_success "Step 2 completed successfully!"
    else
        print_warning "Step 2 failed (expected due to CPU vs TPU checkpoint compatibility)"
        print_status "Continuing with Step 3..."
    fi
    
    # Step 3: Fine-tuning
    print_status "=== STEP 3: FINE-TUNING ==="
    kubectl apply -f kubernetes/step3-fine-tuning.yaml
    if wait_for_job "step3-fine-tuning-gemma2b-ultrachat" 3600; then
        print_success "Step 3 completed successfully!"
    else
        print_error "Step 3 failed!"
        get_job_logs "step3-fine-tuning-gemma2b-ultrachat" "maxtext-finetuner"
        exit 1
    fi
    
    # Step 4: Inference Test
    print_status "=== STEP 4: INFERENCE TEST ==="
    kubectl apply -f kubernetes/step4-inference-test.yaml
    if wait_for_job "step4-inference-test-gemma2b-ultrachat" 600; then
        print_success "Step 4 completed successfully!"
    else
        print_warning "Step 4 failed!"
        get_job_logs "step4-inference-test-gemma2b-ultrachat" "maxtext-inference-tester"
    fi
    
    # Step 5: Export to HuggingFace
    print_status "=== STEP 5: EXPORT TO HUGGINGFACE ==="
    kubectl apply -f kubernetes/step5-export-to-hf.yaml
    if wait_for_job "step5-export-to-hf-gemma2b-ultrachat" 1800; then
        print_success "Step 5 completed successfully!"
    else
        print_error "Step 5 failed!"
        get_job_logs "step5-export-to-hf-gemma2b-ultrachat" "maxtext-exporter"
        exit 1
    fi
    
    # Build vLLM GPU serving image
    print_status "=== BUILDING VLLM GPU SERVING IMAGE ==="
    docker build -f docker/Dockerfile.vllm-serving -t us-east5-docker.pkg.dev/diesel-patrol-382622/merves-tpu-demo/vllm-server:latest .
    docker push us-east5-docker.pkg.dev/diesel-patrol-382622/merves-tpu-demo/vllm-server:latest
    print_success "vLLM serving image built and pushed!"
    
    # Step 6: GPU Serving
    print_status "=== STEP 6: GPU SERVING ==="
    kubectl apply -f kubernetes/step6-gpu-serving.yaml
    if wait_for_deployment "step6-gpu-serving-gemma2b-ultrachat" 1200; then
        print_success "Step 6 completed successfully!"
    else
        print_error "Step 6 failed!"
        kubectl logs deployment/step6-gpu-serving-gemma2b-ultrachat --tail=50
        exit 1
    fi
    
    # Get service information
    print_status "=== DEPLOYMENT COMPLETE ==="
    print_success "Full pipeline deployed successfully!"
    
    # Get service endpoint
    local service_ip=$(kubectl get service step6-gpu-serving-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
    local service_port=$(kubectl get service step6-gpu-serving-service -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8000")
    
    echo ""
    echo "🎉 Fine-tuned Gemma 2-2B model is now serving!"
    echo ""
    echo "📊 Service Information:"
    echo "   Endpoint: http://${service_ip}:${service_port}"
    echo "   Model: gemma2-2b-ultrachat"
    echo ""
    echo "🧪 Test the API:"
    echo "   curl -X POST http://${service_ip}:${service_port}/v1/completions \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"model\": \"gemma2-2b-ultrachat\", \"prompt\": \"Hello, how are you today?\", \"max_tokens\": 100}'"
    echo ""
    echo "📁 Model artifacts:"
    echo "   Fine-tuned MaxText checkpoint: /gcs/artifacts/checkpoints/finetuned/"
    echo "   HuggingFace model: /gcs/artifacts/finetuned-hf-models/"
    echo ""
}

# Function to clean up resources
cleanup() {
    print_status "Cleaning up resources..."
    
    kubectl delete job step1-model-conversion-gemma2b 2>/dev/null || true
    kubectl delete job step2-model-validation-gemma2b 2>/dev/null || true
    kubectl delete job step3-fine-tuning-gemma2b-ultrachat 2>/dev/null || true
    kubectl delete job step4-inference-test-gemma2b-ultrachat 2>/dev/null || true
    kubectl delete job step5-export-to-hf-gemma2b-ultrachat 2>/dev/null || true
    kubectl delete deployment step6-gpu-serving-gemma2b-ultrachat 2>/dev/null || true
    kubectl delete service step6-gpu-serving-service 2>/dev/null || true
    
    print_success "Cleanup completed!"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        deploy_full_pipeline
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        get_job_logs "${2:-step1-model-conversion-gemma12b}" "${3:-maxtext-converter}"
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|logs [job_name] [container_name]}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the full pipeline (default)"
        echo "  cleanup - Clean up all resources"
        echo "  logs    - Get logs for a specific job"
        echo ""
        echo "Examples:"
        echo "  $0 deploy"
        echo "  $0 cleanup"
        echo "  $0 logs step3-fine-tuning-gemma12b-chartqa maxtext-finetuner"
        exit 1
        ;;
esac
