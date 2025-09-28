#!/bin/bash

# TPU Trillium Training Demo - Monitoring Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

check_cluster_status() {
    print_header "Cluster Status"
    
    kubectl get nodes -o wide
    echo
    kubectl get pods --all-namespaces
}

check_training_status() {
    print_header "Training Job Status"
    
    # Check if training job exists
    if kubectl get job gemma-2-27b-finetune &> /dev/null; then
        kubectl describe job gemma-2-27b-finetune
        echo
        
        # Get pod status
        TRAINING_POD=$(kubectl get pods -l job-name=gemma-2-27b-finetune -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
        
        if [ ! -z "$TRAINING_POD" ]; then
            print_status "Training pod: $TRAINING_POD"
            kubectl get pod $TRAINING_POD
            echo
            
            # Show recent logs
            print_status "Recent training logs:"
            kubectl logs $TRAINING_POD --tail=50 || print_warning "Could not fetch logs"
        else
            print_warning "No training pod found"
        fi
    else
        print_warning "Training job not found"
    fi
}

check_serving_status() {
    print_header "Serving Status"
    
    # Check if serving deployment exists
    if kubectl get deployment vllm-gemma-serving &> /dev/null; then
        kubectl describe deployment vllm-gemma-serving
        echo
        
        # Get service status
        kubectl get service vllm-gemma-service
        echo
        
        # Get pod status
        SERVING_PODS=$(kubectl get pods -l app=vllm-serving -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || echo "")
        
        if [ ! -z "$SERVING_PODS" ]; then
            for pod in $SERVING_PODS; do
                print_status "Serving pod: $pod"
                kubectl get pod $pod
                echo
                
                # Show recent logs
                print_status "Recent serving logs for $pod:"
                kubectl logs $pod --tail=20 || print_warning "Could not fetch logs for $pod"
                echo
            done
        else
            print_warning "No serving pods found"
        fi
    else
        print_warning "Serving deployment not found"
    fi
}

check_resources() {
    print_header "Resource Usage"
    
    # Check node resource usage
    kubectl top nodes || print_warning "Metrics server not available"
    echo
    
    # Check pod resource usage
    kubectl top pods || print_warning "Metrics server not available"
}

check_tpu_status() {
    print_header "TPU Status"
    
    # Check TPU nodes
    kubectl get nodes -l cloud.google.com/gke-tpu-accelerator --show-labels
    echo
    
    # Check TPU pods
    kubectl get pods -o wide | grep tpu || print_status "No TPU pods currently running"
}

test_serving_endpoint() {
    print_header "Testing Serving Endpoint"
    
    # Get service endpoint
    EXTERNAL_IP=$(kubectl get service vllm-gemma-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    
    if [ ! -z "$EXTERNAL_IP" ]; then
        print_status "External IP: $EXTERNAL_IP"
        
        # Test health endpoint
        print_status "Testing health endpoint..."
        if curl -s -f http://$EXTERNAL_IP:8000/health > /dev/null; then
            print_status "✅ Health endpoint is responding"
        else
            print_warning "❌ Health endpoint is not responding"
        fi
        
        # Test models endpoint
        print_status "Testing models endpoint..."
        if curl -s -f http://$EXTERNAL_IP:8000/v1/models > /dev/null; then
            print_status "✅ Models endpoint is responding"
            curl -s http://$EXTERNAL_IP:8000/v1/models | python3 -m json.tool
        else
            print_warning "❌ Models endpoint is not responding"
        fi
    else
        print_warning "External IP not available yet"
        kubectl get service vllm-gemma-service
    fi
}

watch_training_logs() {
    print_header "Watching Training Logs"
    
    TRAINING_POD=$(kubectl get pods -l job-name=gemma-2-27b-finetune -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ ! -z "$TRAINING_POD" ]; then
        print_status "Following logs for pod: $TRAINING_POD"
        kubectl logs -f $TRAINING_POD
    else
        print_error "Training pod not found"
        exit 1
    fi
}

watch_serving_logs() {
    print_header "Watching Serving Logs"
    
    SERVING_POD=$(kubectl get pods -l app=vllm-serving -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    if [ ! -z "$SERVING_POD" ]; then
        print_status "Following logs for pod: $SERVING_POD"
        kubectl logs -f $SERVING_POD
    else
        print_error "Serving pod not found"
        exit 1
    fi
}

main() {
    case "${1:-status}" in
        "status"|"all")
            check_cluster_status
            echo
            check_training_status
            echo
            check_serving_status
            echo
            check_resources
            echo
            check_tpu_status
            ;;
        "training")
            check_training_status
            ;;
        "serving")
            check_serving_status
            ;;
        "resources")
            check_resources
            ;;
        "tpu")
            check_tpu_status
            ;;
        "test")
            test_serving_endpoint
            ;;
        "logs-training")
            watch_training_logs
            ;;
        "logs-serving")
            watch_serving_logs
            ;;
        *)
            echo "Usage: $0 [status|training|serving|resources|tpu|test|logs-training|logs-serving]"
            echo "  status:        Show overall status (default)"
            echo "  training:      Show training job status"
            echo "  serving:       Show serving deployment status"
            echo "  resources:     Show resource usage"
            echo "  tpu:           Show TPU status"
            echo "  test:          Test serving endpoint"
            echo "  logs-training: Follow training logs"
            echo "  logs-serving:  Follow serving logs"
            exit 1
            ;;
    esac
}

main "$@"
