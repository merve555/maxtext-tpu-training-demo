#!/usr/bin/env python3
"""
Test script for the deployed vLLM API endpoint.
This script demonstrates how to interact with the fine-tuned Gemma-2-27B model.
"""

import requests
import json
import argparse
import time
import sys
from typing import Dict, Any

def get_service_ip(service_name: str = "vllm-gemma-service") -> str:
    """Get the external IP of the Kubernetes service."""
    import subprocess
    try:
        result = subprocess.run([
            "kubectl", "get", "service", service_name, 
            "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}"
        ], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print(f"Error: Could not get IP for service {service_name}")
        print("Make sure the service is deployed and has an external IP")
        return None

def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def list_models(base_url: str) -> Dict[str, Any]:
    """List available models."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ Available models:")
            for model in models.get("data", []):
                print(f"  - {model.get('id', 'unknown')}")
            return models
        else:
            print(f"❌ Failed to list models: {response.status_code}")
            return {}
    except requests.RequestException as e:
        print(f"❌ Failed to list models: {e}")
        return {}

def generate_text(base_url: str, prompt: str, model: str = "gemma-2-27b-finetuned", 
                 max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate text using the completion endpoint."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["\n\n", "###"]
    }
    
    try:
        print(f"🤖 Generating response for: '{prompt[:50]}...'")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"]
            print(f"✅ Generated text in {end_time - start_time:.2f}s:")
            print(f"📝 {generated_text}")
            return generated_text
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return ""
    except requests.RequestException as e:
        print(f"❌ Generation failed: {e}")
        return ""

def chat_completion(base_url: str, messages: list, model: str = "gemma-2-27b-finetuned",
                   max_tokens: int = 200, temperature: float = 0.7) -> str:
    """Generate text using the chat completion endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        print(f"💬 Sending chat message...")
        start_time = time.time()
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            print(f"✅ Generated response in {end_time - start_time:.2f}s:")
            print(f"🤖 {generated_text}")
            return generated_text
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"Response: {response.text}")
            return ""
    except requests.RequestException as e:
        print(f"❌ Chat completion failed: {e}")
        return ""

def run_demo_prompts(base_url: str):
    """Run a series of demo prompts to showcase the model."""
    demo_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to calculate fibonacci numbers:",
        "What are the benefits of using TPUs for machine learning?",
        "Describe the process of fine-tuning a large language model:",
    ]
    
    print("\n🚀 Running demo prompts...")
    print("=" * 60)
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n📋 Demo {i}/{len(demo_prompts)}")
        print("-" * 40)
        generate_text(base_url, prompt, max_tokens=150, temperature=0.7)
        
        if i < len(demo_prompts):
            print("\nWaiting 2 seconds before next prompt...")
            time.sleep(2)

def interactive_mode(base_url: str):
    """Interactive mode for chatting with the model."""
    print("\n💬 Interactive mode - Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n👤 You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if prompt:
                generate_text(base_url, prompt, max_tokens=200, temperature=0.7)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def main():
    parser = argparse.ArgumentParser(description="Test the vLLM API endpoint")
    parser.add_argument("--ip", help="External IP address of the service")
    parser.add_argument("--port", default="8000", help="Port number (default: 8000)")
    parser.add_argument("--mode", choices=["test", "demo", "interactive"], 
                       default="test", help="Test mode")
    parser.add_argument("--prompt", help="Custom prompt to test")
    
    args = parser.parse_args()
    
    # Get service IP if not provided
    if not args.ip:
        print("🔍 Getting service IP...")
        args.ip = get_service_ip()
        if not args.ip:
            sys.exit(1)
    
    base_url = f"http://{args.ip}:{args.port}"
    print(f"🌐 Using API endpoint: {base_url}")
    
    # Test basic connectivity
    print("\n🔍 Testing API connectivity...")
    if not test_health(base_url):
        print("❌ API is not accessible. Check if the service is running.")
        sys.exit(1)
    
    # List models
    print("\n📋 Listing available models...")
    list_models(base_url)
    
    # Run based on mode
    if args.mode == "test":
        if args.prompt:
            print(f"\n🧪 Testing custom prompt...")
            generate_text(base_url, args.prompt)
        else:
            print(f"\n🧪 Testing with sample prompt...")
            generate_text(base_url, "Explain machine learning in one paragraph:")
    
    elif args.mode == "demo":
        run_demo_prompts(base_url)
    
    elif args.mode == "interactive":
        interactive_mode(base_url)
    
    print(f"\n✅ Testing completed!")

if __name__ == "__main__":
    main()
