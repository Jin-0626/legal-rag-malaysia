#!/usr/bin/env python3
"""
Pull Ollama models via HTTP API
"""
import requests
import json
import sys
from pathlib import Path

def pull_model(model_name: str, ollama_url: str = "http://localhost:11434") -> bool:
    """Pull a model from Ollama"""
    print(f"Pulling model: {model_name}")
    
    try:
        url = f"{ollama_url}/api/pull"
        payload = {"name": model_name}
        
        response = requests.post(url, json=payload, stream=True, timeout=600)
        
        if response.status_code == 200:
            # Stream the response to show progress
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'status' in data:
                            print(f"  {data['status']}", end='\r')
                    except:
                        pass
            print(f"\nOK: {model_name} downloaded successfully")
            return True
        else:
            print(f"FAIL: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"FAIL: {e}")
        return False

def main():
    """Main function"""
    models = ["mistral", "nomic-embed-text"]
    
    print("=== Downloading Ollama Models ===\n")
    
    all_success = True
    for model in models:
        success = pull_model(model)
        all_success = all_success and success
        print()
    
    if all_success:
        print("SUCCESS: All models downloaded!")
        print("\nVerifying models...")
        
        # Verify models
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models_list = response.json().get('models', [])
            print(f"\nAvailable models ({len(models_list)}):")
            for m in models_list:
                print(f"  - {m['name']}")
        except Exception as e:
            print(f"Could not verify models: {e}")
    else:
        print("FAIL: Some models failed to download")
        sys.exit(1)

if __name__ == "__main__":
    main()
