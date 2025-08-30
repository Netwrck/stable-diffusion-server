#!/usr/bin/env python
"""Test the API endpoints with multiple images."""

import requests
import time
import json
from urllib.parse import urlencode

def test_make_image(prompt, save_name):
    """Test the /make_image endpoint."""
    print(f"\n[IMAGE] Generating: {prompt}")
    
    # URL encode the parameters
    params = {
        "prompt": prompt,
        "width": 512,
        "height": 512
    }
    
    url = f"http://localhost:8000/make_image?{urlencode(params)}"
    
    start_time = time.time()
    try:
        response = requests.get(url, timeout=300)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            # Save the image
            with open(save_name, 'wb') as f:
                f.write(response.content)
            print(f"[OK] Saved to {save_name} ({elapsed:.1f}s)")
            return True
        else:
            print(f"[ERROR] Status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"[FAILED] {e}")
        return False

def test_create_and_upload(prompt, save_path):
    """Test the /create_and_upload_image endpoint."""
    print(f"\n[UPLOAD] Creating and uploading: {prompt}")
    
    params = {
        "prompt": prompt,
        "save_path": save_path,
        "width": 512,
        "height": 512
    }
    
    url = f"http://localhost:8000/create_and_upload_image?{urlencode(params)}"
    
    start_time = time.time()
    try:
        response = requests.get(url, timeout=300)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Response: {result} ({elapsed:.1f}s)")
            return True
        else:
            print(f"[WARNING] Status {response.status_code}: {response.text}")
            # Try to save locally anyway if it's a cloud storage error
            return False
    except Exception as e:
        print(f"[FAILED] {e}")
        return False

def main():
    """Test multiple images with different prompts."""
    
    # Test prompts
    test_cases = [
        ("a cute robot playing guitar in a cyberpunk city", "robot_guitar.png"),
        ("magical forest with glowing mushrooms at night", "magic_forest.png"),
        ("steampunk airship flying above victorian london", "steampunk_airship.png"),
        ("astronaut riding a horse on mars", "astronaut_mars.png"),
        ("japanese temple in cherry blossom season, anime style", "temple_sakura.png")
    ]
    
    print("=" * 60)
    print("Testing Stable Diffusion Server API")
    print("=" * 60)
    
    # Test /make_image endpoint
    print("\n[TEST] Testing /make_image endpoint...")
    successful = 0
    for prompt, filename in test_cases[:3]:  # Test first 3 with make_image
        if test_make_image(prompt, filename):
            successful += 1
    
    print(f"\n[RESULT] /make_image: {successful}/3 successful")
    
    # Test /create_and_upload_image endpoint  
    print("\n[TEST] Testing /create_and_upload_image endpoint...")
    successful_upload = 0
    for prompt, filename in test_cases[3:]:  # Test last 2 with create_and_upload
        save_path = f"test_uploads/{filename.replace('.png', '.webp')}"
        if test_create_and_upload(prompt, save_path):
            successful_upload += 1
    
    print(f"\n[RESULT] /create_and_upload_image: {successful_upload}/2 attempted")
    
    print("\n" + "=" * 60)
    print("Testing complete! Check the generated images.")
    print("=" * 60)

if __name__ == "__main__":
    main()