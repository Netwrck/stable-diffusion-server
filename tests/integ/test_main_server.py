import os
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from main import app

client = TestClient(app)

@pytest.mark.skip(reason="requires heavy model and long runtime")
def test_style_transfer_bytes_and_upload_image():
    # Path to the test image
    image_path = "tests/data/gunbladedraw.png"
    
    # Ensure the test image exists
    assert os.path.exists(image_path), f"Test image not found at {image_path}"
    
    # Prepare the test data
    with open(image_path, "rb") as image_file:
        files = {"image_file": ("gunbladedraw.png", image_file, "image/png")}
        
        # Construct the URL with query parameters
        url = f"""/style_transfer_bytes_and_upload_image?prompt=A%20futuristic%20gunblade%20in%20a%20cyberpunk%20style&save_path=test_output/styled_gunblade.webp&strength=0.7&canny=true"""
        
        # Make the POST request with URL parameters
        response = client.post(url, files=files)
    # Check the response
    print(response)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    # Parse the response JSON
    response_data = response.json()
    
    # Check if the 'path' key exists in the response
    assert "path" in response_data, "Response does not contain 'path' key"
    
    # Check if the path is a non-empty string
    assert isinstance(response_data["path"], str) and response_data["path"], "Path should be a non-empty string"
    
    # Optionally, you could check if the file exists in the specified path
    # This depends on how your cloud storage is set up and might not be feasible in all testing environments
    # assert os.path.exists(response_data["path"]), f"Generated image not found at {response_data['path']}"

    print(f"Style transfer successful. Image saved at: {response_data['path']}")


@pytest.mark.skip(reason="requires heavy model and long runtime")
def test_style_transfer_bytes_and_upload_image_without_canny():
    # Path to the test image
    image_path = "tests/data/gunbladedraw.png"
    
    # Ensure the test image exists
    assert os.path.exists(image_path), f"Test image not found at {image_path}"
    
    # Prepare the test data
    with open(image_path, "rb") as image_file:
        files = {"image_file": ("gunbladedraw.png", image_file, "image/png")}
        
        # Construct the URL with query parameters
        url = f"""/style_transfer_bytes_and_upload_image?prompt=A%20futuristic%20gunblade%20in%20a%20cyberpunk%20style&save_path=test_output/styled_gunblade_no_canny.webp&strength=0.7&canny=false"""
        
        # Make the POST request with URL parameters
        response = client.post(url, files=files)
    
    # Check the response
    print(response)
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"
    
    # Parse the response JSON
    response_data = response.json()
    
    # Check if the 'path' key exists in the response
    assert "path" in response_data, "Response does not contain 'path' key"
    
    # Check if the path is a non-empty string
    assert isinstance(response_data["path"], str) and response_data["path"], "Path should be a non-empty string"
    
    # Optionally, you could check if the file exists in the specified path
    # This depends on how your cloud storage is set up and might not be feasible in all testing environments
    # assert os.path.exists(response_data["path"]), f"Generated image not found at {response_data['path']}"

    print(f"Style transfer without canny successful. Image saved at: {response_data['path']}")
