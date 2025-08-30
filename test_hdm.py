import torch
import sys
import os

# Add HDM to path
sys.path.insert(0, "D:\\code\\HDM\\src")

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

try:
    # Import HDM
    import xut
    xut.env.USE_XFORMERS_LAYERS = True
    from hdm.pipeline import HDMXUTPipeline
    
    print("Loading HDM pipeline...")
    hdm_pipe = HDMXUTPipeline.from_pretrained(
        "KBlueLeaf/HDM-xut-340M-anime", 
        trust_remote_code=True
    ).to("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("HDM pipeline loaded successfully!")
    print(f"Pipeline device: {hdm_pipe.device}")
    
    # Try to generate an image
    print("\nGenerating test image...")
    with torch.inference_mode():
        result = hdm_pipe(
            prompts=["a simple test art"],
            negative_prompts="low quality",
            width=512,
            height=512,
            cfg_scale=3.0,
            num_inference_steps=4,
            camera_param={
                "zoom": 1.0,
                "x_shift": 0.0,
                "y_shift": 0.0,
            },
        )
    
    print("Image generation successful!")
    if result and len(result) > 0:
        print(f"Generated {len(result)} images")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()