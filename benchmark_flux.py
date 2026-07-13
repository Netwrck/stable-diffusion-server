#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for Flux dev and img-to-img operations
Tests real-world scenarios with timing, memory usage, and quality metrics
"""

import time
import psutil
import torch
import gc
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import traceback
from PIL import Image
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

import main
from main import initialize_unified_pipelines, clear_gpu_memory
from stable_diffusion_server.image_processing import process_image_for_stable_diffusion


class FluxBenchmark:
    """Comprehensive benchmark suite for Flux pipelines"""
    
    def __init__(self):
        self.results = []
        self.test_prompts = [
            "a beautiful landscape with mountains and lake, photorealistic",
            "anime style character, detailed art, vibrant colors",
            "abstract geometric patterns, modern art style",
            "portrait of a person, studio lighting, professional photo",
            "fantasy dragon in a mystical forest, epic scene"
        ]
        
        self.test_images_dir = Path("test_images")
        self.test_images_dir.mkdir(exist_ok=True)
        
        self.output_dir = Path("benchmark_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create or use existing test image for img2img
        self.test_image_path = self.create_test_image()
        
    def create_test_image(self) -> str:
        """Create a simple test image for img2img benchmarking"""
        test_img_path = self.test_images_dir / "test_base_image.jpg"
        
        if not test_img_path.exists():
            # Create a simple gradient image for testing
            img_array = np.zeros((512, 512, 3), dtype=np.uint8)
            for i in range(512):
                for j in range(512):
                    img_array[i, j] = [i % 256, j % 256, (i + j) % 256]
            
            img = Image.fromarray(img_array)
            img.save(test_img_path, "JPEG", quality=95)
            print(f"Created test image: {test_img_path}")
            
        return str(test_img_path)
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        result = {
            "ram_mb": memory_info.rss / 1024 / 1024,
            "ram_gb": memory_info.rss / 1024 / 1024 / 1024
        }
        
        if torch.cuda.is_available():
            result.update({
                "vram_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "vram_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "vram_allocated_gb": torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                "vram_cached_gb": torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            })
        
        return result
    
    def benchmark_text_to_image(self, prompt: str, test_id: str) -> Dict:
        """Benchmark text-to-image generation"""
        print(f"🎨 Testing text-to-image: {test_id}")
        
        # Clear memory before test
        clear_gpu_memory()
        initial_memory = self.get_memory_usage()
        
        try:
            start_time = time.time()
            
            # Generate image using BASE_PIPE
            image = main.BASE_PIPE(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator().manual_seed(42)
            ).images[0]
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Save output image
            output_path = self.output_dir / f"{test_id}_text2img.jpg"
            image.save(output_path, "JPEG", quality=95)
            
            final_memory = self.get_memory_usage()
            
            return {
                "test_type": "text_to_image",
                "test_id": test_id,
                "prompt": prompt,
                "success": True,
                "generation_time": generation_time,
                "output_path": str(output_path),
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "memory_delta": {
                    "ram_mb": final_memory["ram_mb"] - initial_memory["ram_mb"],
                    "vram_allocated_mb": final_memory.get("vram_allocated_mb", 0) - initial_memory.get("vram_allocated_mb", 0)
                },
                "image_size": image.size,
                "steps": 4
            }
            
        except Exception as e:
            return {
                "test_type": "text_to_image",
                "test_id": test_id,
                "prompt": prompt,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "initial_memory": initial_memory,
                "final_memory": self.get_memory_usage()
            }
    
    def benchmark_img_to_img(self, prompt: str, test_id: str) -> Dict:
        """Benchmark image-to-image generation"""
        print(f"🖼️  Testing img-to-img: {test_id}")
        
        # Clear memory before test
        clear_gpu_memory()
        initial_memory = self.get_memory_usage()
        
        try:
            # Load and process input image
            input_image = Image.open(self.test_image_path).convert("RGB")
            input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
            
            start_time = time.time()
            
            # Generate image using IMG2IMG_PIPE
            image = main.IMG2IMG_PIPE(
                prompt=prompt,
                image=input_image,
                strength=0.7,
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=torch.Generator().manual_seed(42)
            ).images[0]
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Save output image
            output_path = self.output_dir / f"{test_id}_img2img.jpg"
            image.save(output_path, "JPEG", quality=95)
            
            final_memory = self.get_memory_usage()
            
            return {
                "test_type": "img_to_img",
                "test_id": test_id,
                "prompt": prompt,
                "input_image": self.test_image_path,
                "success": True,
                "generation_time": generation_time,
                "output_path": str(output_path),
                "initial_memory": initial_memory,
                "final_memory": final_memory,
                "memory_delta": {
                    "ram_mb": final_memory["ram_mb"] - initial_memory["ram_mb"],
                    "vram_allocated_mb": final_memory.get("vram_allocated_mb", 0) - initial_memory.get("vram_allocated_mb", 0)
                },
                "image_size": image.size,
                "strength": 0.7,
                "steps": 4
            }
            
        except Exception as e:
            return {
                "test_type": "img_to_img", 
                "test_id": test_id,
                "prompt": prompt,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "initial_memory": initial_memory,
                "final_memory": self.get_memory_usage()
            }
    
    def benchmark_step_variations(self) -> List[Dict]:
        """Test different inference step counts for performance vs quality"""
        print("📊 Testing step variations...")
        
        step_counts = [1, 2, 4, 8, 16]
        prompt = "beautiful sunset over ocean, photorealistic"
        results = []
        
        for steps in step_counts:
            clear_gpu_memory()
            initial_memory = self.get_memory_usage()
            
            try:
                start_time = time.time()
                
                image = main.BASE_PIPE(
                    prompt=prompt,
                    height=512,
                    width=512,
                    num_inference_steps=steps,
                    guidance_scale=0.0,
                    generator=torch.Generator().manual_seed(42)
                ).images[0]
                
                end_time = time.time()
                generation_time = end_time - start_time
                
                output_path = self.output_dir / f"steps_{steps}_test.jpg"
                image.save(output_path, "JPEG", quality=95)
                
                final_memory = self.get_memory_usage()
                
                results.append({
                    "test_type": "step_variation",
                    "steps": steps,
                    "prompt": prompt,
                    "success": True,
                    "generation_time": generation_time,
                    "time_per_step": generation_time / steps,
                    "output_path": str(output_path),
                    "memory_delta": {
                        "ram_mb": final_memory["ram_mb"] - initial_memory["ram_mb"],
                        "vram_allocated_mb": final_memory.get("vram_allocated_mb", 0) - initial_memory.get("vram_allocated_mb", 0)
                    }
                })
                
                print(f"  ✓ {steps} steps: {generation_time:.2f}s ({generation_time/steps:.3f}s/step)")
                
            except Exception as e:
                results.append({
                    "test_type": "step_variation", 
                    "steps": steps,
                    "success": False,
                    "error": str(e)
                })
                print(f"  ❌ {steps} steps: Failed - {e}")
        
        return results
    
    def run_all_benchmarks(self) -> Dict:
        """Run complete benchmark suite"""
        print("🚀 Starting Flux Benchmark Suite")
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize pipelines
        print("🔧 Initializing pipelines...")
        try:
            initialize_unified_pipelines()
            print("✓ Pipelines initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize pipelines: {e}")
            return {"error": "Pipeline initialization failed", "details": str(e)}
        
        all_results = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "system_memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024
            },
            "text_to_image_tests": [],
            "img_to_img_tests": [],
            "step_variation_tests": [],
            "summary": {}
        }
        
        # Run text-to-image benchmarks
        print("\n🎨 Running Text-to-Image Benchmarks...")
        for i, prompt in enumerate(self.test_prompts):
            result = self.benchmark_text_to_image(prompt, f"t2i_{i+1}")
            all_results["text_to_image_tests"].append(result)
            self.results.append(result)
        
        # Run img-to-img benchmarks  
        print("\n🖼️  Running Image-to-Image Benchmarks...")
        for i, prompt in enumerate(self.test_prompts):
            result = self.benchmark_img_to_img(prompt, f"i2i_{i+1}")
            all_results["img_to_img_tests"].append(result)
            self.results.append(result)
        
        # Run step variation tests
        print("\n📊 Running Step Variation Tests...")
        step_results = self.benchmark_step_variations()
        all_results["step_variation_tests"] = step_results
        self.results.extend(step_results)
        
        # Generate summary
        all_results["summary"] = self.generate_summary()
        
        # Save results
        results_file = Path("benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 Results saved to: {results_file}")
        return all_results
    
    def generate_summary(self) -> Dict:
        """Generate performance summary statistics"""
        successful_t2i = [r for r in self.results if r.get("test_type") == "text_to_image" and r.get("success")]
        successful_i2i = [r for r in self.results if r.get("test_type") == "img_to_img" and r.get("success")]
        successful_steps = [r for r in self.results if r.get("test_type") == "step_variation" and r.get("success")]
        
        summary = {
            "total_tests": len(self.results),
            "successful_tests": len([r for r in self.results if r.get("success")]),
            "failed_tests": len([r for r in self.results if not r.get("success")])
        }
        
        if successful_t2i:
            t2i_times = [r["generation_time"] for r in successful_t2i]
            summary["text_to_image"] = {
                "count": len(successful_t2i),
                "avg_time": sum(t2i_times) / len(t2i_times),
                "min_time": min(t2i_times),
                "max_time": max(t2i_times)
            }
        
        if successful_i2i:
            i2i_times = [r["generation_time"] for r in successful_i2i]
            summary["img_to_img"] = {
                "count": len(successful_i2i),
                "avg_time": sum(i2i_times) / len(i2i_times),
                "min_time": min(i2i_times),
                "max_time": max(i2i_times)
            }
        
        if successful_steps:
            summary["step_analysis"] = {}
            for result in successful_steps:
                steps = result["steps"]
                summary["step_analysis"][f"{steps}_steps"] = {
                    "total_time": result["generation_time"],
                    "time_per_step": result["time_per_step"]
                }
        
        return summary


def main():
    """Main benchmark execution"""
    benchmark = FluxBenchmark()
    results = benchmark.run_all_benchmarks()
    
    if "error" in results:
        print(f"❌ Benchmark failed: {results['error']}")
        sys.exit(1)
    
    print("\n✅ Benchmark completed successfully!")
    print(f"📊 Summary: {results['summary']['successful_tests']}/{results['summary']['total_tests']} tests passed")
    
    return results


if __name__ == "__main__":
    main()