#!/usr/bin/env python3
"""Runtime optimization utilities for Flux and SDXL/Proteus pipelines."""

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_ENV = {
    "CUDA_MODULE_LOADING": "LAZY",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128,expandable_segments:True",
    "TORCH_COMPILE_DEBUG": "0",
    "TORCHINDUCTOR_CACHE_DIR": "./torch_cache",
    "TOKENIZERS_PARALLELISM": "false",
}

for _key, _value in _DEFAULT_ENV.items():
    os.environ.setdefault(_key, _value)

import torch
from loguru import logger


TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    logger.warning(f"Invalid boolean for {name}={value!r}; using {default}")
    return default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {name}={value!r}; using {default}")
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {name}={value!r}; using {default}")
        return default


def env_float_list(name: str) -> list[float] | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        return [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        logger.warning(f"Invalid comma-separated float list for {name}={value!r}")
        return None


def _cuda_device() -> str | None:
    return "cuda" if torch.cuda.is_available() else None


def _gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    device: str | None
    cpu_offload: bool
    attention_slicing: bool
    vae_slicing: bool
    vae_tiling: bool
    xformers: bool
    torch_compile: bool
    torch_compile_fullgraph: bool
    cache_mode: str
    first_block_cache_threshold: float

    @classmethod
    def from_env(cls) -> "RuntimeProfile":
        name = os.getenv("SDIF_OPTIMIZATION_PROFILE", "balanced").strip().lower()
        if name not in {"fast", "balanced", "low_memory"}:
            logger.warning(f"Unknown SDIF_OPTIMIZATION_PROFILE={name!r}; using balanced")
            name = "balanced"

        device = _cuda_device()
        default_offload = name in {"balanced", "low_memory"}
        default_attention_slicing = name == "low_memory"
        default_vae_slicing = name in {"balanced", "low_memory"}
        default_vae_tiling = name == "low_memory"
        default_xformers = name in {"fast", "balanced"}

        return cls(
            name=name,
            device=device,
            cpu_offload=env_bool("SDIF_CPU_OFFLOAD", default_offload),
            attention_slicing=env_bool("SDIF_ATTENTION_SLICING", default_attention_slicing),
            vae_slicing=env_bool("SDIF_VAE_SLICING", default_vae_slicing),
            vae_tiling=env_bool("SDIF_VAE_TILING", default_vae_tiling),
            xformers=env_bool("SDIF_XFORMERS", default_xformers),
            torch_compile=env_bool("SDIF_TORCH_COMPILE", False),
            torch_compile_fullgraph=env_bool("SDIF_TORCH_COMPILE_FULLGRAPH", False),
            cache_mode=os.getenv("SDIF_CACHE_MODE", "first_block").strip().lower(),
            first_block_cache_threshold=env_float("SDIF_FIRST_BLOCK_CACHE_THRESHOLD", 0.05),
        )


def resolve_model_repo(env_name: str, local_candidates: list[str], default_repo: str) -> str:
    configured = os.getenv(env_name)
    if configured:
        return configured
    for candidate in local_candidates:
        if Path(candidate).exists():
            return candidate
    return default_repo


def resolve_flux_model_repo() -> str:
    return resolve_model_repo(
        "FLUX_MODEL_REPO",
        ["models/FLUX.1-schnell"],
        "black-forest-labs/FLUX.1-schnell",
    )


def resolve_proteus_model_repo() -> str:
    configured = os.getenv("PROTEUS_MODEL_REPO") or os.getenv("SDXL_MODEL_REPO")
    if configured:
        return configured

    local_candidates = ["models/ProteusV0.4", "models/proteus-v4"]
    if env_bool("SDIF_ALLOW_PROTEUS_V02_FALLBACK", False):
        local_candidates.append("models/ProteusV0.2")
    for candidate in local_candidates:
        if Path(candidate).exists():
            return candidate

    return "dataautogpt3/ProteusV0.4"


def sdxl_ays_timesteps(step_count: int) -> list[int] | None:
    """Return the built-in SDXL AYS schedule, downsampled only for explicit low-step evals."""
    try:
        from diffusers.schedulers import AysSchedules
    except Exception as exc:
        logger.warning(f"AYS schedules unavailable: {exc}")
        return None

    schedule = list(AysSchedules["StableDiffusionXLTimesteps"])
    if step_count == len(schedule):
        return schedule
    if step_count <= 0 or step_count > len(schedule):
        return None

    if step_count == 1:
        return [schedule[0]]

    last = len(schedule) - 1
    indices = sorted({round(i * last / (step_count - 1)) for i in range(step_count)})
    if len(indices) != step_count:
        indices = list(range(step_count))
    return [schedule[i] for i in indices]


def build_inference_kwargs(
    family: str,
    task: str,
    steps: int | None = None,
    guidance_scale: float | None = None,
    extra_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build call kwargs for fast server inference without duplicating scheduler args."""
    extra_args = dict(extra_args or {})
    family = family.lower()
    task = task.lower()

    if family == "flux":
        default_steps = env_int(f"FLUX_{task.upper()}_STEPS", env_int("FLUX_NUM_STEPS", 4))
        default_guidance = 1.0 if task in {"control", "inpaint"} else 0.0
        kwargs: dict[str, Any] = {
            "num_inference_steps": int(steps if steps is not None else default_steps),
            "guidance_scale": float(
                guidance_scale
                if guidance_scale is not None
                else env_float(f"FLUX_{task.upper()}_GUIDANCE_SCALE", env_float("FLUX_GUIDANCE_SCALE", default_guidance))
            ),
            "max_sequence_length": env_int("FLUX_MAX_SEQUENCE_LENGTH", 256),
        }
        sigmas = env_float_list("FLUX_SIGMAS")
        if sigmas:
            kwargs.pop("num_inference_steps", None)
            kwargs["sigmas"] = sigmas
        kwargs.update(extra_args)
        return kwargs

    if family in {"sdxl", "proteus"}:
        default_steps = env_int(f"SDXL_{task.upper()}_STEPS", env_int("SDXL_NUM_STEPS", 20))
        step_count = int(steps if steps is not None else default_steps)
        kwargs = {
            "guidance_scale": float(
                guidance_scale
                if guidance_scale is not None
                else env_float(f"SDXL_{task.upper()}_GUIDANCE_SCALE", env_float("SDXL_GUIDANCE_SCALE", 5.0))
            )
        }
        if env_bool("SDXL_USE_AYS", False):
            timesteps = sdxl_ays_timesteps(step_count)
            if timesteps:
                kwargs["timesteps"] = timesteps
            else:
                kwargs["num_inference_steps"] = step_count
        else:
            kwargs["num_inference_steps"] = step_count

        sigmas = env_float_list("SDXL_SIGMAS")
        if sigmas:
            kwargs.pop("num_inference_steps", None)
            kwargs.pop("timesteps", None)
            kwargs["sigmas"] = sigmas
        kwargs.update(extra_args)
        return kwargs

    raise ValueError(f"Unknown inference family: {family}")


class FluxPerformanceOptimizer:
    """Performance optimization suite for diffusers pipelines."""

    def __init__(self) -> None:
        self.compiled_models: set[int] = set()
        self.cache_hooked_models: set[int] = set()
        self.environment_logged = False

    def setup_environment_optimizations(self) -> None:
        if self.environment_logged:
            return
        for key, value in _DEFAULT_ENV.items():
            logger.info(f"{key}={os.environ.get(key, value)}")
        if torch.cuda.is_available():
            logger.info(
                f"CUDA device={torch.cuda.get_device_name(0)} "
                f"vram={_gpu_vram_gb():.1f}GB torch={torch.__version__}"
            )
        self.environment_logged = True

    def apply_memory_optimizations(self, pipeline: Any, profile: RuntimeProfile) -> None:
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)

        if profile.name == "low_memory" and hasattr(pipeline, "enable_sequential_cpu_offload"):
            pipeline.enable_sequential_cpu_offload()
            logger.info(f"{type(pipeline).__name__}: sequential CPU offload enabled")
        elif profile.cpu_offload and hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
            logger.info(f"{type(pipeline).__name__}: model CPU offload enabled")
        elif profile.device:
            try:
                pipeline.to(profile.device)
                logger.info(f"{type(pipeline).__name__}: moved to {profile.device}")
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"{type(pipeline).__name__}: OOM moving to CUDA; falling back to CPU offload")
                self.clear_memory_aggressively()
                if hasattr(pipeline, "enable_model_cpu_offload"):
                    pipeline.enable_model_cpu_offload()

        if profile.attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            try:
                pipeline.enable_attention_slicing("auto")
                logger.info(f"{type(pipeline).__name__}: attention slicing enabled")
            except Exception as exc:
                logger.warning(f"Could not enable attention slicing: {exc}")

        is_flux_pipeline = "Flux" in type(pipeline).__name__
        use_xformers = profile.xformers and (not is_flux_pipeline or env_bool("SDIF_FORCE_XFORMERS", False))
        if use_xformers and hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info(f"{type(pipeline).__name__}: xFormers attention enabled")
            except Exception as exc:
                logger.warning(f"Could not enable xFormers attention: {exc}")
        elif profile.xformers and is_flux_pipeline:
            logger.info(f"{type(pipeline).__name__}: xFormers skipped for Flux rotary attention")

        if profile.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
            try:
                pipeline.enable_vae_slicing()
                logger.info(f"{type(pipeline).__name__}: VAE slicing enabled")
            except Exception as exc:
                logger.warning(f"Could not enable VAE slicing: {exc}")

        if profile.vae_tiling and hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            try:
                pipeline.vae.enable_tiling()
                logger.info(f"{type(pipeline).__name__}: VAE tiling enabled")
            except Exception as exc:
                logger.warning(f"Could not enable VAE tiling: {exc}")

        for attr in ("unet", "vae"):
            module = getattr(pipeline, attr, None)
            if module is None:
                continue
            try:
                module.to(memory_format=torch.channels_last)
            except Exception:
                pass

    def apply_cache_optimizations(self, pipeline: Any, profile: RuntimeProfile) -> None:
        if profile.cache_mode in {"", "none", "off", "false"}:
            return

        module = getattr(pipeline, "transformer", None)
        if module is None:
            logger.info(f"{type(pipeline).__name__}: no transformer cache target")
            return

        module_id = id(module)
        if module_id in self.cache_hooked_models:
            return

        if profile.cache_mode in {"first_block", "fbc", "fbcache"}:
            try:
                from diffusers import FirstBlockCacheConfig, apply_first_block_cache

                apply_first_block_cache(
                    module,
                    FirstBlockCacheConfig(threshold=profile.first_block_cache_threshold),
                )
                self.cache_hooked_models.add(module_id)
                logger.info(
                    f"{type(pipeline).__name__}: First Block Cache enabled "
                    f"threshold={profile.first_block_cache_threshold}"
                )
            except Exception as exc:
                logger.warning(f"Could not enable First Block Cache: {exc}")
            return

        if profile.cache_mode in {"pab", "pyramid_attention"}:
            try:
                from diffusers import PyramidAttentionBroadcastConfig
                from diffusers.hooks.pyramid_attention_broadcast import apply_pyramid_attention_broadcast

                def current_timestep() -> int:
                    timestep = getattr(pipeline, "current_timestep", 0)
                    if callable(timestep):
                        timestep = timestep()
                    if isinstance(timestep, torch.Tensor):
                        timestep = timestep.detach().float().mean().item()
                    return int(timestep or 0)

                config = PyramidAttentionBroadcastConfig(
                    spatial_attention_block_skip_range=env_int("SDIF_PAB_SKIP_RANGE", 2),
                    spatial_attention_timestep_skip_range=(
                        env_int("SDIF_PAB_TIMESTEP_START", 0),
                        env_int("SDIF_PAB_TIMESTEP_END", 1000),
                    ),
                    current_timestep_callback=current_timestep,
                )
                apply_pyramid_attention_broadcast(module, config)
                self.cache_hooked_models.add(module_id)
                logger.info(f"{type(pipeline).__name__}: Pyramid Attention Broadcast enabled")
            except Exception as exc:
                logger.warning(f"Could not enable Pyramid Attention Broadcast: {exc}")
            return

        if profile.cache_mode in {"faster", "faster_cache"}:
            try:
                from diffusers import FasterCacheConfig, apply_faster_cache

                config = FasterCacheConfig(
                    spatial_attention_block_skip_range=env_int("SDIF_FASTER_CACHE_SKIP_RANGE", 2),
                    tensor_format=os.getenv("SDIF_FASTER_CACHE_TENSOR_FORMAT", "BCHW"),
                    is_guidance_distilled=env_bool("SDIF_FASTER_CACHE_GUIDANCE_DISTILLED", True),
                    current_timestep_callback=lambda: int(getattr(pipeline, "current_timestep", 0) or 0),
                )
                apply_faster_cache(module, config)
                self.cache_hooked_models.add(module_id)
                logger.info(f"{type(pipeline).__name__}: FasterCache enabled")
            except Exception as exc:
                logger.warning(f"Could not enable FasterCache: {exc}")
            return

        logger.warning(f"Unknown SDIF_CACHE_MODE={profile.cache_mode!r}; cache disabled")

    def apply_speed_optimizations(self, pipeline: Any, profile: RuntimeProfile) -> None:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = env_bool("SDIF_ALLOW_TF32", True)
            torch.backends.cudnn.allow_tf32 = env_bool("SDIF_ALLOW_TF32", True)
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            except Exception as exc:
                logger.warning(f"Could not enable CUDA SDP flags: {exc}")

        if not profile.torch_compile or not hasattr(torch, "compile"):
            return

        for attr in ("transformer", "unet"):
            module = getattr(pipeline, attr, None)
            if module is None:
                continue
            module_id = id(module)
            if module_id in self.compiled_models:
                continue
            try:
                compiled = torch.compile(
                    module,
                    mode=os.getenv("SDIF_TORCH_COMPILE_MODE", "reduce-overhead"),
                    fullgraph=profile.torch_compile_fullgraph,
                )
                setattr(pipeline, attr, compiled)
                self.compiled_models.add(module_id)
                logger.info(f"{type(pipeline).__name__}: compiled {attr}")
            except Exception as exc:
                logger.warning(f"torch.compile failed for {attr}: {exc}")

    def optimize_pipeline_fully(self, pipeline: Any, role: str = "pipeline") -> Any:
        if pipeline is None:
            return None
        profile = RuntimeProfile.from_env()
        logger.info(f"Optimizing {role} as {type(pipeline).__name__} with profile={profile.name}")
        self.setup_environment_optimizations()
        self.apply_memory_optimizations(pipeline, profile)
        self.apply_cache_optimizations(pipeline, profile)
        self.apply_speed_optimizations(pipeline, profile)
        self.clear_memory_aggressively()
        return pipeline

    def optimize_inference_settings(self, family: str = "flux", task: str = "text") -> dict[str, Any]:
        return build_inference_kwargs(family, task)

    def get_optimal_batch_size(self, image_size: tuple[int, int] = (512, 512)) -> int:
        if not torch.cuda.is_available():
            return 1

        width, height = image_size
        pixels = width * height
        vram_gb = _gpu_vram_gb()
        if vram_gb >= 24:
            return 4 if pixels <= 512 * 512 else 2
        if vram_gb >= 12:
            return 2 if pixels <= 512 * 512 else 1
        return 1

    @staticmethod
    def clear_memory_aggressively() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        for _ in range(2):
            gc.collect()

    def benchmark_pipeline(
        self,
        pipeline: Any,
        prompt: str = "test prompt",
        runs: int = 3,
        family: str = "flux",
        task: str = "text",
        height: int = 512,
        width: int = 512,
    ) -> dict[str, Any]:
        import time

        settings = self.optimize_inference_settings(family, task)
        times: list[float] = []
        with torch.inference_mode():
            pipeline(prompt=prompt, height=height, width=width, **settings)
            for i in range(runs):
                start_time = time.perf_counter()
                pipeline(prompt=f"{prompt} run {i}", height=height, width=width, **settings)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start_time)

        return {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "runs": runs,
            "settings": settings,
        }


flux_optimizer = FluxPerformanceOptimizer()


def optimize_all_pipelines(base_pipe, img2img_pipe=None, inpaint_pipe=None, canny_pipe=None):
    pipelines = [
        ("base", base_pipe),
        ("img2img", img2img_pipe),
        ("inpaint", inpaint_pipe),
        ("controlnet", canny_pipe),
    ]
    optimized = {}
    for name, pipe in pipelines:
        if pipe is None:
            continue
        optimized[name] = flux_optimizer.optimize_pipeline_fully(pipe, name)
    return optimized


def get_performance_recommendations() -> dict[str, Any]:
    profile = RuntimeProfile.from_env()
    recommendations: dict[str, Any] = {
        "profile": profile.name,
        "cache_mode": profile.cache_mode,
        "flux_settings": flux_optimizer.optimize_inference_settings("flux", "text"),
        "sdxl_settings": flux_optimizer.optimize_inference_settings("sdxl", "text"),
        "batch_size": flux_optimizer.get_optimal_batch_size(),
    }

    if torch.cuda.is_available():
        recommendations["gpu_info"] = {
            "name": torch.cuda.get_device_name(0),
            "vram_gb": _gpu_vram_gb(),
            "compute_capability": torch.cuda.get_device_capability(0),
        }
    return recommendations


if __name__ == "__main__":
    recs = get_performance_recommendations()
    print("Diffusion performance recommendations")
    for key, value in recs.items():
        print(f"{key}: {value}")
