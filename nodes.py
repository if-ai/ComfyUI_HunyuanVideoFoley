import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import tempfile
import numpy as np
from loguru import logger
from typing import Optional, Tuple, Any, List, Dict
import folder_paths
import random
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
import shutil
import time
import math
from tqdm import tqdm
from accelerate import init_empty_weights
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection
import comfy.model_management as mm
import comfy.utils
from safetensors.torch import load_file

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Import ComfyUI video types
try:
    from comfy_api.input_impl import VideoFromFile
except ImportError:
    try:
        # Fallback to latest API location
        from comfy_api.latest._input_impl.video_types import VideoFromFile
    except ImportError:
        logger.warning("VideoFromFile not available, will return file paths only")
        VideoFromFile = None

# Add foley models directory to ComfyUI folder paths
foley_models_dir = os.path.join(folder_paths.models_dir, "foley")
if "foley" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["foley"] = ([foley_models_dir], folder_paths.supported_pt_extensions)

# Import the HunyuanVideo-Foley modules
try:
    from hunyuanvideo_foley.utils.model_utils import load_model
    from .utils import denoise_process_safely
    from .utils import feature_process_unified, extract_video_path, create_node_exit_values
    from hunyuanvideo_foley.utils.media_utils import merge_audio_video
    from .model_urls import get_model_url 
except ImportError as e:
    logger.error(f"Failed to import HunyuanVideo-Foley modules: {e}")
    logger.error("Make sure the HunyuanVideo-Foley package is installed and accessible")
    raise

class HunyuanVideoFoleyNode:
    """
    A node for generating audio using the HunyuanVideo-Foley model
    """
    # Class-level model storage for persistence
    _model_dict = None
    _cfg = None
    _device = None
    _model_path = None
    _memory_efficient = False  # Track memory mode
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_prompt": ("STRING", {
                    "multiline": True,
                    "default": "footstep sound, impact, water splash",
                    "display": "textarea",
                    "placeholder": "Describe the audio you want to generate..."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.5,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "display": "slider"
                }),
                "sample_nums": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "display": "number"
                }),
            },
            "optional": {
                # Hybrid input support - either VIDEO or IMAGE
                "video": ("VIDEO",),
                "images": ("IMAGE",),  # Support for frame sequences from PR
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Frames per second (only used with IMAGE input)"
                }),
                # Negative prompt from PR
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "display": "textarea",
                    "placeholder": "Additional negative prompts (optional). Will be combined with built-in quality controls."
                }),
                # Output options
                "output_format": (["video_path", "frames", "both"], {
                    "default": "both",
                    "tooltip": "Choose output format: video file path only works with VIDEO input, frames only works with IMAGE input and VIDEO input"
                }),
                "output_folder": ("STRING", {
                    "default": "hunyuan_foley",
                    "multiline": False,
                    "display": "text",
                    "placeholder": "Subfolder name in ComfyUI/output/"
                }),
                "filename_prefix": ("STRING", {
                    "default": "foley_",
                    "multiline": False,
                    "display": "text",
                    "placeholder": "Prefix for output filename"
                }),
                # Memory optimization options
                "memory_efficient": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox",
                    "tooltip": "Enable memory-efficient mode: unloads models after generation and uses aggressive garbage collection"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "display": "checkbox",
                    "tooltip": "Offload models to CPU when not in use (slower but saves VRAM)"
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox",
                    "tooltip": "Enable or disable the entire audio generation process. If false, returns a silent or null audio output."
                }),
                "silent_audio": ("BOOLEAN", {
                    "default": True,
                    "display": "checkbox",
                    "tooltip": "If true, returns a silent audio clip when disabled or on failure. If false, returns None."
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("video_path", "video_frames", "audio", "status_message")
    FUNCTION = "generate_audio"
    CATEGORY = "HunyuanVideo-Foley"
    DESCRIPTION = "Generate synchronized audio for videos using HunyuanVideo-Foley model"
    
    @classmethod
    def download_models(cls, model_name: str, destination_dir: str) -> Tuple[bool, str]:
        """Download models from URLs if they don't exist."""
        model_info = get_model_url(model_name)
        if not model_info:
            return False, f"Model '{model_name}' not found in configuration."

        os.makedirs(destination_dir, exist_ok=True)

        for model_file in model_info.get("models", []):
            filename = model_file.get("filename")
            url = model_file.get("url")

            if not filename or not url:
                logger.warning("Skipping a model file due to missing filename or URL.")
                continue

            destination_path = os.path.join(destination_dir, filename)

            if os.path.exists(destination_path):
                logger.info(f"'{filename}' already exists. Skipping download.")
                continue

            logger.info(f"Downloading '{filename}' from '{url}'...")

            try:
                comfy.utils.pb_download(url, destination_path)
            except Exception as e:
                return False, f"Failed to download '{filename}': {str(e)}"

        return True, "All models downloaded or already exist."
        
        @classmethod
        def setup_device(cls, device_type: str = "auto", device_id: int = 0):
            """Setup and return the appropriate device with memory optimization"""
            # Try to import ComfyUI's model management if available
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
                logger.info(f"Using ComfyUI device: {device}")
                return device
            except:
                pass
            
            if device_type == "auto":
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{device_id}")
                    # Clear cache before loading
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats(device)
                else:
                    device = torch.device("cpu")
            else:
                device = torch.device(device_type)
            
            logger.info(f"Using device: {device}")
            return device
    
    @classmethod
    def load_models(cls, model_path: str = "", config_path: str = "", 
                   memory_efficient: bool = False, cpu_offload: bool = False) -> Tuple[bool, str]:
        """Load models if not already loaded or if path changed"""
        try:

            # Your proactive VRAM cleanup
            if cpu_offload or memory_efficient:
                logger.info("Proactively unloading other models to make space...")
                mm.unload_all_models()
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache() 

            # His default path logic
            if not model_path.strip():
                foley_models_dir = folder_paths.get_folder_paths("foley")[0]
                model_path = os.path.join(foley_models_dir, "hunyuanvideo-foley-xxl")

            # His auto-downloader logic
            if not os.path.exists(model_path):
                logger.info(f"Model directory not found at '{model_path}'. Attempting to download...")
                success, message = cls.download_models("hunyuanvideo-foley-xxl", model_path)
                if not success:
                    return False, f"Auto-download failed: {message}"
            
            # Your path logic
            if not config_path.strip():
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, "configs", "hunyuanvideo-foley-xxl.yaml")
            
            # Your caching check
            if (cls._model_dict is not None and 
                cls._cfg is not None and 
                (cls._model_path == model_path or cls._model_path == "preloaded") and
                cls._memory_efficient == memory_efficient):
                return True, "Models already loaded"
            
            # Your setup logic
            cls._device = cls.setup_device("auto", 0)
            logger.info(f"Loading models from: {model_path}")
            logger.info(f"Config: {config_path}")
            
            # Your call to the memory-safe loader in utils
            cls._model_dict, cls._cfg = load_model(model_path, config_path, cls._device)
            
            # Your state management logic
            if cls._model_path != "preloaded":
                cls._model_path = model_path
            cls._memory_efficient = memory_efficient
            
            logger.info("Models loaded successfully!")
            return True, "Models loaded successfully!"
        
        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}"
            logger.error(error_msg)
            cls._model_dict = None
            cls._cfg = None
            cls._device = None
            cls._model_path = None
            return False, error_msg
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        # Clamp seed to valid range for numpy (0 to 2^32-1)
        seed = int(seed) % (2**32)
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    

    @classmethod
    def _extract_frames_from_image_input(cls, images, fps=24.0):
        """Convert IMAGE input to video frames and create a temporary video file"""
        import cv2
        
        try:
            if images is None:
                return None, "No images provided"
            
            # Handle different image input formats
            if hasattr(images, 'shape'):
                # Tensor input [batch, height, width, channels]
                if len(images.shape) == 4:
                    frames = images.cpu().numpy()
                else:
                    return None, f"Unexpected image tensor shape: {images.shape}"
            else:
                return None, f"Unsupported image input type: {type(images)}"
            
            # Convert to uint8 if needed
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = frames.astype(np.uint8)
            
            if frames.shape[-1] == 4:
                frames = frames[..., :3]
                
            # Get dimensions
            batch_size, height, width = frames.shape[:3]
            
            # Create temporary video file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_fd)
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                return None, "Failed to open video writer"
            
            # Write frames
            for i in range(batch_size):
                frame = frames[i]
                
                # Handle channels - convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                out.write(frame)
            
            out.release()
            
            logger.info(f"Created temporary video from {batch_size} frames at {fps} FPS: {temp_path}")
            return temp_path, "Success"
            
        except Exception as e:
            return None, f"Error converting images to video: {str(e)}"
    
    # In HunyuanVideoFoleyNode class:
    @torch.inference_mode()
    def generate_audio(self, text_prompt: str, guidance_scale: float, 
                      num_inference_steps: int, sample_nums: int, seed: int,
                      video=None, images=None, fps=24.0,
                      negative_prompt="",
                      output_format="video_path",
                      output_folder: str = "hunyuan_foley",
                      filename_prefix: str = "foley_",
                      memory_efficient: bool = False,
                      cpu_offload: bool = False,
                      enabled: bool = True,
                      silent_audio: bool = True):
        """Generate audio for the input video/images with the given text prompt"""
        
        # 1. Enabled/Disabled Short-Circuit with Intelligent Passthrough
        if not enabled:
            logger.info("HunyuanVideo-Foley node is disabled. Passing through inputs.")
            return create_node_exit_values(
                silent_audio=silent_audio,
                passthrough_video=video,
                passthrough_images=images,
                message="✅ Node disabled. Skipped."
            )
        
                # In the generate_audio function:
        try:
            self.set_seed(seed)

            # Proactive VRAM Cleanup (This part is correct)
            logger.info("Performing pre-run VRAM cleanup...")
            mm.unload_all_models()
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # --- CORRECTED MODEL LOADING LOGIC ---
            # This is the single source of truth for loading/validating models.
            if self._model_dict is None or self._cfg is None:
                logger.info("No models loaded, attempting to load fresh...")
                success, message = self.load_models(memory_efficient=memory_efficient, cpu_offload=cpu_offload)
                if not success:
                    raise Exception(f"Model loading failed: {message}")
            else:
                logger.info("Using pre-loaded models from class state.")
            
            # Final validation check after attempting to load/use pre-loaded.
            if self._model_dict is None or self._cfg is None:
                raise Exception("Models not loaded, and fallback loading failed.")
            # --- END OF FIX ---

            # Validate inputs
            if video is None and images is None:
                raise Exception("Please provide either video or images input!")
            
            # Determine video file path
            logger.info("Processing media features...")
            visual_feats, text_feats, audio_len_in_s = feature_process_unified(
                video_input=video,
                image_input=images,
                prompt=text_prompt,
                negative_prompt=negative_prompt,
                model_dict=self._model_dict,
                cfg=self._cfg,
                fps_hint=fps
            )
            
            # Ensure the core models for denoising are on the correct device before processing.
            # --- State Correction and VRAM Management for Denoising ---
            target_device = self._device
            logger.info(f"Preparing for denoising on device: {target_device}")

            # 1. Offload the large feature tensors to CPU to make space.
            visual_feats['siglip2_feat'] = visual_feats['siglip2_feat'].to("cpu")
            visual_feats['syncformer_feat'] = visual_feats['syncformer_feat'].to("cpu")
            text_feats['text_feat'] = text_feats['text_feat'].to("cpu")
            text_feats['uncond_text_feat'] = text_feats['uncond_text_feat'].to("cpu")
            logger.info("Feature tensors moved to CPU.")

            # 2. Now that VRAM is clear, move the core models to the GPU.
            self._model_dict.foley_model.to(target_device)
            self._model_dict.dac_model.to(target_device)
            logger.info("Core models moved to GPU.")
            # --- End of Fix ---

            logger.info("Generating audio...")
            audio, sample_rate = denoise_process_safely(
                visual_feats, text_feats, audio_len_in_s, self._model_dict, self._cfg,
                guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, batch_size=sample_nums
            )
            
            output_dir = os.path.join(folder_paths.get_output_directory(), output_folder)
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_filename = f"{filename_prefix}audio_{timestamp}_{seed}.wav"
            audio_output_path = os.path.join(output_dir, audio_filename)
            torchaudio.save(audio_output_path, audio[0], sample_rate)
            
            audio_tensor = audio[0].unsqueeze(0)
            if len(audio_tensor.shape) == 2:
                audio_tensor = audio_tensor.unsqueeze(1)
            audio_result = {"waveform": audio_tensor, "sample_rate": sample_rate}
            
            video_file = None
            temp_video_created = False
            
            if video is not None:
                video_file = extract_video_path(video)
            elif images is not None:
                # If the input was an image tensor, we must create a temporary video file
                # so that merge_audio_video has a video stream to work with.
                video_file, _ = self._extract_frames_from_image_input(images, fps)
                temp_video_created = True

            if video_file is None:
                logger.warning("No valid video file path found for output merging/frame extraction.")
                
            video_output_path = ""
            video_frames = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            
            if output_format in ["video_path", "both"]:
                video_filename = f"{filename_prefix}video_{timestamp}_{seed}.mp4"
                video_output_path = os.path.join(output_dir, video_filename)
                try:
                    merge_audio_video(audio_output_path, video_file, video_output_path)
                    logger.info(f"Created video with audio: {video_output_path}")
                except Exception as e:
                    logger.error(f"Failed to merge audio and video: {e}")
                    video_output_path = video_file
            
            if output_format in ["frames", "both"]:
                if images is not None:
                    # If the input was an image tensor, just pass it through.
                    video_frames = images
                elif video_file:
                    # If the input was a video file, extract the frames.
                    try:
                        import cv2
                        cap = cv2.VideoCapture(video_file)
                        frames_list = []
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames_list.append(np.array(frame_rgb, dtype=np.float32) / 255.0)
                        cap.release()
                        if frames_list:
                            video_frames = torch.from_numpy(np.stack(frames_list))
                            logger.info(f"Extracted {len(frames_list)} frames for output")
                    except Exception as e:
                        logger.warning(f"Could not extract frames for output: {e}")
            
            if temp_video_created and os.path.exists(video_file):
                try: os.remove(video_file)
                except: pass
            
            if memory_efficient:
                # Only unload completely if the model was NOT preloaded by an upstream node.
                if self._model_path != "preloaded":
                    logger.info("Memory efficient mode: Unloading models completely.")
                    self._model_dict = None
                    self._cfg = None
                    self._model_path = None
                else:
                    logger.info("Memory efficient mode: Models are preloaded, skipping complete unload.")
            
            success_msg = f"✅ Generated audio successfully"
            return (video_output_path, video_frames, audio_result, success_msg)
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Generation failed: {str(e)}"
            logger.error(error_msg); logger.error(traceback.format_exc())
            return create_node_exit_values(
                silent_audio=silent_audio,
                passthrough_video=video,
                passthrough_images=images,
                message=error_msg
            )
        
        finally:
            logger.info("HunyuanVideo-Foley: Starting guaranteed cleanup...")
            if (memory_efficient or cpu_offload) and self._model_dict:
                logger.info("Offloading models to CPU...")
                for model_obj in self._model_dict.values():
                    if hasattr(model_obj, 'to'):
                        try: model_obj.to("cpu")
                        except Exception: pass
            
            if memory_efficient and self._model_path != "preloaded":
                logger.info("Memory efficient mode: Unloading models completely.")
                self._model_dict, self._cfg, self._model_path = None, None, None

            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("HunyuanVideo-Foley: Cleanup complete.")
        
class LinearFP8Wrapper(nn.Module):
    """FP8 quantization wrapper for linear layers"""
    def __init__(self, original_linear, dtype="fp8_e4m3fn"):
        super().__init__()
        self.dtype = dtype
        self.weight_shape = original_linear.weight.shape
        self.bias = original_linear.bias
        
        # Quantize weights to FP8
        if dtype == "fp8_e4m3fn":
            self.weight_fp8 = original_linear.weight.to(torch.float8_e4m3fn)
        elif dtype == "fp8_e5m2":
            self.weight_fp8 = original_linear.weight.to(torch.float8_e5m2)
        else:
            self.weight_fp8 = original_linear.weight
    
    def forward(self, x):
        # Convert back to computation dtype for matmul
        weight = self.weight_fp8.to(x.dtype)
        return F.linear(x, weight, self.bias)

class HunyuanVideoFoleyModelLoader:
    """Separate model loader with FP8 quantization support"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "quantization": (["none", "fp8_e4m3fn", "fp8_e5m2"], {
                    "default": "none",
                    "tooltip": "FP8 weight-only quantization for VRAM savings"
                }),
                "cpu_offload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload models to CPU when not in use"
                }),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to model weights (leave empty for default)"
                }),
                "config_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to config file (leave empty for default)"
                }),
            }
        }
    
    RETURN_TYPES = ("FOLEY_MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    FUNCTION = "load_model"
    CATEGORY = "HunyuanVideo-Foley/Loaders"
    DESCRIPTION = "Load HunyuanVideo-Foley model with optional FP8 quantization"
    
    def load_model(self, quantization="none", cpu_offload=False, model_path="", config_path=""):
        try:
            # Set default paths
            if not model_path or not model_path.strip():
                foley_models_dir = folder_paths.folder_names_and_paths.get("foley", [None])[0]
                if foley_models_dir and len(foley_models_dir) > 0:
                    # Check for the model in the hunyuanvideo-foley-xxl subdirectory
                    model_path = os.path.join(foley_models_dir[0], "hunyuanvideo-foley-xxl")
                else:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(current_dir, "pretrained_models")
            
            if not config_path or not config_path.strip():
                current_dir = os.path.dirname(os.path.abspath(__file__))
                config_path = os.path.join(current_dir, "configs", "hunyuanvideo-foley-xxl.yaml")
            
            logger.info(f"Model path: {model_path}")
            logger.info(f"Config path: {config_path}")
            
            # Setup device
            device = mm.get_torch_device()
            
            logger.info(f"Loading model with quantization={quantization}")
            
            # Load model and config
            model_dict, cfg = load_model(model_path, config_path, device)
            
            # Apply FP8 quantization if requested
            if quantization != "none" and "vae" in model_dict:
                logger.info(f"Applying {quantization} quantization to VAE...")
                vae_model = model_dict["vae"]
                
                # Quantize linear layers in VAE
                for name, module in vae_model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Replace with FP8 wrapper
                        parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
                        child_name = name.split(".")[-1]
                        parent = vae_model if not parent_name else dict(vae_model.named_modules())[parent_name]
                        setattr(parent, child_name, LinearFP8Wrapper(module, quantization))
                
                logger.info("FP8 quantization applied")
            
            # Package model info
            model_info = {
                "model_dict": model_dict,
                "cfg": cfg,
                "device": device,
                "quantization": quantization,
                "cpu_offload": cpu_offload
            }
            
            status = f"✅ Model loaded with {quantization} quantization" if quantization != "none" else "✅ Model loaded"
            return (model_info, status)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return (None, f"❌ Failed to load model: {str(e)}")

class HunyuanVideoFoleyDependenciesLoader:
    """Load text encoder and feature extractors separately"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FOLEY_MODEL",),
                "load_text_encoder": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load CLAP text encoder"
                }),
                "load_feature_extractor": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load visual feature extractor"
                }),
            }
        }
    
    RETURN_TYPES = ("FOLEY_DEPS", "STRING")
    RETURN_NAMES = ("dependencies", "status")
    FUNCTION = "load_dependencies"
    CATEGORY = "HunyuanVideo-Foley/Loaders"
    DESCRIPTION = "Load model dependencies (text encoder, feature extractors)"
    
    def load_dependencies(self, model, load_text_encoder=True, load_feature_extractor=True):
        try:
            if model is None:
                return (None, "❌ No model provided - check ModelLoader output")
            
            # Check if model is a tuple (from failed load)
            if isinstance(model, tuple):
                model = model[0]
            
            if model is None or not isinstance(model, dict):
                return (None, "❌ Invalid model input - ModelLoader may have failed")
            
            deps = {
                "model_dict": model["model_dict"],
                "cfg": model["cfg"],
                "device": model["device"]
            }
            
            status_parts = []
            
            if load_text_encoder:
                logger.info("Loading text encoder...")
                # Text encoder is already in model_dict
                status_parts.append("text encoder")
            
            if load_feature_extractor:
                logger.info("Loading feature extractor...")
                # Feature extractor is already in model_dict
                status_parts.append("feature extractor")
            
            status = f"✅ Loaded: {', '.join(status_parts)}" if status_parts else "✅ Dependencies ready"
            return (deps, status)
            
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            return (None, f"❌ Failed: {str(e)}")

class HunyuanVideoFoleyTorchCompile:
    """Apply torch.compile optimization to the model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dependencies": ("FOLEY_DEPS",),
                "compile_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Compile VAE with torch.compile for ~30% speedup"
                }),
                "compile_mode": (["default", "reduce-overhead", "max-autotune"], {
                    "default": "default",
                    "tooltip": "Compilation mode (default is fastest compile time)"
                }),
                "backend": (["inductor", "cudagraphs", "eager"], {
                    "default": "inductor",
                    "tooltip": "Compilation backend"
                }),
            }
        }
    
    RETURN_TYPES = ("FOLEY_COMPILED", "STRING")
    RETURN_NAMES = ("compiled_model", "status")
    FUNCTION = "compile_model"
    CATEGORY = "HunyuanVideo-Foley/Optimization"
    DESCRIPTION = "Optimize model with torch.compile for faster inference"
    
    def compile_model(self, dependencies, compile_vae=True, compile_mode="default", backend="inductor"):
        try:
            if dependencies is None:
                return (None, "❌ No dependencies provided - check DependenciesLoader output")
            
            # Check if dependencies is a tuple (from failed load)
            if isinstance(dependencies, tuple):
                dependencies = dependencies[0]
            
            if dependencies is None or not isinstance(dependencies, dict):
                return (None, "❌ Invalid dependencies - previous node may have failed")
            
            # Handle AttributeDict or regular dict
            model_dict = dependencies["model_dict"]
            # Don't copy AttributeDict, just reference it
            if hasattr(model_dict, '__class__') and 'AttributeDict' in str(model_dict.__class__):
                model_dict_ref = model_dict
            else:
                model_dict_ref = model_dict.copy() if hasattr(model_dict, 'copy') else model_dict
            
            compiled = {
                "model_dict": model_dict_ref,
                "cfg": dependencies["cfg"],
                "device": dependencies["device"],
                "compiled": False
            }
            
            # Check for dac_model (the actual VAE used in HunyuanVideo-Foley)
            model_dict = compiled["model_dict"]
            has_dac = hasattr(model_dict, 'dac_model') or (isinstance(model_dict, dict) and "dac_model" in model_dict)
            
            if compile_vae and has_dac:
                logger.info(f"Compiling DAC VAE with mode={compile_mode}, backend={backend}...")
                
                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                
                # Only compile if backend is not eager
                if backend != "eager":
                    # Access dac_model correctly whether it's dict or AttributeDict
                    if hasattr(model_dict, 'dac_model'):
                        dac_model = model_dict.dac_model
                    else:
                        dac_model = model_dict["dac_model"]
                    
                    compiled_dac = torch.compile(
                        dac_model,
                        mode=compile_mode,
                        backend=backend
                    )
                    
                    # Set the compiled model back
                    if hasattr(model_dict, 'dac_model'):
                        model_dict.dac_model = compiled_dac
                    else:
                        model_dict["dac_model"] = compiled_dac
                    
                    compiled["compiled"] = True
                    status = f"✅ DAC VAE compiled with {compile_mode}/{backend}"
                else:
                    status = "✅ Model ready (eager mode - no compilation)"
            else:
                status = "✅ Model ready (no VAE compilation)"
            
            return (compiled, status)
            
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            # Return uncompiled model on failure
            if dependencies:
                return (dependencies, f"⚠️ Compilation failed, using uncompiled: {str(e)}")
            return (None, f"❌ Failed: {str(e)}")

class HunyuanVideoFoleyGeneratorAdvanced(HunyuanVideoFoleyNode):
    """Enhanced generator that can use separately loaded models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        
        # Add optional compiled model input
        base_inputs["optional"]["compiled_model"] = ("FOLEY_COMPILED",)
        # --- QoL FIELDS HERE ---
        base_inputs["optional"]["enabled"] = ("BOOLEAN", {
            "default": True,
            "display": "checkbox",
            "tooltip": "Enable or disable this node. If disabled, it will pass through silent audio."
        })
        base_inputs["optional"]["silent_audio"] = ("BOOLEAN", {
            "default": True,
            "display": "checkbox",
            "tooltip": "If true, returns a silent audio clip when disabled or on failure. If false, returns None."
        })
        # -----------------------------

        return base_inputs
    
    FUNCTION = "generate_audio_advanced"
    CATEGORY = "HunyuanVideo-Foley"
    DESCRIPTION = "Generate audio with optional pre-loaded/optimized models"
    
    def generate_audio_advanced(self, **kwargs):
    
        compiled_model = kwargs.get("compiled_model")

        if compiled_model:
            if isinstance(compiled_model, tuple): compiled_model = compiled_model[0]
            if compiled_model and isinstance(compiled_model, dict):
                self.__class__._model_dict = compiled_model.get("model_dict")
                self.__class__._cfg = compiled_model.get("cfg")
                self.__class__._device = compiled_model.get("device")
                self.__class__._model_path = "preloaded"
                self.__class__._memory_efficient = kwargs.get("memory_efficient", False)
                logger.info("Using pre-loaded/compiled model from input.")
            else:
                logger.warning("Invalid compiled_model input. Clearing state to force reload.")
                self.__class__._model_dict, self.__class__._cfg, self.__class__._model_path = None, None, None
        else:
            logger.info("No pre-loaded model detected. Clearing state to force fresh load.")
            self.__class__._model_dict, self.__class__._cfg, self.__class__._model_path = None, None, None
            
        return self.generate_audio(**kwargs)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "HunyuanVideoFoley": HunyuanVideoFoleyNode,
    "HunyuanVideoFoleyModelLoader": HunyuanVideoFoleyModelLoader,
    "HunyuanVideoFoleyDependenciesLoader": HunyuanVideoFoleyDependenciesLoader,
    "HunyuanVideoFoleyTorchCompile": HunyuanVideoFoleyTorchCompile,
    "HunyuanVideoFoleyGeneratorAdvanced": HunyuanVideoFoleyGeneratorAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanVideoFoley": "HunyuanVideo-Foley Generator",
    "HunyuanVideoFoleyModelLoader": "HunyuanVideo-Foley Model Loader (FP8)",
    "HunyuanVideoFoleyDependenciesLoader": "HunyuanVideo-Foley Dependencies",
    "HunyuanVideoFoleyTorchCompile": "HunyuanVideo-Foley Torch Compile",
    "HunyuanVideoFoleyGeneratorAdvanced": "HunyuanVideo-Foley Generator (Advanced)",
}