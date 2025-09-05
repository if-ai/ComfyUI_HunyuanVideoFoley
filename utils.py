"""
Utility functions for ComfyUI HunyuanVideo-Foley custom node
"""

import os
import tempfile
import torch
import numpy as np
from typing import Union, Optional, Tuple
from loguru import logger
import decord
from tqdm import tqdm
from PIL import Image
from einops import rearrange
import comfy.model_management as mm
# We need to import the original library functions that our safe wrappers will call.
from hunyuanvideo_foley.utils.feature_utils import encode_text_feat, encode_video_with_siglip2, encode_video_with_sync
from hunyuanvideo_foley.utils.config_utils import AttributeDict


def tensor_to_video(video_tensor: torch.Tensor, output_path: str, fps: int = 30) -> str:
    """
    Convert a video tensor to a video file
    """
    try:
        import cv2
        video_np = video_tensor.detach().cpu().numpy() if isinstance(video_tensor, torch.Tensor) else np.array(video_tensor)
        
        if video_np.ndim == 4 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np, (0, 2, 3, 1))
        elif video_np.ndim == 5 and video_np.shape[1] == 3:
            video_np = np.transpose(video_np[0], (0, 2, 3, 1))
        
        video_np = (video_np * 255).astype(np.uint8) if video_np.max() <= 1.0 else video_np.astype(np.uint8)
        
        frames, height, width, channels = video_np.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i in range(frames):
            frame = video_np[i]
            if channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        logger.info(f"Video saved to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to convert tensor to video: {e}")
        raise

def get_video_info(video_path: str) -> dict:
    """
    Get information about a video file
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
        cap.release()
        return info
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}

def ensure_video_file(video_input: Union[str, torch.Tensor, np.ndarray]) -> str:
    """
    Ensure the video input is converted to a file path
    """
    if isinstance(video_input, str):
        if os.path.exists(video_input): return video_input
        else: raise FileNotFoundError(f"Video file not found: {video_input}")
    elif isinstance(video_input, (torch.Tensor, np.ndarray)):
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "input_video.mp4")
        return tensor_to_video(video_input, output_path)
    else:
        raise ValueError(f"Unsupported video input type: {type(video_input)}")

def validate_model_files(model_path: str) -> Tuple[bool, str]:
    """
    Validate that all required model files exist
    """
    required_files = ["hunyuanvideo_foley.pth", "vae_128d_48k.pth", "synchformer_state_dict.pth"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        return False, f"Missing model files: {', '.join(missing_files)}"
    return True, "All required model files found"

def get_optimal_device() -> torch.device:
    """
    Get the optimal device for model execution
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        props = torch.cuda.get_device_properties(device)
        logger.info(f"Using CUDA device: {props.name} with {props.total_memory / 1e9:.1f}GB memory")
        return device
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Using MPS device (Apple Silicon)")
        return torch.device("mps")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")

def check_memory_requirements(device: torch.device, required_gb: float = 16.0) -> Tuple[bool, str]:
    """
    Check if the device has enough memory for model execution
    """
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        if total_memory < required_gb:
            return False, f"GPU has {total_memory:.1f}GB memory, but {required_gb}GB is recommended"
        else:
            return True, f"GPU has {total_memory:.1f}GB memory (sufficient)"
    return True, "Using non-CUDA device (memory check not applicable)"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable format
    """
    seconds = float(seconds)
    if seconds < 60: return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60: return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"

# --- NEW MEMORY-SAFE AND UNIFIED FUNCTIONS ---

def extract_video_path(video):
    """Extract a file path from various potential video input types. Moved here to be a standalone utility."""
    if hasattr(video, '__class__') and 'VideoFromFile' in video.__class__.__name__:
        if hasattr(video, '_VideoFromFile__file'): return getattr(video, '_VideoFromFile__file')
        for attr in ['file', 'path', 'filename']:
            if hasattr(video, attr) and isinstance(getattr(video, attr), str): return getattr(video, attr)
    if isinstance(video, str): return video
    if isinstance(video, dict) and 'path' in video: return video['path']
    return None

@torch.inference_mode()
def _encode_visual_features_safely(frames_uint8, model_dict, fps_hint):
    """ Memory-safe visual feature extraction using batched preprocessing and inference. """
    dev = model_dict.device
    
    logger.info("Aggressively clearing VRAM before feature extraction...")
    import gc
    mm.unload_all_models(); gc.collect()
    if dev.type == 'cuda': torch.cuda.empty_cache()
    
    visual_features = {}
    pil_list = [Image.fromarray(f).convert("RGB") for f in frames_uint8]
    
    # Process SigLIP2
    try:
        logger.info("Moving SigLIP2 to device for feature extraction...")
        model_dict.siglip2_model.to(dev)
        
        # Use the batched helper to create the tensor without OOM
        clip_frames = _batched_frames_to_tensor(pil_list, model_dict.siglip2_preprocess, 16, dev)
        
        # Use a safe, fixed batch size for inference, as the smart_batcher is too complex for now.
        visual_features['siglip2_feat'] = encode_video_with_siglip2(clip_frames, model_dict, batch_size=4).to(dev)
        del clip_frames
    finally:
        logger.info("Offloading SigLIP2 from device."); model_dict.siglip2_model.to("cpu")
        if dev.type == 'cuda': torch.cuda.empty_cache()

    # Process Syncformer
    try:
        logger.info("Moving Syncformer to device for feature extraction...")
        model_dict.syncformer_model.to(dev)
        
        # Syncformer's preprocess can often handle a full tensor, but we'll be safe
        # and use the same batched creation method.
        sync_frames = _batched_frames_to_tensor(pil_list, model_dict.syncformer_preprocess, 16, dev)

        visual_features['syncformer_feat'] = encode_video_with_sync(sync_frames, model_dict)
        del sync_frames
    finally:
        logger.info("Offloading Syncformer from device."); model_dict.syncformer_model.to("cpu")
        if dev.type == 'cuda': torch.cuda.empty_cache()

    audio_len_in_s = len(frames_uint8) / fps_hint
    return AttributeDict(visual_features), audio_len_in_s

def feature_process_unified(video_input, image_input, prompt, model_dict, cfg, negative_prompt="", fps_hint=24.0, max_frames=450):
    """ Unified, memory-safe feature processing for either a video path or an image tensor. """
    frames_uint8 = None; fps = fps_hint
    
    if image_input is not None:
        logger.info("Processing features from IMAGE tensor input.")
        frames_uint8 = (image_input.cpu().clip(0, 1) * 255).byte().numpy() if image_input.dtype != torch.uint8 else image_input.cpu().numpy()
    elif video_input is not None:
        logger.info("Processing features from VIDEO path input.")
        video_path = extract_video_path(video_input)
        if not video_path: raise ValueError("Invalid video path provided.")
        
        video_reader = decord.VideoReader(video_path)
        total_frames = len(video_reader)
        fps = video_reader.get_avg_fps()
        
        frame_indices = np.linspace(0, total_frames - 1, num=min(total_frames, max_frames), dtype=int)
        frames_uint8 = video_reader.get_batch(frame_indices).asnumpy()

    if frames_uint8 is None: raise ValueError("No valid video or image frames to process.")

    visual_feats, audio_len_in_s = _encode_visual_features_safely(frames_uint8, model_dict, fps)

    # Use the provided negative prompt, or fall back to a default.
    neg_prompt = negative_prompt if negative_prompt and negative_prompt.strip() else "noisy, harsh"
    
    prompts = [neg_prompt, prompt]
    clap_model = model_dict.clap_model
    try:
        clap_model.to(model_dict.device)
        text_feat_res, _ = encode_text_feat(prompts, model_dict)
    finally:
        clap_model.to("cpu")

    text_feat = text_feat_res[1:]; uncond_text_feat = text_feat_res[:1]
    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = AttributeDict({'text_feat': text_feat, 'uncond_text_feat': uncond_text_feat})
    return visual_feats, text_feats, audio_len_in_s

# In utils.py, add this entire block at the end of the file

from diffusers.utils.torch_utils import randn_tensor
from hunyuanvideo_foley.utils.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

def _retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    """ Helper function forked from the library's pipeline.py """
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def _prepare_latents(scheduler, batch_size, num_channels_latents, length, dtype, device):
    """ Helper function forked from the library's pipeline.py """
    shape = (batch_size, num_channels_latents, int(length))
    latents = randn_tensor(shape, device=device, dtype=dtype)
    if hasattr(scheduler, "init_noise_sigma"):
        latents = latents * scheduler.init_noise_sigma
    return latents

def denoise_process_safely(visual_feats, text_feats, audio_len_in_s, model_dict, cfg, **kwargs):
    """ Memory-safe fork of denoise_process that ensures all tensors are on the correct device. """
    target_dtype = model_dict.foley_model.dtype
    autocast_enabled = target_dtype != torch.float32
    device = model_dict.device

    # --- Device Correction Fix ---
    visual_feats.siglip2_feat = visual_feats.siglip2_feat.to(device)
    visual_feats.syncformer_feat = visual_feats.syncformer_feat.to(device)
    text_feats.text_feat = text_feats.text_feat.to(device)
    text_feats.uncond_text_feat = text_feats.uncond_text_feat.to(device)
    # --- End of Fix ---
    
    guidance_scale = kwargs.get('guidance_scale', 4.5)
    num_inference_steps = kwargs.get('num_inference_steps', 50)
    batch_size = kwargs.get('batch_size', 1)

    scheduler = FlowMatchDiscreteScheduler(
        shift=cfg.diffusion_config.sample_flow_shift,
        reverse=cfg.diffusion_config.flow_reverse,
        solver=cfg.diffusion_config.flow_solver,
        use_flux_shift=cfg.diffusion_config.sample_use_flux_shift,
        flux_base_shift=cfg.diffusion_config.flux_base_shift,
        flux_max_shift=cfg.diffusion_config.flux_max_shift,
    )

    timesteps, _ = _retrieve_timesteps(scheduler, num_inference_steps, device)

    latents = _prepare_latents(
        scheduler, batch_size=batch_size,
        num_channels_latents=cfg.model_config.model_kwargs.audio_vae_latent_dim,
        length=audio_len_in_s * cfg.model_config.model_kwargs.audio_frame_rate,
        dtype=target_dtype, device=device,
    )

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising steps"):
        latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_input = scheduler.scale_model_input(latent_input, t)
        t_expand = t.repeat(latent_input.shape[0])

        siglip2_feat = visual_feats.siglip2_feat.repeat(batch_size, 1, 1)
        uncond_siglip2_feat = model_dict.foley_model.get_empty_clip_sequence(bs=batch_size, len=siglip2_feat.shape[1]).to(device)
        siglip2_feat_input = torch.cat([uncond_siglip2_feat, siglip2_feat], dim=0) if guidance_scale > 1.0 else siglip2_feat

        syncformer_feat = visual_feats.syncformer_feat.repeat(batch_size, 1, 1)
        uncond_syncformer_feat = model_dict.foley_model.get_empty_sync_sequence(bs=batch_size, len=syncformer_feat.shape[1]).to(device)
        syncformer_feat_input = torch.cat([uncond_syncformer_feat, syncformer_feat], dim=0) if guidance_scale > 1.0 else syncformer_feat

        text_feat_repeated = text_feats.text_feat.repeat(batch_size, 1, 1)
        uncond_text_feat_repeated = text_feats.uncond_text_feat.repeat(batch_size, 1, 1)
        text_feat_input = torch.cat([uncond_text_feat_repeated, text_feat_repeated], dim=0) if guidance_scale > 1.0 else text_feat_repeated

        with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=target_dtype):
            noise_pred = model_dict.foley_model(
                x=latent_input, t=t_expand, cond=text_feat_input,
                clip_feat=siglip2_feat_input, sync_feat=syncformer_feat_input,
                return_dict=True,
            )["x"]

        noise_pred = noise_pred.to(dtype=torch.float32)

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    with torch.no_grad():
        audio = model_dict.dac_model.decode(latents)
        audio = audio.float().cpu()

    audio = audio[:, :int(audio_len_in_s * model_dict.dac_model.sample_rate)]
    return audio, model_dict.dac_model.sample_rate


# In utils.py, add this function

def create_node_exit_values(silent_audio, passthrough_video=None, passthrough_images=None, message="Process skipped or failed."):
    """
    Creates a standardized tuple of return values for exiting a node early.
    Handles passthrough of video or image inputs.
    """
    audio_output = {"waveform": torch.zeros((1, 1, 1)), "sample_rate": 48000} if silent_audio else None
    
    # Prioritize passing through the image tensor if it exists
    if passthrough_images is not None:
        frames_output = passthrough_images
    else:
        frames_output = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
        
    # Pass through the video path if it exists
    video_path_output = extract_video_path(passthrough_video) if passthrough_video else ""
    
    return (video_path_output, frames_output, audio_output, message)

def smart_encode_video_with_siglip2(clip_frames, model_dict, safety_margin_gb=1.5):
    device = model_dict.device
    total_frames = clip_frames.shape[1]
    
    # 1. Get Initial Memory State
    # We must unload everything else first to get a clean measurement.
    mm.unload_all_models()
    torch.cuda.empty_cache()
    
    # Move the essential model to the GPU
    model_dict.siglip2_model.to(device)
    
    # Measure VRAM *after* the model is loaded.
    torch.cuda.synchronize() # Wait for move to complete
    vram_after_model_load = torch.cuda.memory_reserved(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    available_vram = total_vram - vram_after_model_load
    safe_vram_budget = available_vram - (safety_margin_gb * 1024**3)

    # 2. Estimate Memory Per Frame
    # To do this, we process ONE frame to see how much VRAM it uses.
    # This is our "profiling" step.
    try:
        # Measure memory before the forward pass
        vram_before_profiling_pass = torch.cuda.memory_reserved(device)
        
        # Process a single frame
        _ = model_dict.siglip2_model.get_image_features(pixel_values=clip_frames[:, 0, ...])
        
        torch.cuda.synchronize() # Wait for computation to complete
        vram_after_profiling_pass = torch.cuda.memory_reserved(device)
        
        # The memory used is the difference
        memory_per_frame = vram_after_profiling_pass - vram_before_profiling_pass
        
        if memory_per_frame <= 0:
            # Something went wrong or memory was reused, fall back to a safe default.
            memory_per_frame = 200 * 1024**2 # Guess: 200MB per frame
            
    except Exception as e:
        # If even one frame OOMs, we can't proceed.
        logger.error("OOM during profiling pass. Cannot determine memory per frame.")
        raise e

    # 3. Calculate the Optimal Batch Size
    if safe_vram_budget > 0 and memory_per_frame > 0:
        # The number of frames we can fit is our budget divided by the cost per frame.
        optimal_batch_size = int(safe_vram_budget // memory_per_frame)
        # Clamp the value to be at least 1 and at most the total number of frames.
        optimal_batch_size = max(1, min(optimal_batch_size, total_frames))
    else:
        # If something went wrong, fall back to a very safe default.
        optimal_batch_size = 1

    logger.info(f"Smart Batcher: Available VRAM = {available_vram/1024**3:.2f}GB")
    logger.info(f"Smart Batcher: Memory per frame = {memory_per_frame/1024**2:.2f}MB")
    logger.info(f"Smart Batcher: Calculated optimal batch size = {optimal_batch_size}")
    
    # 4. Run the Standard Batched Inference with the Calculated Batch Size
    # Now we call the original function, but with our dynamically calculated batch size.
    return encode_video_with_siglip2(clip_frames, model_dict, batch_size=optimal_batch_size)    

def _batched_frames_to_tensor(pil_list, preprocess_func, batch_size, device):
    """ Preprocesses and moves frames to device in small batches to save VRAM. """
    tensor_chunks = []
    for i in range(0, len(pil_list), batch_size):
        batch_pil = pil_list[i : i + batch_size]
        batch_preprocessed = [preprocess_func(im) for im in batch_pil]
        batch_tensor = torch.stack(batch_preprocessed, dim=0).to(device)
        tensor_chunks.append(batch_tensor)
        # Immediately clear the list to save RAM
        del batch_pil, batch_preprocessed, batch_tensor
    
    # Once all chunks are on the device, concatenate them.
    # This is much more memory-efficient than creating one giant tensor at once.
    full_tensor = torch.cat(tensor_chunks, dim=0).unsqueeze(0)
    del tensor_chunks
    return full_tensor    