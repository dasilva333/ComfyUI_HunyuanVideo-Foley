# foley_audio.py
import os
import sys
from typing import List, Tuple
import math
import torch
import numpy as np

# --- repo root so we can import the package -------------------------------
HERE = os.path.dirname(__file__)
HVF_ROOT = os.path.join(HERE, "HunyuanVideo-Foley")  # contains: hunyuanvideo_foley/, configs/, HunyuanVideo-Foley/
if HVF_ROOT not in sys.path:
    sys.path.append(HVF_ROOT)

# HunyuanVideo-Foley internals
from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process  # type: ignore
# Import the functions from utils.py located in the same directory as foley_audio.py
from .utils import feature_process_from_images  # type: ignore

# --------------------------------------------------------------------------
# Small cache so we don't reload weights every node execution
# --------------------------------------------------------------------------
class _FoleyState:
    model_key: Tuple[str, str, str, int] | None = None
    model_dict = None
    cfg = None
    device = None


_STATE = _FoleyState()


def _select_device(device: str = "cuda", gpu_id: int = 0) -> torch.device:
    device = (device or "cuda").lower()
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(gpu_id)}")
        # fall back but keep type consistent
        return torch.device("cpu")
    if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    # "auto" or unknown -> prefer CUDA if present
    return torch.device(f"cuda:{int(gpu_id)}") if torch.cuda.is_available() else torch.device("cpu")


def _ensure_model_loaded(model_path_dir: str, config_path: str, device: str = "cuda", gpu_id: int = 0):
    """
    Load (or reuse) the HunyuanVideo-Foley model bundle on a concrete device.
    """
    torch_device = _select_device(device, gpu_id)

    global _STATE
    key = (os.path.abspath(model_path_dir), os.path.abspath(config_path), str(torch_device), int(gpu_id))
    if _STATE.model_key == key and _STATE.model_dict is not None:
        return _STATE.model_dict, _STATE.cfg

    model_dict, cfg = load_model(model_path_dir, config_path, torch_device)
    _STATE.model_key = key
    _STATE.model_dict = model_dict
    _STATE.cfg = cfg
    _STATE.device = torch_device
    return model_dict, cfg


def _images_tensor_to_uint8_list(images: torch.Tensor) -> List[np.ndarray]:
    """
    Converts a ComfyUI IMAGE tensor to a list of HxWx3 uint8 RGB frames.

    Expect shapes:
      - [N, H, W, C] float32 (0..1)  OR
      - [H, W, C] float32 (0..1)    (single frame)
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)  # [1, H, W, C]
    if images.ndim != 4:
        raise ValueError(f"Expected 4D tensor [N,H,W,C], got {tuple(images.shape)}")

    imgs = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()  # [N,H,W,C]
    out = [imgs[i, :, :, :3] for i in range(imgs.shape[0])]  # drop alpha if present
    return out


class HunyuanFoleyAudio:
    """
    Generate Foley audio from a sequence of frames + an audio prompt.

    Returns an AUDIO object compatible with VHS Video Combine's 'audio' input:
      {'waveform': torch.float32 [1, C, S], 'sample_rate': int}
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_dir_default = os.path.join(HVF_ROOT, "HunyuanVideo-Foley")
        config_path_default = os.path.join(HVF_ROOT, "configs", "hunyuanvideo-foley-xxl.yaml")

        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "audio_prompt": ("STRING", {
                    "multiline": True,
                    "default": "suspenseful piano with rising tension; two-hit sting (ta-dum)"
                }),
                "model_path_dir": ("STRING", {"default": model_dir_default}),
                "config_path": ("STRING", {"default": config_path_default}),
                "num_inference_steps": ("INT", {"default": 10, "min": 1, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                # Give manual control so you can keep it on GPU (default) or swap if needed.
                "device": (["cuda", "cpu", "mps", "auto"], {"default": "cuda"}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan Foley"

    # --- mirror infer() -> denoise like the reference CLI -------------------
    def _infer_from_images(
        self,
        images_uint8: List[np.ndarray],
        prompt: str,
        model_dict,
        cfg,
        guidance_scale: float,
        num_inference_steps: int,
        fps_hint: float,
    ):
        visual_feats, text_feats, audio_len_in_s = feature_process_from_images(
            images_uint8=images_uint8,
            prompt=prompt,
            model_dict=model_dict,
            cfg=cfg,
            fps_hint=fps_hint,
        )

        audio_batch, sample_rate = denoise_process(
            visual_feats,
            text_feats,
            audio_len_in_s,
            model_dict,
            cfg,
            guidance_scale=guidance_scale,
            num_inference_steps=int(num_inference_steps),
        )
        # match CLI return: first item
        return audio_batch[0], sample_rate

    def generate(
        self,
        images: torch.Tensor,
        frame_rate: float,
        audio_prompt: str,
        model_path_dir: str,
        config_path: str,
        num_inference_steps: int,
        guidance_scale: float,
        device: str = "cuda",
        gpu_id: int = 0,
    ):
        frames = _images_tensor_to_uint8_list(images)
        
        # VRAM frame size cap
        MAX_FRAMES = 64  # Limit the number of frames to 64

        # If the number of frames exceeds MAX_FRAMES, downsample the frames
        if len(frames) > MAX_FRAMES:
            step = math.ceil(len(frames) / MAX_FRAMES)
            frames = frames[::step]

        if len(frames) == 0:
            silent = torch.zeros(1, 1, 1, dtype=torch.float32)
            return ({"waveform": silent, "sample_rate": 44100},)

        # 1) Load / reuse model on requested device
        model_dict, cfg = _ensure_model_loaded(model_path_dir, config_path, device=device, gpu_id=gpu_id)

        # 2) Infer (mirrors the original infer() flow, but from in-mem frames)
        audio_tensor, sample_rate = self._infer_from_images(
            frames, audio_prompt, model_dict, cfg, guidance_scale, num_inference_steps, fps_hint=float(frame_rate)
        )

        # 3) Shape to VHS expected format: [1, C, S] float32
        waveform = audio_tensor.to(torch.float32)
        if waveform.ndim == 1:  # [S] -> [1,S]
            waveform = waveform.unsqueeze(0)
        # audio is [C,S]; wrap batch dim
        waveform = waveform.unsqueeze(0)  # [1, C, S]

        return ({"waveform": waveform, "sample_rate": int(sample_rate)},)


# --------------------------------------------------------------------------
# ComfyUI registration
# --------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyAudio": HunyuanFoleyAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanFoleyAudio": "Hunyuan Foley (Audio)",
}
