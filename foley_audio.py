# foley_audio.py
import os
import sys
import tempfile
from typing import List, Tuple

import torch
import numpy as np

# --- Make sure we can import the repo's python package ------------------------
HERE = os.path.dirname(__file__)
HVF_REPO_DIR = os.path.join(HERE, "HunyuanVideo-Foley")  # contains 'hunyuanvideo_foley' package
if HVF_REPO_DIR not in sys.path:
    sys.path.append(HVF_REPO_DIR)

# HunyuanVideo-Foley internals
from hunyuanvideo_foley.utils.model_utils import load_model, denoise_process  # type: ignore
from hunyuanvideo_foley.utils.feature_utils import feature_process  # type: ignore
from PIL import Image
from hunyuanvideo_foley.utils.config_utils import AttributeDict  # type: ignore
from hunyuanvideo_foley.utils.feature_utils import (
    encode_text_feat,
    encode_video_with_siglip2,
    encode_video_with_sync,
)

try:
    from einops import rearrange
except Exception:
    rearrange = None  # we'll fallback to view/permute if needed


def _encode_sync_feats_local(sync_frames_btchw: torch.Tensor, model_dict, segment_size: int = 16, step_size: int = 8):
    """
    Re-implementation of encode_video_with_sync WITHOUT autocast-to-fp16.
    sync_frames_btchw: [B, T, 3, 224, 224] float32 on model device
    returns: [B, S*Tseg(=8), D]
    """
    model = model_dict.syncformer_model
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype  # typically float32

    x = sync_frames_btchw.to(device=dev, dtype=dtype)
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    num_segments = (t - segment_size) // step_size + 1
    segments = [x[:, i*step_size:i*step_size+segment_size] for i in range(num_segments)]
    x = torch.stack(segments, dim=1)  # [B, S, Tseg, 3, 224, 224]

    if rearrange:
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
    else:
        b_, s_, t_, c_, h_, w_ = x.shape
        x = x.reshape(b_*s_, 1, t_, c_, h_, w_)

    outs = []
    bs = b * num_segments
    # (batching optional; typical S small anyway)
    for i in range(0, bs, bs):
        outs.append(model(x[i:i+bs]))
    x = torch.cat(outs, dim=0)  # [B*S, 1, 8, D]

    if rearrange:
        x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
    else:
        bsm, _, tseg, d = x.shape
        s = bsm // b
        x = x.view(b, s, 1, tseg, d).permute(0, 1, 3, 4, 2).reshape(b, s*tseg, d)
    return x


def _feature_process_from_images(
    images_uint8: List[np.ndarray],
    prompt: str,
    model_dict,
    cfg,
    fps_hint: float,
):
    """
    Build (visual_feats, text_feats, audio_len_in_s) directly from in-memory frames.
    This version is VRAM-friendly: downsamples long clips and batches SigLIP2/Syncformer.
    """
    import math
    from PIL import Image
    from hunyuanvideo_foley.utils.config_utils import AttributeDict

    dev = model_dict.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- knobs tuned for 8–10GB VRAM ---
    MAX_FRAMES = 64          # cap total frames we process
    SIGLIP_BATCH = 4         # per-forward batch for SigLIP2
    SYNC_BATCH = 1           # per-forward batch for Synchformer

    # Downsample the input frame list if too long
    frames = images_uint8
    if len(frames) > MAX_FRAMES:
        step = math.ceil(len(frames) / MAX_FRAMES)
        frames = frames[::step]

    # Reasonable audio length from (downsampled) frames
    fps = max(float(fps_hint or 8.0), 1.0)
    audio_len_in_s = max(1.0, len(frames) / fps)

    # ---- SigLIP2 visual feats (expects 512x512 normalize) ----
    pil_list = [Image.fromarray(f).convert("RGB") for f in frames]
    siglip_list = [model_dict.siglip2_preprocess(im) for im in pil_list]  # [C,512,512] float
    clip_frames = torch.stack(siglip_list, dim=0).unsqueeze(0).to(dev)     # [1,T,3,512,512]

    # ---- Syncformer feats (224 pipeline) ----
    sync_list = [model_dict.syncformer_preprocess(im) for im in pil_list]  # [3,224,224] float32
    sync_frames = torch.stack(sync_list, dim=1).unsqueeze(0).to(dev)       # [1,T,3,224,224]

    # Mixed precision only if CUDA
    use_amp = (hasattr(dev, "type") and dev.type == "cuda")

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
        siglip2_feat = encode_video_with_siglip2(
            clip_frames, model_dict, batch_size=SIGLIP_BATCH
        ).to(dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp, dtype=torch.float16):
        syncformer_feat = encode_video_with_sync(
            sync_frames, model_dict, batch_size=SYNC_BATCH
        ).to(dev)

    visual_feats = AttributeDict({
        "siglip2_feat": siglip2_feat,
        "syncformer_feat": syncformer_feat,
    })

    # ---- Text feats (CLAP), identical to repo’s feature_process() flow ----
    neg_prompt = "noisy, harsh"
    prompts = [neg_prompt, prompt]
    text_feat_res, _ = encode_text_feat(prompts, model_dict)

    text_feat = text_feat_res[1:]
    uncond_text_feat = text_feat_res[:1]
    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = AttributeDict({
        "text_feat": text_feat,
        "uncond_text_feat": uncond_text_feat,
    })

    return visual_feats, text_feats, audio_len_in_s


# ------------------------------------------------------------------------------
# Small cache so we don't reload weights every node execution
# ------------------------------------------------------------------------------
class _FoleyState:
    model_key: Tuple[str, str, str, int] | None = None
    model_dict = None
    cfg = None
    device = None


_STATE = _FoleyState()


def _ensure_model_loaded(model_path_dir: str, config_path: str, device: str = "auto", gpu_id: int = 0):
    """
    Load (or reuse) the HunyuanVideo-Foley model bundle on a concrete device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            torch_device = torch.device(f"cuda:{gpu_id}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_device = torch.device("mps")
        else:
            torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device)

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

    assert images.ndim == 4, f"Expected 4D tensor [N,H,W,C], got shape {tuple(images.shape)}"
    imgs = (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()  # [N,H,W,C]
    out = [imgs[i, :, :, :3] for i in range(imgs.shape[0])]  # drop alpha if present
    return out


def _write_temp_mp4(frames: List[np.ndarray], fps: float) -> str:
    """
    Writes frames to a temporary .mp4 and returns the file path.
    Uses libx264 via imageio-ffmpeg. Dimensions need not be multiples of 2;
    imageio handles padding.
    """
    os.makedirs(os.path.join(tempfile.gettempdir(), "foley_tmp"), exist_ok=True)
    tmp_path = os.path.join(tempfile.gettempdir(), "foley_tmp", next(tempfile._get_candidate_names()) + ".mp4")
    # imageio expects RGB uint8 arrays
    imageio.mimsave(tmp_path, frames, fps=fps, codec="libx264", quality=8)
    return tmp_path


# ------------------------------------------------------------------------------
# ComfyUI node
# ------------------------------------------------------------------------------
class HunyuanFoleyAudio:
    """
    Generate Foley audio from a sequence of frames + an audio prompt.
    Returns an AUDIO object compatible with VHS Video Combine's 'audio' input:
      {'waveform': torch.float32 [1, C, S], 'sample_rate': int}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ComfyUI IMAGE (tensor [N,H,W,C] float in 0..1)
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "audio_prompt": ("STRING", {
                    "multiline": True,
                    "default": "suspenseful piano with rising tension; two-hit sting (ta-dum)"
                }),
                # Paths rooted at this custom node directory by default
                "model_path_dir": ("STRING", { 
                    "default": os.path.join(HVF_REPO_DIR, "HunyuanVideo-Foley")
                }),
                "config_path": ("STRING", {
                    "default": os.path.join(HVF_REPO_DIR, "configs", "hunyuanvideo-foley-xxl.yaml")
                }),
                "num_inference_steps": ("INT", {"default": 10, "min": 10, "max": 200, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                # Keep for future expansion; load_model already picks the right device internally.
                # You can expose 'device'/'gpu_id' here if you prefer explicit control.
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Hunyuan Foley"

    def generate(
        self,
        images: torch.Tensor,
        frame_rate: float,
        audio_prompt: str,
        model_path_dir: str,
        config_path: str,
        num_inference_steps: int,
        guidance_scale: float,
    ):
        # 1) Convert frames to uint8 RGB list
        frames = _images_tensor_to_uint8_list(images)
        if len(frames) == 0:
            # Return a silent audio stub if nothing to do
            silent = torch.zeros(1, 1, 1, dtype=torch.float32)
            return ({"waveform": silent, "sample_rate": 44100},)

        try:
            # 3) Load / reuse model
            model_dict, cfg = _ensure_model_loaded(model_path_dir, config_path)

            # Build features directly from in-memory frames (no disk I/O)
            visual_feats, text_feats, audio_len_in_s = _feature_process_from_images(
                images_uint8=frames,        # was frames_uint8
                prompt=audio_prompt,
                model_dict=model_dict,
                cfg=cfg,
                fps_hint=float(frame_rate), # was fps
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

            # denoise_process returns batch [B,C,S]; we want [1,C,S]
            if isinstance(audio_batch, torch.Tensor):
                # pick the first in batch
                waveform = audio_batch[0].to(torch.float32)  # [C,S]
                if waveform.ndim == 1:  # [S] -> mono
                    waveform = waveform.unsqueeze(0)  # [1,S]
                waveform = waveform.unsqueeze(0)  # [1,C,S]
            else:
                raise RuntimeError("Unexpected audio tensor type from denoise_process.")

            return ({"waveform": waveform, "sample_rate": int(sample_rate)},)

        finally:
            # Clean up temp file
            try:
                if os.path.exists(tmp_video):
                    os.remove(tmp_video)
            except Exception:
                pass


# ------------------------------------------------------------------------------
# ComfyUI registration
# ------------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "HunyuanFoleyAudio": HunyuanFoleyAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanFoleyAudio": "Hunyuan Foley (Audio)",
}
