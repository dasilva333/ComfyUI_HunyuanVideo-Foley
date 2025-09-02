# utils.py
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from typing import List

from hunyuanvideo_foley.utils.feature_utils import encode_text_feat, encode_video_with_siglip2, encode_video_with_sync
from hunyuanvideo_foley.utils.config_utils import AttributeDict

@torch.inference_mode()
def encode_video_features_via_images(
    frames_uint8: List[np.ndarray],
    model_dict,
    fps_hint: float,
    *,
    max_frames: int = 64,
    siglip2_batch: int = 4,
    syncformer_batch: int = 1,
):
    """
    Build visual features (SigLIP2 + Synchformer) directly from in-memory frames.

    frames_uint8: list of HxWx3 uint8 RGB arrays
    fps_hint:     fps that the provided frame sequence represents
    """
    dev = model_dict.device

    # Downsample long clips to keep VRAM under control.
    frames = frames_uint8
    if len(frames) > max_frames:
        step = np.ceil(len(frames) / max_frames)
        frames = frames[::int(step)]

    # PIL conversion once
    pil_list = [Image.fromarray(f).convert("RGB") for f in frames]

    # utils.py, in encode_video_features_via_images
    # ---- SigLIP2 path (with offloading) ----
    siglip2_model = model_dict.siglip2_model
    try:
        print("Hunyuan Foley: Moving siglip2_model to GPU.")
        siglip2_model.to(dev)
        siglip_list = [model_dict.siglip2_preprocess(im) for im in pil_list]
        clip_frames = torch.stack(siglip_list, dim=0).unsqueeze(0).to(dev)
        siglip2_feat = encode_video_with_siglip2(clip_frames, model_dict, batch_size=siglip2_batch).to(dev)
    finally:
        print("Hunyuan Foley: Moving siglip2_model back to CPU.")
        siglip2_model.to("cpu")

    # utils.py, in encode_video_features_via_images
    # --- Syncformer path (with offloading) ---
    syncformer_model = model_dict.syncformer_model
    try:
        print("Hunyuan Foley: Moving syncformer_model to GPU.")
        syncformer_model.to(dev)
        sync_list = []
        for fr in frames:
            im = Image.fromarray(fr).convert("RGB")
            x = model_dict.syncformer_preprocess(im)
            sync_list.append(x)
        sync_frames = torch.stack(sync_list, dim=0).unsqueeze(0).to(dev)
        syncformer_feat = encode_video_with_sync_v2(sync_frames, model_dict, batch_size=syncformer_batch).to(dev)
    finally:
        print("Hunyuan Foley: Moving syncformer_model back to CPU.")
        syncformer_model.to("cpu")

    vid_len_in_s = max(1.0, len(frames) / max(float(fps_hint), 1.0))

    visual_features = {
        "siglip2_feat": siglip2_feat,
        "syncformer_feat": syncformer_feat,
    }
    return visual_features, vid_len_in_s


def feature_process_from_images(
    images_uint8: List[np.ndarray],
    prompt: str,
    model_dict,
    cfg,
    fps_hint: float,
):
    """
    Mirror of feature_process(video_path, ...) but sources frames from memory.
    Returns (visual_feats, text_feats, audio_len_in_s)
    """
    # VRAM frame size cap
    MAX_FRAMES = 64  # Limit the number of frames to 64

    # Downsample the input frame list if too long
    frames = images_uint8
    if len(frames) > MAX_FRAMES:
        step = np.ceil(len(frames) / MAX_FRAMES)
        frames = frames[::int(step)]

    visual_feats, audio_len_in_s = encode_video_features_via_images(
        frames_uint8=frames,  # Use downsampled frames
        model_dict=model_dict,
        fps_hint=fps_hint,
        max_frames=MAX_FRAMES,  # Keep it consistent with the limit
        siglip2_batch=4,
        syncformer_batch=1,
    )

    neg_prompt = "noisy, harsh"
    prompts = [neg_prompt, prompt]
    clap_model = model_dict.clap_model
    try:
        clap_model.to(model_dict.device)
        text_feat_res, _ = encode_text_feat(prompts, model_dict)
    finally:
        clap_model.to("cpu") # Ensure it's offloaded

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

    # No need for AttributeDict here anymore, we are using plain dictionaries
    visual_feats = AttributeDict({
        "siglip2_feat": visual_feats["siglip2_feat"],
        "syncformer_feat": visual_feats["syncformer_feat"]
    })

    return visual_feats, text_feats, audio_len_in_s

@torch.inference_mode()
def encode_video_with_sync_v2(x: torch.Tensor, model_dict, batch_size: int = -1):
    """
    Renamed version of `encode_video_with_sync` for unique identification.
    x: [B, T, C, H, W] with C=3, H=W=224
    """
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    segment_size = 16
    step_size   = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = [x[:, i*step_size : i*step_size + segment_size] for i in range(num_segments)]
    x = torch.stack(segments, dim=1)  # (B, S, T, 3, 224, 224)

    # utils.py, in encode_video_with_sync_v2
    # Align to model device & dtype (typically float32)
    model = model_dict.syncformer_model
    target_param = next(model.parameters())
    x = x.to(device=target_param.device, dtype=target_param.dtype)

    if batch_size < 0:
        batch_size = b * num_segments

    x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")

    outputs = []
    for i in range(0, b * num_segments, batch_size):
        # keep types consistent; no autocast to half here
        outputs.append(model(x[i : i + batch_size]))

    x = torch.cat(outputs, dim=0)              # [B*S, 1, 8, 768]
    x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
    return x

# =================================================================================
# FORKED MODEL LOADING LOGIC
# Forked from hunyuanvideo_foley.utils.model_utils to apply memory-safe loading
# =================================================================================
import os
from loguru import logger
from torchvision import transforms
from torchvision.transforms import v2
from transformers import AutoTokenizer, AutoModel, ClapTextModelWithProjection
from hunyuanvideo_foley.models.dac_vae.model.dac import DAC
from hunyuanvideo_foley.models.synchformer import Synchformer
from hunyuanvideo_foley.models.hifi_foley import HunyuanVideoFoley
from hunyuanvideo_foley.utils.config_utils import load_yaml, AttributeDict
from hunyuanvideo_foley.utils.model_utils import load_state_dict

def load_model(model_path, config_path, device):
    logger.info("Starting model loading process (custom memory-safe version)...")
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Model weights dir: {model_path}")
    logger.info(f"Target device: {device}")
    
    cfg = load_yaml(config_path)
    logger.info("Configuration loaded successfully")
    
    model_dict = AttributeDict({})

    # --- MEMORY-SAFE SEQUENTIAL LOADING (Corrected Paths) ---

    # 1. HunyuanVideoFoley
    logger.info("Loading HunyuanVideoFoley main model (CPU -> GPU)...")
    # Initialize on CPU
    foley_model = HunyuanVideoFoley(cfg, dtype=torch.bfloat16, device="cpu").to(dtype=torch.bfloat16)
    # Load weights into CPU model
    foley_model = load_state_dict(foley_model, os.path.join(model_path, "hunyuanvideo_foley.pth"))
    # Move fully loaded model to GPU
    foley_model.to(device).eval()
    if device.type == 'cuda': torch.cuda.empty_cache()
    logger.info("HunyuanVideoFoley model loaded.")

    # 2. DAC-VAE
    logger.info(f"Loading DAC VAE model (CPU -> GPU)...")
    dac_path = os.path.join(model_path, "vae_128d_48k.pth")
    dac_model = DAC.load(dac_path, map_location="cpu")
    dac_model.to(device).eval()
    dac_model.requires_grad_(False)
    if device.type == 'cuda': torch.cuda.empty_cache()
    logger.info("DAC VAE model loaded.")

    # 3. Siglip2 visual-encoder
    logger.info("Loading SigLIP2 visual encoder (CPU -> GPU)...")
    siglip2_preprocess = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
    siglip2_model = AutoModel.from_pretrained("google/siglip2-base-patch16-512").eval().to("cpu")
    siglip2_model.to(device)
    if device.type == 'cuda': torch.cuda.empty_cache()
    logger.info("SigLIP2 model loaded.")

    # 4. clap text-encoder
    logger.info("Loading CLAP text encoder (CPU -> GPU)...")
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/larger_clap_general")
    clap_model = ClapTextModelWithProjection.from_pretrained("laion/larger_clap_general").to("cpu")
    clap_model.to(device)
    if device.type == 'cuda': torch.cuda.empty_cache()
    logger.info("CLAP model loaded.")

    # 5. syncformer
    logger.info(f"Loading Synchformer model (CPU -> GPU)...")
    syncformer_path = os.path.join(model_path, "synchformer_state_dict.pth")
    syncformer_preprocess = v2.Compose(
        [
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    syncformer_model = Synchformer().to("cpu")
    syncformer_model.load_state_dict(torch.load(syncformer_path, map_location="cpu"))
    syncformer_model.to(device).eval()
    if device.type == 'cuda': torch.cuda.empty_cache()
    logger.info("Synchformer model loaded.")

    # --- END OF SEQUENTIAL LOADING ---

    logger.info("Creating model dictionary with attribute access...")
    model_dict.foley_model = foley_model
    model_dict.dac_model = dac_model
    model_dict.siglip2_preprocess = siglip2_preprocess
    model_dict.siglip2_model = siglip2_model
    model_dict.clap_tokenizer = clap_tokenizer
    model_dict.clap_model = clap_model
    model_dict.syncformer_preprocess = syncformer_preprocess
    model_dict.syncformer_model = syncformer_model
    model_dict.device = device
    
    logger.info("All models loaded successfully!")
    logger.info("Available model components:")
    for key in model_dict.keys():
        logger.info(f"  - {key}")

    return model_dict, cfg