import torch
import numpy as np
from PIL import Image
from einops import rearrange
from typing import List

from hunyuanvideo_foley.utils.feature_utils import encode_text_feat, encode_video_with_siglip2, encode_video_with_sync


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

    # ---- SigLIP2 path (expects 512x512 normalized) ----
    siglip_list = [model_dict.siglip2_preprocess(im) for im in pil_list]  # [C,512,512], float
    clip_frames = torch.stack(siglip_list, dim=0).unsqueeze(0).to(dev)     # [1,T,3,512,512]
    siglip2_feat = encode_video_with_siglip2(
        clip_frames, model_dict, batch_size=siglip2_batch
    ).to(dev)

    # --- Syncformer visual features (expects [B, T, 3, 224, 224]) ---
    sync_list = []
    for fr in frames:  # <-- was frames_sync (undefined). Use the downsampled `frames`.
        im = Image.fromarray(fr).convert("RGB")
        x = model_dict.syncformer_preprocess(im)  # -> torch.FloatTensor [3,224,224]
        sync_list.append(x)

    # stack along time axis (T), then add batch dim -> [1, T, 3, 224, 224]
    sync_frames = torch.stack(sync_list, dim=0).unsqueeze(0).to(model_dict.device)

    # encode (no fp16 autocast here; Syncformer runs in float32)
    syncformer_feat = encode_video_with_sync_v2(
        sync_frames, model_dict, batch_size=syncformer_batch  # <-- use your param
    ).to(model_dict.device)

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
    text_feat_res, _ = encode_text_feat(prompts, model_dict)

    text_feat = text_feat_res[1:]
    uncond_text_feat = text_feat_res[:1]

    if cfg.model_config.model_kwargs.text_length < text_feat.shape[1]:
        text_seq_length = cfg.model_config.model_kwargs.text_length
        text_feat = text_feat[:, :text_seq_length]
        uncond_text_feat = uncond_text_feat[:, :text_seq_length]

    text_feats = {
        "text_feat": text_feat,
        "uncond_text_feat": uncond_text_feat,
    }

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
