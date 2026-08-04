"""OpenAI backend for Vgent — added so the framework can run without a local LVLM.

Vgent ships wrappers for Qwen2.5-VL / InternVL2.5 / LLaVA-Video / LongVU and
dispatches on `MODEL_MAP` in utils/vgent.py. This module implements the same
three-function contract, so every stage of the paper's pipeline (entity
extraction, keyword extraction, structured verification, aggregation, final
multimodal generation) runs against the OpenAI API instead:

    load_model(path)  -> (processor, video_llm, image_processor, _)
    load_video(p,args)-> (raw_video, _, _, frame_idx, fps, video_inputs, size_list)
    mllm_response(video_llm, processor, image_processor, text, image_inputs,
                  video, max_new_tokens, size_list=None, fps=None) -> str

Two deliberate choices:

* **BGE stays local.** Only the *LVLM* roles are swapped. The paper's thresholds
  — entity merging at 0.7, keyword match at 0.5 — are calibrated against
  `BAAI/bge-large-en-v1.5` cosine similarities, and OpenAI embeddings have a
  different similarity scale, so substituting them would silently invalidate
  both constants. BGE is ~1.3 GB and runs fine on CPU.

* **Frames, not video.** `video_inputs[0]` must stay a TCHW tensor because
  `construct_graph`/`refine_nodes` call `torch.split(..., chunk_size)` on it.
  We keep that contract and convert a chunk to JPEG frames only at the moment of
  the API call, uniformly subsampling to `VGENT_MAX_FRAMES` (default 16) — a
  64-frame chunk sent whole would be 64 images per request, and detail=low costs
  103 tokens per image on gpt-4o, so the cap is the difference between a $2 run
  and a $9 one.
"""
import base64
import io
import os
import time

import numpy as np
import torch
from PIL import Image

MAX_FRAMES = int(os.environ.get("VGENT_MAX_FRAMES", 16))
VISION_MODEL = os.environ.get("VGENT_VISION_MODEL", "gpt-4o")
TEXT_MODEL = os.environ.get("VGENT_TEXT_MODEL", "gpt-4o-mini")
DETAIL = os.environ.get("VGENT_IMAGE_DETAIL", "low")


class OpenAIVLM:
    """Holds the client and accumulates usage so the bench can bill each stage."""

    def __init__(self, vision_model=VISION_MODEL, text_model=TEXT_MODEL):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY") or open(
            os.path.expanduser("~/.config/openai/api_key")).read().strip()
        self.client = OpenAI(api_key=key)
        self.vision_model, self.text_model = vision_model, text_model
        self.usage = {"vision_calls": 0, "text_calls": 0, "frames_sent": 0,
                      "in_tok": 0, "out_tok": 0, "by_model": {}}

    def _account(self, model, usage, n_frames=0):
        self.usage["in_tok"] += usage.prompt_tokens
        self.usage["out_tok"] += usage.completion_tokens
        self.usage["frames_sent"] += n_frames
        b = self.usage["by_model"].setdefault(model, {"calls": 0, "in": 0, "out": 0})
        b["calls"] += 1
        b["in"] += usage.prompt_tokens
        b["out"] += usage.completion_tokens

    def chat(self, content, model, max_tokens, n_frames=0, retries=4):
        last = None
        for attempt in range(retries):
            try:
                r = self.client.chat.completions.create(
                    model=model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}])
                self._account(model, r.usage, n_frames)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:                    # noqa: BLE001
                last = e
                # Vgent's own call sites retry on parse failure, but a rate limit
                # would burn those attempts; back off here instead.
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OpenAI call failed after {retries}: {last}")


def _to_b64(frame, max_px=512):
    """One TCHW/HWC frame -> base64 JPEG."""
    if isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):      # CHW -> HWC
            arr = arr.permute(1, 2, 0)
        arr = arr.numpy()
    else:
        arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _sample(video, k=MAX_FRAMES):
    """Uniformly subsample a chunk down to k frames, preserving order."""
    n = len(video)
    if n <= k:
        return [video[i] for i in range(n)]
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [video[int(i)] for i in idx]


# ------------------------------------------------------------------ the contract

def load_model(model_name=VISION_MODEL):
    # Positions match the local wrappers: Vgent unpacks this as
    # (self.processor, self.video_llm, self.image_processor, _).
    return None, OpenAIVLM(vision_model=model_name), None, None


def load_video(video_path, args):
    """Sample at args.fps into a uint8 TCHW tensor.

    uint8 rather than float: these frames are only ever JPEG-encoded for the
    API, so the float conversion the local wrappers need would quadruple memory
    for nothing.
    """
    from decord import VideoReader, cpu

    # VGENT_DECODE_MAX_PX caps the decoded long side. Only needed for the
    # 95-minute concatenated corpus video: 5,700 frames at 1280x720 is 15.8 GB,
    # and _to_b64 thumbnails every frame to 512 before it reaches the API, so
    # decoding larger than that is pure memory for no information.
    cap = int(os.environ.get("VGENT_DECODE_MAX_PX", "0") or 0)
    kw = {}
    if cap:
        vr0 = VideoReader(video_path, ctx=cpu(), num_threads=1)
        h, w = vr0[0].shape[:2]
        del vr0
        if max(h, w) > cap:
            sc = cap / max(h, w)
            kw = {"width": int(round(w * sc)), "height": int(round(h * sc))}
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1, **kw)
    native_fps = vr.get_avg_fps() or 30.0
    target_fps = float(getattr(args, "fps", 1.0) or 1.0)
    n = max(1, int(round(len(vr) / native_fps * target_fps)))
    idx = np.linspace(0, len(vr) - 1, n).round().astype(int).tolist()
    frames = np.stack([vr[i].asnumpy() for i in idx])        # THWC uint8
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)     # TCHW
    return [video], None, None, idx, target_fps, [video], None


def mllm_response(video_llm, tokenizer, processor, text, image_inputs, video,
                  max_new_tokens=512, size_list=None, fps=None):
    """Text-only when `video` is None; otherwise frames + text to the vision model."""
    client: OpenAIVLM = video_llm

    if video is None and not image_inputs:
        return client.chat([{"type": "text", "text": text}],
                           client.text_model, max_new_tokens)

    frames = []
    if video is not None:
        seq = video[0] if (isinstance(video, (list, tuple)) and len(video)
                           and hasattr(video[0], "__len__")) else video
        frames = _sample(seq)
    if image_inputs:
        frames = list(image_inputs) + frames

    content = [{"type": "text", "text": text}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{_to_b64(f)}", "detail": DETAIL}})
    return client.chat(content, client.vision_model, max_new_tokens,
                       n_frames=len(frames))
