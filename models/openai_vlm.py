"""OpenAI backend for Vgent, drop-in for models/{qwenvl,llavavideo,internvl,longvu}.py.

Implements exactly the three-function contract Vgent's `MODEL_MAP` dispatch
expects, so `utils/vgent.py`, `utils/retrieval.py`, `utils/prompts.py` and
`vgent_rag.py` run byte-for-byte unchanged:

    load_model(path)   -> (processor, video_llm, image_processor, _)
    load_video(p,args) -> (raw_video, _, _, frame_idx, fps, video_inputs, size_list)
    mllm_response(video_llm, tokenizer, processor, text, image_inputs, video,
                  max_new_tokens, size_list=None, fps=None) -> str

Frame sampling is a transcription of `models/utils.fetch_video(resize=False)`
via the same `smart_nframes` helper, so the frame *indices* are identical to
what Qwen2.5-VL would have received at the same `--fps`. The one difference is
that decoding is capped at `VGENT_DECODE_MAX_PX` on the long side: every frame
is JPEG-thumbnailed to `VGENT_IMAGE_PX` before it reaches the API, so decoding
a 1080p hour-long video at full resolution would cost ~20 GB of RAM to throw
the pixels away immediately.

Environment:
    VGENT_VISION_MODEL   default gpt-4o-mini   -- every call that carries frames
    VGENT_TEXT_MODEL     default gpt-4o-mini   -- text-only calls
    VGENT_MAX_FRAMES     default 0 = NO CAP    -- frames forwarded per API call
    VGENT_IMAGE_DETAIL   default low           -- OpenAI image `detail`
    VGENT_IMAGE_PX       default 512           -- JPEG long side
    VGENT_DECODE_MAX_PX  default 512           -- decode long side
    VGENT_USAGE_LOG      optional jsonl path   -- one row per API call

`VGENT_MAX_FRAMES=0` is the faithful setting: whatever `construct_graph` /
`refine_nodes` / `aggregate_nodes` slice out is forwarded whole, the way a local
LVLM would receive it. A non-zero value uniformly subsamples and is only there
so the frame budget can be ablated.
"""
import base64
import io
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from models.utils import smart_nframes

VISION_MODEL = os.environ.get("VGENT_VISION_MODEL", "gpt-4o-mini")
TEXT_MODEL = os.environ.get("VGENT_TEXT_MODEL", "gpt-4o-mini")
MAX_FRAMES = int(os.environ.get("VGENT_MAX_FRAMES", "0") or 0)   # 0 = no cap
DETAIL = os.environ.get("VGENT_IMAGE_DETAIL", "low")
IMAGE_PX = int(os.environ.get("VGENT_IMAGE_PX", "512"))
DECODE_MAX_PX = int(os.environ.get("VGENT_DECODE_MAX_PX", "512"))
USAGE_LOG = os.environ.get("VGENT_USAGE_LOG") or None

# Billing/context weight of one image, needed to keep a request inside the
# model's context window. Measured, not guessed -- see probe_cost.py, which
# writes the table this dict is seeded from.
IMAGE_TOKENS = {
    ("gpt-4o", "low"): 85,          ("gpt-4o", "high"): 255,
    ("gpt-4o-mini", "low"): 2833,   ("gpt-4o-mini", "high"): 8500,
    ("gpt-4.1", "low"): 85,         ("gpt-4.1", "high"): 255,
    ("gpt-4.1-mini", "low"): 415,   ("gpt-4.1-mini", "high"): 415,
    ("gpt-4.1-nano", "low"): 630,   ("gpt-4.1-nano", "high"): 630,
    ("gpt-5-mini", "low"): 308,     ("gpt-5-mini", "high"): 308,
    ("gpt-5-nano", "low"): 384,     ("gpt-5-nano", "high"): 384,
    ("gpt-5.4-mini", "low"): 308,   ("gpt-5.4-mini", "high"): 308,
    ("gpt-5.4-nano", "low"): 308,   ("gpt-5.4-nano", "high"): 308,
}
# Input context per model. This is what decides whether the paper's frame budget
# survives: aggregate_nodes sends n_refine*chunk = 320 frames in ONE call, so a
# model needs 320 * tokens_per_image of headroom or the request is truncated.
CONTEXT = {
    "gpt-4o": 128000, "gpt-4o-mini": 128000,
    "gpt-4.1": 1000000, "gpt-4.1-mini": 1000000, "gpt-4.1-nano": 1000000,
    "gpt-5-mini": 272000, "gpt-5-nano": 272000,
    "gpt-5.4-mini": 272000, "gpt-5.4-nano": 272000,
}
CONTEXT_OVERRIDE = int(os.environ.get("VGENT_CONTEXT_LIMIT", "0") or 0)


def image_tokens(model, detail=DETAIL):
    if (model, detail) in IMAGE_TOKENS:
        return IMAGE_TOKENS[(model, detail)]
    # Unknown model: assume the worst so we clamp rather than 400 the request.
    return 2833 if "mini" in model else 85


def context_limit(model):
    return CONTEXT_OVERRIDE or CONTEXT.get(model, 128000)


# Hard provider cap on images per request, independent of the context window:
#   400 - "Too many images in request: 501, maximum allowed: 500."
# This bites even on 1M-context models. It is reached in aggregate_nodes for
# "order" questions, where node2indices uses n_refine=8 rather than args.n_refine
# (retrieval.py:22), i.e. 8 * chunk_size = 512 frames in one call.
MAX_IMAGES_PER_REQUEST = int(os.environ.get("VGENT_MAX_IMAGES_PER_REQUEST", "500"))


def max_images_for(model, detail=DETAIL, reserve=8000):
    """Images that fit in one request: the tighter of context and provider cap."""
    by_ctx = (context_limit(model) - reserve) // image_tokens(model, detail)
    return max(1, min(by_ctx, MAX_IMAGES_PER_REQUEST))


class OpenAIVLM:
    """Client plus per-call usage accounting.

    `stage` is set by the driver before each pipeline step; every logged row
    carries it, so cost can be attributed to construct_graph vs refine vs
    aggregate after the fact.
    """

    def __init__(self, vision_model=VISION_MODEL, text_model=TEXT_MODEL):
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY") or open(
            os.path.expanduser("~/.config/openai/api_key")).read().strip()
        self.client = OpenAI(api_key=key)
        self.vision_model, self.text_model = vision_model, text_model
        self.stage = "?"
        self.video_name = "?"
        self.usage = {"calls": 0, "vision_calls": 0, "frames_sent": 0,
                      "frames_dropped": 0, "in_tok": 0, "out_tok": 0,
                      "by_stage": {}}

    def _account(self, model, usage, n_frames, dropped, elapsed):
        self.usage["calls"] += 1
        self.usage["in_tok"] += usage.prompt_tokens
        self.usage["out_tok"] += usage.completion_tokens
        self.usage["frames_sent"] += n_frames
        self.usage["frames_dropped"] += dropped
        if n_frames:
            self.usage["vision_calls"] += 1
        b = self.usage["by_stage"].setdefault(
            self.stage, {"calls": 0, "in": 0, "out": 0, "frames": 0})
        b["calls"] += 1
        b["in"] += usage.prompt_tokens
        b["out"] += usage.completion_tokens
        b["frames"] += n_frames
        if USAGE_LOG:
            with open(USAGE_LOG, "a") as f:
                f.write(json.dumps({
                    "video": self.video_name, "stage": self.stage,
                    "model": model, "frames": n_frames, "dropped": dropped,
                    "in": usage.prompt_tokens, "out": usage.completion_tokens,
                    "sec": round(elapsed, 2)}) + "\n")

    def chat(self, content, model, max_tokens, n_frames=0, dropped=0, retries=5):
        last = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": content}],
                    **_call_kwargs(model, max_tokens))
                self._account(model, r.usage, n_frames, dropped, time.time() - t0)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:                    # noqa: BLE001
                last = e
                # Vgent's own call sites retry on JSON parse failure; a rate
                # limit would silently burn those attempts, so back off here.
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI call failed after {retries}: {last}")



def _call_kwargs(model, max_tokens):
    """Per-family completion parameters.

    The gpt-5 family are reasoning models and reject the classic arguments:
    `max_tokens` must be `max_completion_tokens`, and `temperature` may only
    take its default. They also spend the budget on hidden reasoning tokens
    before emitting anything, so a cap sized for a short JSON reply returns an
    EMPTY string -- silently, with a valid response object. `reasoning_effort`
    is pinned to "minimal"/"low" and the budget is widened so the visible reply
    survives. Vgent parses every reply as JSON, so an empty string is
    indistinguishable from a model that failed the task.
    """
    if model.startswith("gpt-5"):
        effort = os.environ.get("VGENT_REASONING_EFFORT", "low")
        return {"max_completion_tokens": max(max_tokens, 2048),
                "reasoning_effort": effort}
    return {"temperature": 0, "max_tokens": max_tokens}


def _to_b64(frame, max_px=IMAGE_PX):
    """One CHW/HWC frame (tensor or array, uint8 or float) -> base64 JPEG."""
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


def _subsample(seq, k):
    """Uniformly pick k of len(seq), preserving temporal order."""
    n = len(seq)
    if k <= 0 or n <= k:
        return list(range(n))
    return [int(i) for i in np.linspace(0, n - 1, k).round().astype(int)]


# ------------------------------------------------------------------ contract

def load_model(model_name=VISION_MODEL):
    # Position-for-position with the local wrappers: Vgent unpacks this as
    # (self.processor, self.video_llm, self.image_processor, _).
    return None, OpenAIVLM(vision_model=model_name), None, None


def load_video(video_path, args):
    """Decode at args.fps into a uint8 TCHW tensor.

    Frame indices match models/utils.fetch_video(resize=False) exactly: same
    smart_nframes (including its FPS_MIN_FRAMES=128 floor) and the same
    linspace over the decoded range.
    """
    import decord

    vr = decord.VideoReader(video_path, num_threads=2)
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    frame_idx = [i for i in range(0, total_frames, max(1, round(video_fps)))]
    nframes = smart_nframes({"fps": args.fps}, total_frames=total_frames,
                            video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    h, w = vr[0].shape[:2]
    del vr
    kw = {}
    if DECODE_MAX_PX and max(h, w) > DECODE_MAX_PX:
        sc = DECODE_MAX_PX / max(h, w)
        kw = {"width": int(round(w * sc)), "height": int(round(h * sc))}
    vr = decord.VideoReader(video_path, num_threads=2, **kw)

    # Batched in slices: one get_batch over 3600 indices materialises the whole
    # decoded video at once and decord is happier in chunks.
    out = []
    for i in range(0, len(idx), 256):
        out.append(torch.from_numpy(vr.get_batch(idx[i:i + 256]).asnumpy()))
    video = torch.cat(out).permute(0, 3, 1, 2).contiguous()   # TCHW uint8
    sample_fps = round(nframes / max(total_frames, 1e-6) * video_fps, 2)
    return [video], None, None, frame_idx, sample_fps, [video], None


def mllm_response(video_llm, tokenizer, processor, text, image_inputs, video,
                  max_new_tokens=512, size_list=None, fps=None):
    """Text-only when `video` is None, else frames + text to the vision model."""
    client: OpenAIVLM = video_llm

    if video is None and not image_inputs:
        return client.chat([{"type": "text", "text": text}],
                           client.text_model, max_new_tokens)

    seq = []
    if video is not None:
        seq = video[0] if (isinstance(video, (list, tuple)) and len(video)
                           and hasattr(video[0], "__len__")) else video
    frames = list(image_inputs or []) + list(seq)

    n_in = len(frames)
    keep = _subsample(frames, MAX_FRAMES) if MAX_FRAMES else list(range(n_in))
    # A hard ceiling regardless: 320 low-detail frames is 906k tokens on
    # gpt-4o-mini, seven times its context window, and the request would 400.
    ceiling = max_images_for(client.vision_model, DETAIL)
    if len(keep) > ceiling:
        keep = [keep[i] for i in _subsample(keep, ceiling)]
    frames = [frames[i] for i in keep]

    content = [{"type": "text", "text": text}]
    for f in frames:
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{_to_b64(f)}", "detail": DETAIL}})
    return client.chat(content, client.vision_model, max_new_tokens,
                       n_frames=len(frames), dropped=n_in - len(frames))
