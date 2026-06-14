import os
import gc
import cv2
import uuid
import json
import random
import threading
from pathlib import Path
from typing import Optional, List

import subprocess
subprocess.run(
    "pip install --upgrade 'setuptools<81' wheel",
    shell=True, check=True,
)
subprocess.run(
    "pip install --no-build-isolation --no-deps "
    "git+https://github.com/inference-sh/Real-ESRGAN.git",
    shell=True, check=True,
)

import spaces
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from RealESRGAN import RealESRGAN

from gradio import Server
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

import aoti
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

app = Server()

BASE_DIR    = Path(__file__).resolve().parent
OUTPUT_DIR  = BASE_DIR / "outputs"
FRAMES_DIR  = BASE_DIR / "outputs" / "frames"
EXAMPLES_DIR = BASE_DIR / "example-file"

OUTPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)

MODEL_ID          = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
MAX_DIM           = 832
MIN_DIM           = 480
SQUARE_DIM        = 640
MULTIPLE_OF       = 16
MAX_SEED          = np.iinfo(np.int32).max
FIXED_FPS         = 16
MIN_FRAMES_MODEL  = 8
MAX_FRAMES_MODEL  = 80
MIN_DURATION      = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION      = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_LABEL = (
    torch.cuda.get_device_name(torch.cuda.current_device()).lower()
    if torch.cuda.is_available() else str(device).lower()
)

print("Downloading RealESRGAN x2 weights...")
hf_hub_download(
    repo_id="ai-forever/Real-ESRGAN",
    filename="RealESRGAN_x2.pth",
    local_dir="models/upscalers/",
)

class LazyRealESRGAN:
    def __init__(self, dev, scale):
        self.dev   = dev
        self.scale = scale
        self._model = None

    def _load(self):
        if self._model is None:
            self._model = RealESRGAN(self.dev, scale=self.scale)
            self._model.load_weights(
                f"models/upscalers/RealESRGAN_x{self.scale}.pth",
                download=False,
            )

    def predict(self, img: Image.Image) -> Image.Image:
        self._load()
        return self._model.predict(img)

esrgan_x2 = LazyRealESRGAN(device, scale=2)

print("Loading Wan 2.2 I2V 14B pipeline...")
pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        "cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers",
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

print("Loading Lightning LoRA weights...")
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v",
)
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2",
    load_into_transformer_2=True,
)
pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1.0, 1.0])
pipe.fuse_lora(adapter_names=["lightx2v"],   lora_scale=3.0, components=["transformer"])
pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1.0, components=["transformer_2"])
pipe.unload_lora_weights()

print("Quantizing models...")
quantize_(pipe.text_encoder,  Int8WeightOnlyConfig())
quantize_(pipe.transformer,   Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

print("Loading AOTI compiled modules...")
spaces.aoti_load(module=pipe.transformer,   repo_id="cbensimon/WanTransformer3DModel-sm120-cu130-raa")
spaces.aoti_load(module=pipe.transformer_2, repo_id="cbensimon/WanTransformer3DModel-sm120-cu130-raa")

pipe_lock = threading.Lock()

DEFAULT_PROMPT = "make this image come alive, cinematic motion, smooth animation"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, "
    "形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"
)

def resize_image(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w == h:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)
    aspect = w / h
    MAX_AR, MIN_AR = MAX_DIM / MIN_DIM, MIN_DIM / MAX_DIM
    img = image
    if aspect > MAX_AR:
        cw = int(round(h * MAX_AR))
        l  = (w - cw) // 2
        img = image.crop((l, 0, l + cw, h))
        tw, th = MAX_DIM, MIN_DIM
    elif aspect < MIN_AR:
        ch = int(round(w / MIN_AR))
        t  = (h - ch) // 2
        img = image.crop((0, t, w, t + ch))
        tw, th = MIN_DIM, MAX_DIM
    else:
        if w > h:
            tw = MAX_DIM; th = int(round(tw / aspect))
        else:
            th = MAX_DIM; tw = int(round(th * aspect))
    fw = max(MIN_DIM, min(MAX_DIM, round(tw / MULTIPLE_OF) * MULTIPLE_OF))
    fh = max(MIN_DIM, min(MAX_DIM, round(th / MULTIPLE_OF) * MULTIPLE_OF))
    return img.resize((fw, fh), Image.LANCZOS)


def get_num_frames(duration_seconds: float) -> int:
    return 1 + int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL, MAX_FRAMES_MODEL,
    ))


def extract_frames_from_video(video_path: str, duration_seconds: float) -> List[Image.Image]:
    """
    Extract one frame per whole second from the generated video.
    E.g. 3.5 s → frames at ~0.5 s, ~1.5 s, ~2.5 s (midpoints of each second bucket).
    Returns PIL images.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FIXED_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs   = total_frames / fps

    n_buckets = max(1, int(duration_seconds))
    frames_out = []
    for i in range(n_buckets):
        t = i + 0.5          # midpoint of each second bucket
        t = min(t, total_secs - 0.05)
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_out.append(Image.fromarray(rgb))
    cap.release()
    return frames_out


def upscale_frames(pil_frames: List[Image.Image], run_id: str):
    """
    Upscale each frame with RealESRGAN x2.
    Returns list of dicts:
      {orig_url, upscaled_url, orig_file, upscaled_file, label}
    """
    results = []
    for i, frame in enumerate(pil_frames):
        label = f"Frame {i+1} (~{i+1}s)"

        orig_name = f"{run_id}_orig_{i+1}.jpg"
        orig_path = FRAMES_DIR / orig_name
        frame.save(orig_path, format="JPEG", quality=92)

        try:
            up = esrgan_x2.predict(frame.convert("RGB"))
        except Exception as e:
            print(f"Upscale failed for frame {i+1}: {e}")
            up = frame

        up_name = f"{run_id}_up_{i+1}.jpg"
        up_path = FRAMES_DIR / up_name
        up.save(up_path, format="JPEG", quality=95)

        results.append({
            "label":         label,
            "orig_file":     orig_name,
            "upscaled_file": up_name,
            "orig_url":      f"/frames/{orig_name}",
            "upscaled_url":  f"/frames/{up_name}",
        })
    return results


@spaces.GPU(duration=120, size="xlarge")
def infer(
    image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    duration_seconds: float,
    guidance_scale: float,
    guidance_scale_2: float,
    seed: int,
    randomize_seed: bool,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized      = resize_image(image)
    num_frames   = get_num_frames(duration_seconds)

    with pipe_lock:
        frames = pipe(
            image=resized,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=resized.height,
            width=resized.width,
            num_frames=num_frames,
            guidance_scale=float(guidance_scale),
            guidance_scale_2=float(guidance_scale_2),
            num_inference_steps=int(steps),
            generator=torch.Generator(device="cuda").manual_seed(current_seed),
        ).frames[0]

    run_id  = uuid.uuid4().hex
    vid_name = f"video_{run_id}.mp4"
    vid_path = OUTPUT_DIR / vid_name
    export_to_video(frames, str(vid_path), fps=FIXED_FPS)

    # Extract + upscale frames
    pil_frames    = extract_frames_from_video(str(vid_path), duration_seconds)
    frame_results = upscale_frames(pil_frames, run_id)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vid_name, current_seed, frame_results


def get_example_items():
    return [
        {
            "url":    "/example-file/wan_i2v_input.JPG",
            "prompt": "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind. Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams.",
        },
        {
            "url":    "/example-file/wan22_input_2.jpg",
            "prompt": "A sleek lunar vehicle glides into view, kicking up moon dust as astronauts in white spacesuits hop aboard. In the background, a VTOL craft descends and lands silently. Ethereal aurora borealis ribbons dance across the star-filled sky.",
        },
        {
            "url":    "/example-file/kill_bill.jpeg",
            "prompt": "Uma Thurman's character holds her katana steady. Suddenly the blade begins to soften, melting like heated metal. The steel warps and droops, flowing downward in silvery rivulets. Her expression shifts from calm readiness to bewilderment.",
        },
    ]


@app.get("/example-file/{filename}")
async def example_file(filename: str):
    path = EXAMPLES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": f"Not found: {filename}"}, status_code=404)
    suffix = Path(filename).suffix.lower()
    mt = {".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",".webp":"image/webp"}.get(suffix,"image/jpeg")
    return FileResponse(path, media_type=mt)


@app.get("/frames/{filename}")
async def serve_frame(filename: str):
    path = FRAMES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/download/{filename}")
async def download_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, filename=filename, media_type="video/mp4")


@app.get("/download-frame/{filename}")
async def download_frame_file(filename: str):
    path = FRAMES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Frame not found"}, status_code=404)
    return FileResponse(path, filename=filename, media_type="image/jpeg")


@app.post("/api/generate")
async def generate_video_endpoint(
    prompt:           str          = Form(...),
    negative_prompt:  str          = Form(DEFAULT_NEGATIVE_PROMPT),
    seed:             str          = Form("42"),
    randomize_seed:   str          = Form("true"),
    duration:         str          = Form("3.5"),
    steps:            str          = Form("4"),
    guidance_scale:   str          = Form("1.0"),
    guidance_scale_2: str          = Form("1.0"),
    image: Optional[UploadFile]    = File(None),
):
    temp_path = None
    try:
        if image is None or not image.filename:
            return JSONResponse({"success": False, "error": "No image uploaded."}, status_code=400)

        suffix    = Path(image.filename).suffix or ".png"
        temp_path = OUTPUT_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
        content   = await image.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        pil_image = Image.open(temp_path).convert("RGB")

        vid_name, used_seed, frame_results = infer(
            image            = pil_image,
            prompt           = prompt,
            negative_prompt  = negative_prompt,
            steps            = int(steps),
            duration_seconds = float(duration),
            guidance_scale   = float(guidance_scale),
            guidance_scale_2 = float(guidance_scale_2),
            seed             = int(seed),
            randomize_seed   = (randomize_seed.lower() == "true"),
        )

        return JSONResponse({
            "success":       True,
            "seed":          used_seed,
            "url":           f"/download/{vid_name}",
            "filename":      vid_name,
            "device":        DEVICE_LABEL,
            "frame_results": frame_results,
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    examples      = get_example_items()
    examples_json = json.dumps(examples)
    min_dur       = MIN_DURATION
    max_dur       = MAX_DURATION
    default_prompt = DEFAULT_PROMPT
    default_neg    = DEFAULT_NEGATIVE_PROMPT

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Wan2.2-I2V-Fast</title>
  <link href="https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --ub-aubergine:      #2C001E;
      --ub-aubergine-dark: #1f0015;
      --ub-orange:         #E95420;
      --ub-orange-hover:   #c4461a;
      --ub-panel:          #3D3D3D;
      --ub-border:         rgba(255,255,255,0.1);
      --ub-text:           #FFFFFF;
      --ub-muted:          #b0b0b0;
      --ub-input:          #2b2b2b;
      --panel-radius:      8px;
    }}
    *  {{ box-sizing:border-box; font-family:'Ubuntu',sans-serif; margin:0; padding:0; }}
    body {{
      background: var(--ub-aubergine);
      color: var(--ub-text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}

    /* ── TOPBAR ── */
    .topbar {{
      background: var(--ub-aubergine-dark);
      padding: 0 24px;
      border-bottom: 1px solid var(--ub-border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 52px;
      flex-shrink: 0;
    }}
    .topbar-title {{
      font-weight: 700;
      letter-spacing: .5px;
      color: var(--ub-orange);
      font-size: 15px;
    }}
    .topbar-examples-btn {{
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(233,84,32,.12);
      border: 1px solid rgba(233,84,32,.35);
      color: var(--ub-orange);
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: background .2s, border-color .2s, transform .15s;
      font-family: 'Ubuntu', sans-serif;
      letter-spacing: .3px;
      white-space: nowrap;
    }}
    .topbar-examples-btn:hover {{
      background: rgba(233,84,32,.22);
      border-color: var(--ub-orange);
      transform: translateY(-1px);
    }}
    .btn-thumbs {{ display:flex; gap:3px; margin-right:2px; }}
    .btn-thumb  {{
      width:20px; height:20px; border-radius:3px;
      object-fit:cover; border:1px solid rgba(233,84,32,.4); display:block;
    }}
    .topbar-right {{ display:flex; align-items:center; gap:10px; }}
    .topbar-github-btn {{
      display: flex;
      align-items: center;
      gap: 6px;
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.18);
      color: #e0e0e0;
      padding: 6px 13px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      transition: background .2s, border-color .2s, color .2s, transform .15s;
      font-family: 'Ubuntu', sans-serif;
      letter-spacing: .3px;
      white-space: nowrap;
    }}
    .topbar-github-btn:hover {{
      background: rgba(255,255,255,.13);
      border-color: rgba(255,255,255,.4);
      color: #ffffff;
      transform: translateY(-1px);
    }}
    .topbar-github-btn svg {{ flex-shrink:0; }}

    /* ── CONTAINER ── */
    .container {{
      max-width: 1300px;
      margin: 0 auto;
      padding: 30px 20px;
      flex: 1;
      width: 100%;
    }}
    .header-text {{ text-align:center; margin-bottom:30px; }}
    .header-text h1 {{ font-size:2.2rem; margin-bottom:10px; }}
    .header-text p  {{ color:var(--ub-muted); font-size:14px; line-height:1.6; }}
    .badge {{
      display:inline-block;
      background:rgba(233,84,32,.15);
      border:1px solid rgba(233,84,32,.4);
      color:var(--ub-orange);
      font-size:12px; padding:3px 10px;
      border-radius:20px; margin-top:8px; font-weight:500;
    }}

    /* ── MAIN LAYOUT ── */
    .layout {{
      display: grid;
      grid-template-columns: 440px 1fr;
      gap: 24px;
      align-items: stretch;
      height: 720px;
    }}
    .panel {{
      background: var(--ub-panel);
      border-radius: var(--panel-radius);
      box-shadow: 0 8px 24px rgba(0,0,0,.25);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      height: 100%;
    }}
    .panel-header {{
      padding: 16px 20px;
      background: rgba(0,0,0,.2);
      border-bottom: 1px solid var(--ub-border);
      font-weight: 500;
      font-size: 1.05rem;
      flex-shrink: 0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .panel-body-scroll {{
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }}

    /* ── FORMS ── */
    .form-group {{ margin-bottom:18px; flex-shrink:0; }}
    .label {{
      display:block; font-weight:500; font-size:13px;
      color:var(--ub-muted); margin-bottom:7px;
      text-transform:uppercase; letter-spacing:.5px;
    }}
    .textarea, .input {{
      width:100%;
      background:var(--ub-input);
      border:1px solid var(--ub-border);
      color:var(--ub-text);
      padding:11px 12px; border-radius:4px;
      outline:none; font-size:14px;
      font-family:'Ubuntu',sans-serif;
      transition:border-color .2s;
    }}
    .textarea:focus, .input:focus {{ border-color:var(--ub-orange); }}
    .textarea {{ min-height:80px; resize:vertical; line-height:1.5; }}

    /* ── UPLOAD ZONE ── */
    .upload-zone {{
      background:var(--ub-input);
      border:1.5px dashed rgba(255,255,255,.2);
      border-radius:4px;
      cursor:pointer;
      transition:background .2s, border-color .2s;
      position:relative; overflow:hidden;
      aspect-ratio:16/9;
      display:flex; align-items:center; justify-content:center;
    }}
    .upload-zone:hover, .upload-zone.dragover {{
      border-color:var(--ub-orange);
      background:rgba(233,84,32,.05);
    }}
    .upload-zone input[type="file"] {{ display:none; }}
    .upload-placeholder {{
      display:flex; flex-direction:column;
      align-items:center; gap:10px;
      color:var(--ub-muted); pointer-events:none;
    }}
    .upload-placeholder svg {{ opacity:.5; }}
    .upload-placeholder span {{ font-size:13px; }}
    .preview-img {{
      position:absolute; inset:0;
      width:100%; height:100%;
      object-fit:contain; display:none; background:#111;
    }}
    .remove-img-btn {{
      position:absolute; top:8px; right:8px;
      background:rgba(0,0,0,.7); border:none; color:white;
      border-radius:50%; width:26px; height:26px;
      display:none; align-items:center; justify-content:center;
      cursor:pointer; font-size:14px; z-index:5;
      transition:background .2s;
    }}
    .remove-img-btn:hover {{ background:var(--ub-orange); }}

    /* ── SLIDER INPUT ── */
    .slider-wrap {{ display:flex; align-items:center; gap:12px; }}
    .slider-wrap input[type="range"] {{
      flex:1; -webkit-appearance:none;
      height:4px; background:var(--ub-input);
      border-radius:2px; outline:none; cursor:pointer;
      border:1px solid var(--ub-border);
    }}
    .slider-wrap input[type="range"]::-webkit-slider-thumb {{
      -webkit-appearance:none;
      width:16px; height:16px; border-radius:50%;
      background:var(--ub-orange); cursor:pointer;
    }}
    .slider-val {{
      min-width:40px; text-align:right;
      font-size:13px; color:var(--ub-text); font-weight:500;
    }}

    /* ── ADVANCED ACCORDION ── */
    .advanced-toggle {{
      width:100%; background:none;
      border:none; border-top:1px solid var(--ub-border);
      color:var(--ub-orange); text-align:left;
      padding:12px 0; font-weight:500; font-size:13px;
      cursor:pointer; display:flex;
      justify-content:space-between; align-items:center;
      flex-shrink:0; margin-top:4px;
    }}
    .adv-icon {{ font-weight:bold; font-size:18px; line-height:1; }}
    .advanced-body {{ display:none; padding-top:12px; flex-shrink:0; }}
    .advanced-body.open {{ display:block; }}
    .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}

    /* ── STATUS LOG ── */
    .status-container {{
      margin-top:16px; margin-bottom:16px;
      border:1px solid var(--ub-border);
      border-radius:4px; background:#200014;
      display:flex; flex-direction:column;
      flex:1; min-height:80px; max-height:150px;
    }}
    .status-header {{
      padding:7px 12px; font-size:10px; font-weight:700;
      color:var(--ub-muted); background:rgba(0,0,0,.4);
      border-bottom:1px solid var(--ub-border);
      text-transform:uppercase; letter-spacing:.6px; flex-shrink:0;
    }}
    .status-log {{
      padding:8px 10px;
      font-family:'Courier New',Courier,monospace;
      font-size:11.5px; color:#eee; overflow-y:auto;
      flex:1; display:flex; flex-direction:column; gap:3px;
    }}
    .log-time    {{ color:#666; margin-right:7px; }}
    .log-info    {{ color:#5bc0eb; }}
    .log-success {{ color:#9bc53d; }}
    .log-error   {{ color:#ff5e5b; }}
    .log-warn    {{ color:#f4c842; }}

    /* ── BUTTONS ── */
    .btn {{
      width:100%; padding:13px; border:none; border-radius:4px;
      font-size:15px; font-weight:700; cursor:pointer;
      transition:opacity .2s, background .2s; flex-shrink:0;
      letter-spacing:.3px; font-family:'Ubuntu',sans-serif;
    }}
    .btn-primary {{
      background:var(--ub-orange); color:white;
      box-shadow:0 4px 14px rgba(233,84,32,.3);
    }}
    .btn-primary:hover {{ background:var(--ub-orange-hover); }}
    .btn:disabled {{ opacity:.55; cursor:not-allowed; }}

    .action-icon {{
      display:none; background:none; border:none;
      color:var(--ub-muted); cursor:pointer; padding:4px;
      transition:color .2s;
    }}
    .action-icon:hover {{ color:var(--ub-orange); }}

    /* ── OUTPUT PANEL ── */
    .panel-body-output {{
      flex:1; display:flex; flex-direction:column;
      padding:0; position:relative;
    }}
    .output-stage {{
      position:absolute; top:0; left:0; right:0; bottom:0;
      background:#0d0008; overflow:hidden;
      display:flex; align-items:center; justify-content:center;
    }}
    .output-empty {{ color:var(--ub-muted); text-align:center; z-index:1; }}
    .output-empty svg {{ opacity:.3; margin-bottom:12px; }}
    .output-empty div {{ font-size:14px; }}
    .output-video {{
      position:absolute; top:0; left:0;
      width:100%; height:100%;
      object-fit:contain; display:none; background:#000;
    }}

    /* ── LOADER ── */
    .loader {{
      position:absolute; inset:0;
      background:rgba(15,0,10,.78);
      backdrop-filter:blur(6px);
      display:none; flex-direction:column;
      align-items:center; justify-content:center; z-index:20;
    }}
    .spinner-single {{
      width:52px; height:52px;
      border:3px solid rgba(255,255,255,.08);
      border-top-color:var(--ub-orange);
      border-radius:50%;
      animation:spin 1s cubic-bezier(.4,0,.2,1) infinite;
      margin-bottom:18px;
    }}
    .loader-text {{
      font-weight:500; font-size:14px; color:#fff;
      letter-spacing:1px;
      animation:pulse 1.5s ease-in-out infinite;
    }}
    .loader-sub {{ font-size:12px; color:var(--ub-muted); margin-top:6px; }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.45}} }}
    @keyframes spin  {{ to{{transform:rotate(360deg)}} }}

    .seed-badge {{
      display:none; position:absolute; bottom:12px; right:12px;
      background:rgba(0,0,0,.65); color:#ccc;
      padding:5px 10px; border-radius:20px;
      font-size:12px; backdrop-filter:blur(4px); z-index:5;
    }}
    .checkbox-label {{
      display:flex; align-items:center; gap:8px;
      font-size:14px; color:var(--ub-text); cursor:pointer;
    }}
    .checkbox-label input {{
      cursor:pointer; accent-color:var(--ub-orange); width:15px; height:15px;
    }}

    /* ── FRAME UPSCALE SECTION ── */
    .frames-section {{
      margin-top:36px;
    }}
    .frames-section-header {{
      display:flex; align-items:center; gap:12px;
      border-bottom:1px solid var(--ub-border);
      padding-bottom:12px; margin-bottom:20px;
    }}
    .frames-section-header h3 {{
      font-size:1.1rem; color:var(--ub-text);
    }}
    .frames-badge {{
      background:rgba(155,197,61,.15);
      border:1px solid rgba(155,197,61,.35);
      color:#9bc53d;
      font-size:11px; font-weight:600;
      padding:3px 9px; border-radius:12px;
    }}
    .frames-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
      gap:22px;
    }}

    /* ── COMPARISON CARD ── */
    .cmp-card {{
      background:var(--ub-panel);
      border-radius:var(--panel-radius);
      overflow:hidden;
      border:1px solid var(--ub-border);
      box-shadow:0 4px 16px rgba(0,0,0,.25);
    }}
    .cmp-card-header {{
      display:flex; align-items:center;
      justify-content:space-between;
      padding:10px 14px;
      background:rgba(0,0,0,.25);
      border-bottom:1px solid var(--ub-border);
      font-size:13px; font-weight:500;
    }}
    .cmp-label {{ color:var(--ub-text); }}
    .cmp-dl-btn {{
      display:flex; align-items:center; gap:5px;
      background:rgba(233,84,32,.15);
      border:1px solid rgba(233,84,32,.35);
      color:var(--ub-orange);
      padding:4px 10px; border-radius:12px;
      font-size:11px; font-weight:600;
      cursor:pointer; transition:background .2s;
      font-family:'Ubuntu',sans-serif;
    }}
    .cmp-dl-btn:hover {{ background:rgba(233,84,32,.3); }}

    /* ── SLIDER COMPARE ── */
    .cmp-stage {{
      position:relative;
      width:100%;
      aspect-ratio:16/9;
      overflow:hidden;
      cursor:ew-resize;
      user-select:none;
      background:#111;
    }}
    .cmp-img {{
      position:absolute; top:0; left:0;
      width:100%; height:100%;
      object-fit:contain;
    }}
    .cmp-img-before {{ z-index:1; }}
    .cmp-img-after  {{
      z-index:2;
      clip-path:inset(0 50% 0 0);   /* initially right half hidden */
    }}
    /* the draggable divider */
    .cmp-divider {{
      position:absolute; top:0; bottom:0;
      left:50%;
      width:2px;
      background:var(--ub-orange);
      z-index:10;
      pointer-events:none;
    }}
    .cmp-handle {{
      position:absolute; top:50%;
      left:50%;
      transform:translate(-50%,-50%);
      width:32px; height:32px;
      border-radius:50%;
      background:var(--ub-orange);
      z-index:11;
      display:flex; align-items:center; justify-content:center;
      pointer-events:none;
      box-shadow:0 2px 8px rgba(0,0,0,.5);
    }}
    .cmp-handle svg {{ flex-shrink:0; }}
    /* labels */
    .cmp-pill {{
      position:absolute; bottom:10px;
      background:rgba(0,0,0,.65);
      color:#fff; font-size:10px; font-weight:700;
      padding:3px 8px; border-radius:10px;
      z-index:12; pointer-events:none;
      backdrop-filter:blur(3px);
      letter-spacing:.4px;
    }}
    .cmp-pill-left  {{ left:10px; }}
    .cmp-pill-right {{ right:10px; color:#9bc53d; }}

    /* ── EXAMPLES SECTION ── */
    .examples-section {{ margin-top:40px; scroll-margin-top:20px; }}
    .examples-section-header {{
      display:flex; align-items:center; justify-content:space-between;
      border-bottom:1px solid var(--ub-border);
      padding-bottom:12px; margin-bottom:20px;
    }}
    .examples-section-header h3 {{ font-size:1.1rem; color:var(--ub-text); }}
    .examples-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
      gap:18px;
    }}
    .ex-card {{
      background:var(--ub-panel); border-radius:6px;
      overflow:hidden; cursor:pointer;
      transition:transform .2s, box-shadow .2s, border-color .2s;
      border:1px solid var(--ub-border);
    }}
    .ex-card:hover {{
      transform:translateY(-4px);
      box-shadow:0 10px 28px rgba(0,0,0,.4);
      border-color:rgba(233,84,32,.5);
    }}
    .ex-img-wrap {{
      width:100%; aspect-ratio:16/9;
      background:#1a001a; overflow:hidden; position:relative;
    }}
    .ex-img-wrap.loading::before {{
      content:''; position:absolute; inset:0;
      background:linear-gradient(90deg,#2c001e 25%,#3d1030 50%,#2c001e 75%);
      background-size:200% 100%;
      animation:shimmer 1.5s infinite; z-index:1;
    }}
    @keyframes shimmer {{
      0%{{background-position:-200% 0}} 100%{{background-position:200% 0}}
    }}
    .ex-img-wrap img {{
      width:100%; height:100%; object-fit:cover; display:block;
      transition:transform .3s; position:relative; z-index:2;
    }}
    .ex-card:hover .ex-img-wrap img {{ transform:scale(1.04); }}
    .ex-use-badge {{
      position:absolute; top:8px; left:8px; z-index:3;
      background:var(--ub-orange); color:white;
      font-size:10px; font-weight:700; padding:3px 8px; border-radius:10px;
      opacity:0; transition:opacity .2s; pointer-events:none;
    }}
    .ex-card:hover .ex-use-badge {{ opacity:1; }}
    .ex-card p {{
      padding:10px 12px 13px; font-size:12px;
      color:var(--ub-muted); line-height:1.5;
      display:-webkit-box; -webkit-line-clamp:3;
      -webkit-box-orient:vertical; overflow:hidden;
    }}

    @media(max-width:900px) {{
      .layout {{ grid-template-columns:1fr; height:auto; }}
      .panel-body-output {{ height:380px; flex:none; }}
      .output-stage {{ position:relative; height:100%; }}
      .btn-thumbs {{ display:none; }}
      .frames-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>

<div class="topbar">
  <span class="topbar-title">Wan2.2-I2V-Fast — Lightning LoRA · FP8 · AoT Compiled</span>
  <div class="topbar-right">

    <!-- GitHub link -->
    <a class="topbar-github-btn"
       href="https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast"
       target="_blank" rel="noopener noreferrer" title="View source on GitHub">
      <svg width="15" height="15" fill="currentColor" viewBox="0 0 24 24">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385
          .6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235
          -3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695
          -.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23
          1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605
          -2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225
          -.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23
          .96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23
          3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225
          0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22
          0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57
          A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
      </svg>
      GitHub
    </a>

    <!-- Examples scroll button -->
    <button class="topbar-examples-btn" id="examplesScrollBtn" title="Browse examples">
      <div class="btn-thumbs" id="btnThumbs"></div>
      <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24" style="margin-right:4px;">
        <rect x="3" y="3" width="7" height="7" rx="1"></rect>
        <rect x="14" y="3" width="7" height="7" rx="1"></rect>
        <rect x="3" y="14" width="7" height="7" rx="1"></rect>
        <rect x="14" y="14" width="7" height="7" rx="1"></rect>
      </svg>
      Examples
    </button>

  </div>
</div>

<div class="container">
  <div class="header-text">
    <h1>Wan2.2 Image-to-Video</h1>
    <p>Upload an image and describe the animation. Generates video in 4–8 steps with Lightning LoRA.<br>
       Frames are automatically extracted &amp; upscaled with RealESRGAN ×2.</p>
    <span class="badge">14B · FP8 Quantized · ZeroGPU · RealESRGAN Upscale</span>
  </div>

  <div class="layout">
    <!-- Left: Settings -->
    <div class="panel">
      <div class="panel-header">Settings</div>
      <div class="panel-body-scroll">

        <div class="form-group">
          <label class="label">Input Image</label>
          <div class="upload-zone" id="dropZone">
            <input type="file" id="fileInput" accept="image/*" />
            <div class="upload-placeholder" id="uploadPlaceholder">
              <svg width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                <path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
              </svg>
              <span>Click or drag &amp; drop an image</span>
            </div>
            <img id="previewImg" class="preview-img" alt="Preview" />
            <button id="removeImgBtn" class="remove-img-btn" title="Remove">×</button>
          </div>
        </div>

        <div class="form-group">
          <label class="label">Prompt</label>
          <textarea id="promptInput" class="textarea" placeholder="Describe the animation...">{default_prompt}</textarea>
        </div>

        <div class="form-group">
          <label class="label">Duration — <span id="durationVal">3.5</span>s</label>
          <div class="slider-wrap">
            <input type="range" id="duration" min="{min_dur}" max="{max_dur}" step="0.1" value="3.5"
              oninput="document.getElementById('durationVal').textContent=parseFloat(this.value).toFixed(1)">
            <span class="slider-val">{min_dur}–{max_dur}s</span>
          </div>
        </div>

        <button class="advanced-toggle" id="advToggle">
          <span>Advanced Settings</span>
          <span class="adv-icon" id="advIcon">+</span>
        </button>
        <div class="advanced-body" id="advBody">
          <div class="form-group">
            <label class="label">Negative Prompt</label>
            <textarea id="negPrompt" class="textarea" style="min-height:60px;font-size:12px;">{default_neg}</textarea>
          </div>
          <div class="grid-2">
            <div class="form-group">
              <label class="label">Steps</label>
              <input type="number" id="steps" class="input" value="4" min="1" max="30">
            </div>
            <div class="form-group">
              <label class="label">Seed</label>
              <input type="number" id="seed" class="input" value="42">
            </div>
            <div class="form-group">
              <label class="label">Guidance (Stage 1)</label>
              <input type="number" id="guidance" class="input" value="1.0" step="0.5" min="0" max="10">
            </div>
            <div class="form-group">
              <label class="label">Guidance (Stage 2)</label>
              <input type="number" id="guidance2" class="input" value="1.0" step="0.5" min="0" max="10">
            </div>
            <div class="form-group" style="grid-column:span 2">
              <label class="checkbox-label">
                <input type="checkbox" id="randomize" checked> Randomize Seed
              </label>
            </div>
          </div>
        </div>

        <div class="status-container">
          <div class="status-header">Execution Log</div>
          <div class="status-log" id="statusLog">
            <div><span class="log-time">[{DEVICE_LABEL}]</span><span>System ready...</span></div>
          </div>
        </div>

        <button class="btn btn-primary" id="runBtn">Generate Video</button>
      </div>
    </div>

    <!-- Right: Output -->
    <div class="panel">
      <div class="panel-header">
        <span>Output</span>
        <button id="downloadBtn" class="action-icon" title="Download Video">
          <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
          </svg>
        </button>
      </div>
      <div class="panel-body-output">
        <div class="output-stage" id="outputStage">
          <div class="output-empty" id="outputEmpty">
            <svg width="52" height="52" fill="none" stroke="currentColor" stroke-width="1.2" viewBox="0 0 24 24">
              <polygon points="23 7 16 12 23 17 23 7"></polygon>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
            <div>Video will appear here</div>
          </div>
          <video id="outputVideo" class="output-video" controls autoplay loop></video>
          <div class="seed-badge" id="seedBadge"></div>
          <div class="loader" id="loader">
            <div class="spinner-single"></div>
            <div class="loader-text">Generating video + upscaling frames...</div>
            <div class="loader-sub">This may take a minute</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Frame Upscale Results -->
  <div class="frames-section" id="framesSection" style="display:none;">
    <div class="frames-section-header">
      <h3>Frame Upscale Results</h3>
      <span class="frames-badge" id="framesBadge">RealESRGAN ×2</span>
    </div>
    <div class="frames-grid" id="framesGrid"></div>
  </div>

  <!-- Examples -->
  <div class="examples-section" id="examplesSection">
    <div class="examples-section-header">
      <h3>Examples</h3>
    </div>
    <div class="examples-grid" id="examplesGrid"></div>
  </div>
</div>

<script>
const examples = {examples_json};
let currentFile     = null;
let currentFilename = "";

const dropZone          = document.getElementById('dropZone');
const fileInput         = document.getElementById('fileInput');
const uploadPlaceholder = document.getElementById('uploadPlaceholder');
const previewImg        = document.getElementById('previewImg');
const removeImgBtn      = document.getElementById('removeImgBtn');
const promptInput       = document.getElementById('promptInput');
const runBtn            = document.getElementById('runBtn');
const downloadBtn       = document.getElementById('downloadBtn');
const statusLog         = document.getElementById('statusLog');
const outputVideo       = document.getElementById('outputVideo');
const outputEmpty       = document.getElementById('outputEmpty');
const loader            = document.getElementById('loader');
const seedBadge         = document.getElementById('seedBadge');
const framesSection     = document.getElementById('framesSection');
const framesGrid        = document.getElementById('framesGrid');
const examplesSection   = document.getElementById('examplesSection');

/* ── scroll to examples ── */
document.getElementById('examplesScrollBtn').onclick = () =>
  examplesSection.scrollIntoView({{behavior:'smooth',block:'start'}});

/* ── topbar thumbnails ── */
const btnThumbs = document.getElementById('btnThumbs');
examples.slice(0,3).forEach(ex => {{
  const img = document.createElement('img');
  img.src = ex.url; img.className = 'btn-thumb'; img.alt = '';
  img.onerror = () => {{ img.style.display='none'; }};
  btnThumbs.appendChild(img);
}});

/* ── log ── */
function logMsg(msg, cls='') {{
  const d = document.createElement('div');
  const t = new Date().toLocaleTimeString('en-US',{{hour12:false}});
  d.innerHTML = `<span class="log-time">[${{t}}]</span><span class="${{cls}}">${{msg}}</span>`;
  statusLog.appendChild(d);
  statusLog.scrollTop = statusLog.scrollHeight;
}}

/* ── advanced ── */
document.getElementById('advToggle').onclick = function() {{
  const b = document.getElementById('advBody');
  b.classList.toggle('open');
  document.getElementById('advIcon').innerText = b.classList.contains('open') ? '−' : '+';
}};

/* ── image upload ── */
function setImage(file) {{
  if (!file) return;
  currentFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewImg.style.display = 'block';
  uploadPlaceholder.style.display = 'none';
  removeImgBtn.style.display = 'flex';
  logMsg(`Image loaded: ${{file.name}}`,'log-info');
}}
function clearImage() {{
  currentFile = null;
  previewImg.src = '';
  previewImg.style.display = 'none';
  uploadPlaceholder.style.display = 'flex';
  removeImgBtn.style.display = 'none';
  fileInput.value = '';
}}
dropZone.onclick = e => {{
  if (e.target===dropZone||e.target===uploadPlaceholder||uploadPlaceholder.contains(e.target))
    fileInput.click();
}};
fileInput.onchange  = e => {{ if(e.target.files[0]) setImage(e.target.files[0]); fileInput.value=''; }};
dropZone.ondragover  = e => {{ e.preventDefault(); dropZone.classList.add('dragover'); }};
dropZone.ondragleave = ()=> dropZone.classList.remove('dragover');
dropZone.ondrop = e => {{
  e.preventDefault(); dropZone.classList.remove('dragover');
  if(e.dataTransfer.files[0]) setImage(e.dataTransfer.files[0]);
}};
removeImgBtn.onclick = e => {{ e.stopPropagation(); clearImage(); }};

/* ── download video ── */
downloadBtn.onclick = () => {{
  if(!currentFilename) return;
  const a=document.createElement('a'); a.href=`/download/${{currentFilename}}`; a.download=currentFilename; a.click();
}};

/* ── comparison slider logic ── */
function initCompare(stage) {{
  const after    = stage.querySelector('.cmp-img-after');
  const divider  = stage.querySelector('.cmp-divider');
  const handle   = stage.querySelector('.cmp-handle');
  let dragging = false;

  function setPos(x) {{
    const rect = stage.getBoundingClientRect();
    let pct = Math.max(0, Math.min(1, (x - rect.left) / rect.width));
    const pctPx = (pct*100).toFixed(2);
    after.style.clipPath   = `inset(0 ${{(100-pct*100).toFixed(2)}}% 0 0)`;
    divider.style.left     = pctPx + '%';
    handle.style.left      = pctPx + '%';
  }}

  stage.addEventListener('mousedown',  e=>{{ dragging=true; setPos(e.clientX); }});
  window.addEventListener('mouseup',   ()=>{{ dragging=false; }});
  window.addEventListener('mousemove', e=>{{ if(dragging) setPos(e.clientX); }});

  stage.addEventListener('touchstart', e=>{{ dragging=true; setPos(e.touches[0].clientX); }},{{passive:true}});
  window.addEventListener('touchend',  ()=>{{ dragging=false; }});
  window.addEventListener('touchmove', e=>{{ if(dragging) setPos(e.touches[0].clientX); }},{{passive:true}});
}}

/* ── build frame comparison cards ── */
function buildFrameCards(frameResults) {{
  framesGrid.innerHTML = '';
  if(!frameResults || frameResults.length===0) return;

  framesSection.style.display = 'block';
  document.getElementById('framesBadge').textContent =
    `RealESRGAN ×2 — ${{frameResults.length}} frame${{frameResults.length>1?'s':''}}`;

  frameResults.forEach(fr => {{
    const card = document.createElement('div');
    card.className = 'cmp-card';

    // header
    const hdr = document.createElement('div');
    hdr.className = 'cmp-card-header';
    hdr.innerHTML = `
      <span class="cmp-label">${{fr.label}}</span>
      <button class="cmp-dl-btn" onclick="dlFrame('${{fr.upscaled_file}}')">
        <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5"
          stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Download upscaled
      </button>`;

    // stage
    const stage = document.createElement('div');
    stage.className = 'cmp-stage';
    stage.innerHTML = `
      <img class="cmp-img cmp-img-before" src="${{fr.orig_url}}" alt="Original">
      <img class="cmp-img cmp-img-after"  src="${{fr.upscaled_url}}" alt="Upscaled">
      <div class="cmp-divider"></div>
      <div class="cmp-handle">
        <svg width="16" height="16" fill="none" stroke="white" stroke-width="2.5"
          stroke-linecap="round" stroke-linejoin="round" viewBox="0 0 24 24">
          <polyline points="15 18 9 12 15 6"></polyline>
          <polyline points="9 18 3 12 9 6" style="transform:translateX(6px)"></polyline>
        </svg>
      </div>
      <span class="cmp-pill cmp-pill-left">ORIGINAL</span>
      <span class="cmp-pill cmp-pill-right">UPSCALED ×2</span>`;

    card.appendChild(hdr);
    card.appendChild(stage);
    framesGrid.appendChild(card);
    initCompare(stage);
  }});


}}

function dlFrame(filename) {{
  const a=document.createElement('a'); a.href=`/download-frame/${{filename}}`; a.download=filename; a.click();
}}

/* ── generate ── */
runBtn.onclick = async () => {{
  const prompt = promptInput.value.trim();
  if(!prompt)       {{ logMsg('Prompt is required.','log-error'); return; }}
  if(!currentFile)  {{ logMsg('Please upload an input image first.','log-error'); return; }}

  logMsg('Initializing generation...','log-info');

  const fd = new FormData();
  fd.append('prompt',           prompt);
  fd.append('negative_prompt',  document.getElementById('negPrompt').value);
  fd.append('seed',             document.getElementById('seed').value);
  fd.append('randomize_seed',   document.getElementById('randomize').checked);
  fd.append('duration',         document.getElementById('duration').value);
  fd.append('steps',            document.getElementById('steps').value);
  fd.append('guidance_scale',   document.getElementById('guidance').value);
  fd.append('guidance_scale_2', document.getElementById('guidance2').value);
  fd.append('image', currentFile);

  loader.style.display      = 'flex';
  runBtn.disabled           = true;
  downloadBtn.style.display = 'none';
  seedBadge.style.display   = 'none';
  outputVideo.style.display = 'none';
  outputEmpty.style.display = 'flex';
  framesSection.style.display = 'none';

  logMsg('Sending to Wan 2.2 I2V pipeline (FP8 + Lightning LoRA + RealESRGAN)...','log-info');

  try {{
    const res  = await fetch('/api/generate', {{method:'POST', body:fd}});
    const data = await res.json();

    if(data.success) {{
      logMsg(`Video done! Seed: ${{data.seed}}`, 'log-success');
      logMsg(`Upscaled ${{(data.frame_results||[]).length}} frame(s) with RealESRGAN ×2`, 'log-success');
      currentFilename = data.filename;

      outputVideo.src = data.url;
      outputVideo.style.display = 'block';
      outputEmpty.style.display = 'none';
      downloadBtn.style.display = 'block';
      seedBadge.innerText = `Seed: ${{data.seed}}`;
      seedBadge.style.display = 'block';

      buildFrameCards(data.frame_results || []);
    }} else {{
      logMsg('Error: ' + data.error, 'log-error');
    }}
  }} catch(e) {{
    logMsg('Network or server error.','log-error');
  }} finally {{
    loader.style.display = 'none';
    runBtn.disabled = false;
    logMsg('Ready for next input.','');
  }}
}};

/* ── load example ── */
async function loadExample(url, prompt) {{
  clearImage();
  promptInput.value = prompt;
  logMsg('Loading example image...','log-info');
  try {{
    const res = await fetch(url);
    if(!res.ok) throw new Error(`HTTP ${{res.status}}`);
    const blob = await res.blob();
    const filename = url.split('/').pop();
    setImage(new File([blob], filename, {{type:blob.type||'image/jpeg'}}));
    window.scrollTo({{top:0, behavior:'smooth'}});
    logMsg('Example loaded — ready to generate.','log-success');
  }} catch(e) {{
    logMsg(`Failed to load example: ${{e.message}}`, 'log-error');
  }}
}}

/* ── build example cards ── */
const exGrid = document.getElementById('examplesGrid');
examples.forEach(ex => {{
  const card = document.createElement('div');
  card.className = 'ex-card';

  const wrap = document.createElement('div');
  wrap.className = 'ex-img-wrap loading';

  const img = document.createElement('img');
  img.alt = 'Example'; img.loading = 'lazy';
  img.onload  = () => {{ wrap.classList.remove('loading'); }};
  img.onerror = () => {{
    wrap.classList.remove('loading');
    wrap.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#555;font-size:12px;">No preview</div>`;
  }};
  img.src = ex.url;

  const badge = document.createElement('span');
  badge.className = 'ex-use-badge'; badge.textContent = 'Use this';

  wrap.appendChild(img); wrap.appendChild(badge);

  const p = document.createElement('p');
  p.textContent = ex.prompt;

  card.appendChild(wrap); card.appendChild(p);
  card.onclick = () => loadExample(ex.url, ex.prompt);
  exGrid.appendChild(card);
}});
</script>
</body>
</html>
"""

app.launch()
