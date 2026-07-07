import spaces
import torch
from diffusers.pipelines.wan.pipeline_wan_i2v import WanImageToVideoPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.utils.export_utils import export_to_video
from diffusers import Flux2KleinPipeline
import gradio as gr
import tempfile
import numpy as np
from PIL import Image
import random
import gc
import cv2
import os
import base64
from io import BytesIO
from typing import List

from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization import Int8WeightOnlyConfig

import aoti
from typing import Iterable

# --------------------------- theme ---------------------------

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red", c50="#FFF0E5", c100="#FFE0CC", c200="#FFC299", c300="#FFA366",
    c400="#FF8533", c500="#FF4500", c600="#E63E00", c700="#CC3700", c800="#B33000",
    c900="#992900", c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self, *, primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate, text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600", block_border_width="3px",
            block_shadow="*shadow_drop_lg", button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px", color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --------------------------- theme ---------------------------

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

MAX_DIM = 832
MIN_DIM = 480
SQUARE_DIM = 640
MULTIPLE_OF = 16

MAX_SEED = np.iinfo(np.int32).max

FIXED_FPS = 16
MIN_FRAMES_MODEL = 8
MAX_FRAMES_MODEL = 80

MIN_DURATION = round(MIN_FRAMES_MODEL / FIXED_FPS, 1)
MAX_DURATION = round(MAX_FRAMES_MODEL / FIXED_FPS, 1)

device = "cuda"

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    transformer=WanTransformer3DModel.from_pretrained(
        'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    transformer_2=WanTransformer3DModel.from_pretrained(
        'cbensimon/Wan2.2-I2V-A14B-bf16-Diffusers',
        subfolder='transformer_2',
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    ),
    torch_dtype=torch.bfloat16,
).to('cuda')

pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v"
)
kwargs_lora = {}
kwargs_lora["load_into_transformer_2"] = True
pipe.load_lora_weights(
    "Kijai/WanVideo_comfy",
    weight_name="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    adapter_name="lightx2v_2", **kwargs_lora
)
pipe.set_adapters(["lightx2v", "lightx2v_2"], adapter_weights=[1., 1.])
pipe.fuse_lora(adapter_names=["lightx2v"], lora_scale=3., components=["transformer"])
pipe.fuse_lora(adapter_names=["lightx2v_2"], lora_scale=1., components=["transformer_2"])
pipe.unload_lora_weights()

quantize_(pipe.text_encoder, Int8WeightOnlyConfig())
quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
quantize_(pipe.transformer_2, Float8DynamicActivationFloat8WeightConfig())

spaces.aoti_load(
    module=pipe.transformer,
    repo_id='cbensimon/WanTransformer3DModel-sm120-cu130-raa',
)
spaces.aoti_load(
    module=pipe.transformer_2,
    repo_id='cbensimon/WanTransformer3DModel-sm120-cu130-raa',
)

print("Loading FLUX.2 Klein 4B model...")
klein_pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
).to(device)
print("FLUX.2 Klein 4B loaded successfully.")

default_prompt_i2v = "make this image come alive, cinematic motion, smooth animation"
default_negative_prompt = "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走"

def resize_image(image: Image.Image) -> Image.Image:
    """
    Resizes an image to fit within the model's constraints, preserving aspect ratio as much as possible.
    """
    width, height = image.size

    # Handle square case
    if width == height:
        return image.resize((SQUARE_DIM, SQUARE_DIM), Image.LANCZOS)

    aspect_ratio = width / height

    MAX_ASPECT_RATIO = MAX_DIM / MIN_DIM
    MIN_ASPECT_RATIO = MIN_DIM / MAX_DIM

    image_to_resize = image

    if aspect_ratio > MAX_ASPECT_RATIO:
        # Very wide image -> crop width to fit 832x480 aspect ratio
        target_w, target_h = MAX_DIM, MIN_DIM
        crop_width = int(round(height * MAX_ASPECT_RATIO))
        left = (width - crop_width) // 2
        image_to_resize = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < MIN_ASPECT_RATIO:
        # Very tall image -> crop height to fit 480x832 aspect ratio
        target_w, target_h = MIN_DIM, MAX_DIM
        crop_height = int(round(width / MIN_ASPECT_RATIO))
        top = (height - crop_height) // 2
        image_to_resize = image.crop((0, top, width, top + crop_height))
    else:
        if width > height:  # Landscape
            target_w = MAX_DIM
            target_h = int(round(target_w / aspect_ratio))
        else:  # Portrait
            target_h = MAX_DIM
            target_w = int(round(target_h * aspect_ratio))

    final_w = round(target_w / MULTIPLE_OF) * MULTIPLE_OF
    final_h = round(target_h / MULTIPLE_OF) * MULTIPLE_OF

    final_w = max(MIN_DIM, min(MAX_DIM, final_w))
    final_h = max(MIN_DIM, min(MAX_DIM, final_h))

    return image_to_resize.resize((final_w, final_h), Image.LANCZOS)


def get_num_frames(duration_seconds: float):
    return 1 + int(np.clip(
        int(round(duration_seconds * FIXED_FPS)),
        MIN_FRAMES_MODEL,
        MAX_FRAMES_MODEL,
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
    total_secs = total_frames / fps

    n_buckets = max(1, int(duration_seconds))
    frames_out = []
    for i in range(n_buckets):
        t = i + 0.5
        t = min(t, total_secs - 0.05)
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_out.append(Image.fromarray(rgb))
    cap.release()
    return frames_out


def update_dimensions_for_klein(image: Image.Image):
    """Calculate dimensions for Flux2Klein, snapped to multiples of 16."""
    w, h = image.size
    scale = min(1024 / w, 1024 / h)
    nw = int(w * scale)
    nh = int(h * scale)
    return (nw // 16) * 16, (nh // 16) * 16


def upscale_frames_batch(frames: List[Image.Image], progress=None) -> List[Image.Image]:
    """
    Upscale ALL extracted frames in a single pass (no separate GPU decorator).
    Called from within the main @spaces.GPU function so everything
    shares one GPU session.
    """
    upscaled = []
    for i, frame in enumerate(frames):
        if progress:
            progress(0.5 + 0.5 * (i / len(frames)),
                     desc=f"Upscaling frame {i+1}/{len(frames)}...")
        target_w, target_h = update_dimensions_for_klein(frame)
        current_seed = random.randint(0, MAX_SEED)
        result = klein_pipe(
            prompt="high quality, ultra detailed, sharp focus, 8k resolution",
            image=frame,
            height=target_h,
            width=target_w,
            guidance_scale=1.0,
            num_inference_steps=4,
            generator=torch.Generator(device=device).manual_seed(current_seed),
        ).images[0]
        upscaled.append(result)
    return upscaled


def get_duration(
    input_image,
    prompt,
    steps,
    negative_prompt,
    duration_seconds,
    guidance_scale,
    guidance_scale_2,
    seed,
    randomize_seed,
    progress,
):
    BASE_FRAMES_HEIGHT_WIDTH = 81 * 832 * 624
    BASE_STEP_DURATION = 15
    width, height = resize_image(input_image).size
    num_frames = get_num_frames(duration_seconds)
    factor = num_frames * width * height / BASE_FRAMES_HEIGHT_WIDTH
    step_duration = BASE_STEP_DURATION * factor ** 1.5
    n_upscale_frames = max(1, int(duration_seconds))
    upscale_budget = n_upscale_frames * 15
    return 10 + int(steps) * step_duration + upscale_budget


@spaces.GPU(duration=get_duration, size="xlarge")
def generate_and_upscale_gpu(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Single GPU session: generate video → extract frames → upscale ALL frames.
    Everything runs under one @spaces.GPU allocation.
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    # ── Step 1: Generate video ──────────────────────────────────
    progress(0, desc="Generating video...")
    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    # ── Step 2: Extract frames (1 per second) ───────────────────
    progress(0.5, desc="Extracting frames...")
    frames = extract_frames_from_video(video_path, duration_seconds)

    if not frames:
        return video_path, current_seed, []

    # ── Step 3: Upscale ALL frames in this same GPU session ─────
    upscaled_frames = upscale_frames_batch(frames, progress)

    upscaled_pairs = list(zip(frames, upscaled_frames))
    return video_path, current_seed, upscaled_pairs


@spaces.GPU(duration=get_duration, size="xlarge")
def generate_video(
    input_image,
    prompt,
    steps=4,
    negative_prompt=default_negative_prompt,
    duration_seconds=MAX_DURATION,
    guidance_scale=1,
    guidance_scale_2=1,
    seed=42,
    randomize_seed=False,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate video only (used by Examples which don't need upscaling).
    """
    if input_image is None:
        raise gr.Error("Please upload an input image.")

    num_frames = get_num_frames(duration_seconds)
    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    resized_image = resize_image(input_image)

    output_frames_list = pipe(
        image=resized_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=resized_image.height,
        width=resized_image.width,
        num_frames=num_frames,
        guidance_scale=float(guidance_scale),
        guidance_scale_2=float(guidance_scale_2),
        num_inference_steps=int(steps),
        generator=torch.Generator(device="cuda").manual_seed(current_seed),
    ).frames[0]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        video_path = tmpfile.name
    export_to_video(output_frames_list, video_path, fps=FIXED_FPS)

    return video_path, current_seed


def pil_to_temp_path(img: Image.Image, prefix: str = "frame") -> str:
    """Save a PIL image to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".png", prefix=prefix + "_", delete=False) as f:
        img.save(f, format="PNG")
        return f.name


def run_pipeline(
    input_image, prompt, steps, negative_prompt,
    duration_seconds, guidance_scale, guidance_scale_2,
    seed, randomize_seed,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Orchestrate the full pipeline and return outputs for Gradio.
    Calls generate_and_upscale_gpu which runs everything in ONE GPU session.
    """
    MAX_CARDS = 10

    video_path, current_seed, upscaled_pairs = generate_and_upscale_gpu(
        input_image, prompt, steps, negative_prompt,
        duration_seconds, guidance_scale, guidance_scale_2,
        seed, randomize_seed, progress,
    )

    slider_outputs = []
    download_outputs = []
    visibility_outputs = []

    for i in range(MAX_CARDS):
        if i < len(upscaled_pairs):
            orig, upscaled = upscaled_pairs[i]
            orig_path = pil_to_temp_path(orig, prefix=f"original_sec{i+1}")
            upscaled_path = pil_to_temp_path(upscaled, prefix=f"upscaled_sec{i+1}")
            slider_outputs.append((orig_path, upscaled_path))
            download_outputs.append(upscaled_path)
            visibility_outputs.append(gr.update(visible=True))
        else:
            slider_outputs.append(None)
            download_outputs.append(None)
            visibility_outputs.append(gr.update(visible=False))

    results = [video_path, current_seed]
    for i in range(MAX_CARDS):
        results.append(slider_outputs[i])
        results.append(download_outputs[i])
        results.append(visibility_outputs[i])

    return results

css = '''
.upscale-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    background: var(--background-fill-secondary);
}
.card-header {
    font-size: 1.1em;
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--body-text-color);
}
'''

MAX_CARDS = 10

with gr.Blocks(title="wan2.2-i2v-fast") as demo:
    gr.Markdown("# **Wan2.2-I2V-Fast**")
    gr.Markdown(
        "Run **Wan 2.2 I2V (14B)** in just 4-8 steps with "
        "[Lightning LoRA](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Wan22-Lightning), "
        "FP8 quantization & AoT compilation — compatible with 🧨 diffusers and ZeroGPU⚡️  \n"
        "**+ FLUX.2 Klein 4B** frame upscaler — automatically extracts 1 frame/sec and upscales to ~1024px."
        " [GitHub ↗](https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image_component = gr.Image(type="pil", label="Input Image", height=300)
            prompt_input = gr.Textbox(
                label="Prompt", value=default_prompt_i2v, lines=3
            )
            duration_seconds_input = gr.Slider(
                minimum=MIN_DURATION, maximum=MAX_DURATION, step=0.1, value=3.5,
                label="Duration (seconds)",
                info=f"Clamped to model's {MIN_FRAMES_MODEL}-{MAX_FRAMES_MODEL} frames at {FIXED_FPS}fps."
            )

            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt", value=default_negative_prompt, lines=3
                )
                seed_input = gr.Slider(
                    label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42,
                    interactive=True
                )
                randomize_seed_checkbox = gr.Checkbox(
                    label="Randomize seed", value=True, interactive=True
                )
                steps_slider = gr.Slider(
                    minimum=1, maximum=30, step=1, value=4,
                    label="Inference Steps"
                )
                guidance_scale_input = gr.Slider(
                    minimum=0.0, maximum=10.0, step=0.5, value=1,
                    label="Guidance Scale - high noise stage"
                )
                guidance_scale_2_input = gr.Slider(
                    minimum=0.0, maximum=10.0, step=0.5, value=1,
                    label="Guidance Scale 2 - low noise stage"
                )

            generate_button = gr.Button("Generate Video & Upscale Frames", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(
                label="Generated Video", autoplay=True, interactive=False
            )

            with gr.Accordion("get_upscaled_samples()", open=False):
            
                gr.Markdown("### Upscaled Frame Comparisons")
                gr.Markdown(
                    "*Drag the slider on each card to compare the **original frame** (left) "
                    "vs the **FLUX.2 Klein 4B upscaled** version (right). "
                    "Click download to save the upscaled image.*"
                )
    
                card_groups = []   # list of (group, slider, download_btn)
                slider_components = []
                download_components = []
                group_components = []
    
                for i in range(MAX_CARDS):
                    with gr.Group(visible=False, elem_classes="upscale-card") as card_group:
                        gr.Markdown(f"**Frame {i+1}** — Second {i+1}", elem_classes="card-header")
                        img_slider = gr.ImageSlider(
                            label=f"Original ↔ Upscaled (Second {i+1})",
                            type="filepath",
                            interactive=True,
                        )
                        download_btn = gr.File(
                            label=f"Download Upscaled Frame {i+1}",
                            interactive=False,
                        )
    
                    slider_components.append(img_slider)
                    download_components.append(download_btn)
                    group_components.append(card_group)

    ui_inputs = [
        input_image_component, prompt_input, steps_slider,
        negative_prompt_input, duration_seconds_input,
        guidance_scale_input, guidance_scale_2_input,
        seed_input, randomize_seed_checkbox,
    ]

    ui_outputs = [video_output, seed_input]
    for i in range(MAX_CARDS):
        ui_outputs.append(slider_components[i])
        ui_outputs.append(download_components[i])
        ui_outputs.append(group_components[i])

    generate_button.click(
        fn=run_pipeline,
        inputs=ui_inputs,
        outputs=ui_outputs,
    )

    gr.Examples(
        examples=[
            [
                "example-file/6b2842cf438d086f556eef05cc29d2d1.jpg",
                "make this image come alive, cinematic motion, smooth animation.",
                4,
            ],
            [
                "example-file/wan_i2v_input.JPG",
                "POV selfie video, white cat with sunglasses standing on surfboard, relaxed smile, tropical beach behind (clear water, green hills, blue sky with clouds). Surfboard tips, cat falls into ocean, camera plunges underwater with bubbles and sunlight beams. Brief underwater view of cat's face, then cat resurfaces, still filming selfie, playful summer vacation mood.",
                4,
            ],
            [
                "example-file/wan22_input_2.jpg",
                "A sleek lunar vehicle glides into view from left to right, kicking up moon dust as astronauts in white spacesuits hop aboard with characteristic lunar bouncing movements. In the distant background, a VTOL craft descends straight down and lands silently on the surface. Throughout the entire scene, ethereal aurora borealis ribbons dance across the star-filled sky, casting shimmering curtains of green, blue, and purple light that bathe the lunar landscape in an otherworldly, magical glow.",
                4,
            ],
            [
                "example-file/kill_bill.jpeg",
                "Uma Thurman's character, Beatrix Kiddo, holds her razor-sharp katana blade steady in the cinematic lighting. Suddenly, the polished steel begins to soften and distort, like heated metal starting to lose its structural integrity. The blade's perfect edge slowly warps and droops, molten steel beginning to flow downward in silvery rivulets while maintaining its metallic sheen. The transformation starts subtly at first - a slight bend in the blade - then accelerates as the metal becomes increasingly fluid. The camera holds steady on her face as her piercing eyes gradually narrow, not with lethal focus, but with confusion and growing alarm as she watches her weapon dissolve before her eyes. Her breathing quickens slightly as she witnesses this impossible transformation. The melting intensifies, the katana's perfect form becoming increasingly abstract, dripping like liquid mercury from her grip. Molten droplets fall to the ground with soft metallic impacts. Her expression shifts from calm readiness to bewilderment and concern as her legendary instrument of vengeance literally liquefies in her hands, leaving her defenseless and disoriented.",
                6,
            ],
        ],
        inputs=[input_image_component, prompt_input, steps_slider],
        outputs=[video_output, seed_input],
        fn=generate_video,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.queue().launch(theme=orange_red_theme, mcp_server=True, css=css, show_error=True)
