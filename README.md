# **[Wan2.2-I2V-Fast](https://huggingface.co/spaces/prithivMLmods/wan2.2-i2v-fast)**

Wan2.2-I2V-Fast is a highly optimized, experimental Image-to-Video generation pipeline powered by the massive 14B parameter `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model. Engineered for rapid inference and VRAM efficiency, this suite utilizes cutting-edge acceleration techniques including INT8/FP8 dynamic quantization via `torchao`, Ahead-Of-Time Induction (AOTI) compilation for transformer modules, and a specialized Lightning LoRA adapter to render cinematic animations in just 4 to 8 steps.

Version 2.0 introduces a powerful new frame upscaling feature: the system automatically extracts one frame per second from the generated video and upscales them using the `FLUX.2-klein-4B` model. The suite is wrapped in an interactive Gradio interface featuring dynamic comparison sliders to view the original versus upscaled keyframes.

🤗 huggingface [demo] — [hf.co/spaces/prithivmlmods/wan2.2-i2v-fast](https://huggingface.co/spaces/prithivMLmods/wan2.2-i2v-fast)

> [!NOTE]
> This app is a preview version 1.0, and more updates are coming soon.

### **Key Features**

* **Ultra-Fast 14B Inference:** Achieves massive speedups by stacking `torchao` weight-only/dynamic activation quantization, AOTI compiled transformer modules, and the `Kijai/WanVideo_comfy` Lightx2v LoRA.
* **FLUX.2 Klein Frame Upscaling:** Automatically extracts midpoint frames for each generated second of video and upscales them using `FLUX.2-klein-4B`, providing a rich before-and-after interactive slider comparison.
* **Granular Animation Controls:** Fine-tune video length (rendered at 16 FPS), sampling steps, dual-stage guidance scales, and randomized seed injections.
* **Smart Dimension Handling:** Automatically crops and resizes input images to align with the model's strict aspect ratio and token multiple requirements ($832\times480$, $480\times832$, or $640\times640$).
* **Optimized Hardware Allocation:** Employs dynamic memory management via PyTorch expandable segments to successfully run massive transformer blocks inside active ZeroGPU environments.

### **Repository Structure**

```text
├── demo-notebook/
│   └── wan2_2_i2v_fast_uv.ipynb
├── example-file/
│   ├── 6b2842cf438d086f556eef05cc29d2d1.jpg
│   ├── kill_bill.jpeg
│   ├── wan_i2v_input.JPG
│   └── wan22_input_2.jpg
├── aoti.py
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
├── uv.lock
└── wan_app.py

```

### **Installation and Requirements**

To run Wan2.2-I2V-Fast locally, configure a Python environment equipped to handle advanced compilation and heavy model weights. A modern CUDA-enabled GPU is required.

This repository specifically relies on **PyTorch 2.11.0 and CUDA 13.0** (`--extra-index-url https://download.pytorch.org/whl/cu130`).

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, ensuring rapid virtual environment synchronization and reproducible execution.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast.git
cd wan2.2-i2v-fast

```

**Step 3 — Initialize the project and install dependencies**
This will automatically parse the `uv.lock` and `requirements.txt` to fetch the correct PyTorch 2.11.0 + cu130 wheels.

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run wan_app.py

```

---

#### **Standard PIP Installation**

**1. Install Pre-requirements**
Ensure your local system package manager is upgraded:

```bash
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, diffusion utilities, and ecosystem structures. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/diffusers.git
git+https://github.com/huggingface/peft.git
--extra-index-url https://download.pytorch.org/whl/cu130
gradio[oauth,mcp]==6.16.0
huggingface_hub
spaces==0.50.4
imageio-ffmpeg
torch==2.11.0
opencv-python
sentencepiece
torchvision
torchaudio
omegaconf
termcolor
pydantic
torchao
fastapi
kernels
imageio
hf_xet
pyyaml
pillow
numpy
ftfy
av

```

### **Usage**

Once the Gradio application initializes, load the dashboard by pointing your browser to the local loopback endpoint (typically `http://127.0.0.1:7860/`).

1. **Upload Reference:** Drop a starting image directly into the designated input pane. The system will automatically resize it for structural compatibility.
2. **Describe Motion:** Enter your animation directions in the prompt box.
3. **Configure Duration:** Adjust the slider to determine the length of the video (from 0.5s up to 5.0s, rendered at 16 FPS).
4. **Advanced Settings (Optional):** Expand the Advanced Settings block to tweak dual-stage Guidance Scales, Inference Steps (default is 4), and Negative Prompts.
5. **Generate:** Click **Generate Video & Upscale Frames**.
6. **Review Upscales:** After the video renders, expand the `get_upscaled_samples()` accordion to use the interactive sliders, comparing your raw video frames against the FLUX.2 Klein 4B enhanced frames.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast.git](https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast.git)
