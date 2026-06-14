# **wan2.2-i2v-fast**

wan2.2-i2v-fast is an experimental, highly optimized Image-to-Video generation suite powered by the massive 14B parameter `Wan-AI/Wan2.2-I2V-A14B-Diffusers` model. Designed for unprecedented inference speed and VRAM efficiency, this application implements cutting-edge acceleration techniques, including INT8/FP8 dynamic quantization via `torchao`, Ahead-Of-Time Induction (AOTI) compilation for transformer modules, and a specialized Lightning LoRA adapter to produce smooth cinematic animations in as few as 4 to 8 steps.

Beyond core video generation, the pipeline incorporates an automated `Real-ESRGAN` post-processing pass to extract and $2\times$ upscale individual keyframes. The entire suite is wrapped in a bespoke, headless Gradio interface featuring a dark Ubuntu-inspired theme, interactive comparison sliders, and real-time execution logging.

> [!NOTE]
> This app is a preview version 1.0, and more updates are coming soon.

### **Key Features**

* **Ultra-Fast 14B Inference:** Achieves massive speedups by stacking `torchao` weight-only/dynamic activation quantization, AOTI compiled transformer modules, and the `Kijai/WanVideo_comfy` Lightx2v LoRA.
* **Automated Frame Upscaling:** Automatically extracts midpoint frames for each generated second of video and upscales them using Real-ESRGAN x2, providing a rich before-and-after interactive slider comparison.
* **Custom Headless UI:** Abandons standard Gradio blocks for a highly responsive, custom frontend design. It includes a drag-and-drop media zone, execution logs, dynamic resolution scaling, and inline video rendering.
* **Granular Animation Controls:** Fine-tune video length (min 8 frames to max 80 frames), sampling steps, dual-stage guidance scales, and randomized seed injections.
* **Smart Dimension Handling:** Automatically crops and resizes input images to align with the model's strict aspect ratio and token multiple requirements ($832\times480$ / $480\times832$ / $640\times640$).

### **Repository Structure**

```text
├── demo-notebook/
│   └── wan2_2_i2v_fast_uv.ipynb
├── example-file/
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
└── wan2-2_i2v_app.py

```

### **Installation and Requirements**

To run wan2.2-i2v-fast locally, configure a Python environment equipped to handle advanced compilation and heavy model weights. A modern CUDA-enabled GPU is required. This build specifically relies on **PyTorch 2.11.0 and CUDA 13.0** to leverage the latest GPU optimizations, utilizing packages directly from `https://download.pytorch.org/whl/cu130`.

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

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run wan2-2_i2v_app.py

```

---

#### **Standard PIP Installation**

**1. Install Pre-requirements**
Ensure your local system package manager is upgraded:

```bash
pip install pip>=26.1.2

```

**2. Install Core Dependencies**
Install the primary deep learning stack, ensuring that **PyTorch 2.11.0 for CUDA 13.0** is fetched correctly. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

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

Once the FastAPI web deployment initializes, load the dashboard by pointing your browser to the local loopback endpoint (typically `http://127.0.0.1:7860/`).

1. **Upload Reference:** Drop a starting image directly into the designated input pane. The system will automatically resize it for structural compatibility.
2. **Describe Motion:** Enter your animation directions in the prompt box (e.g., *"cinematic motion, smooth animation"*).
3. **Configure Duration:** Adjust the slider to determine the length of the video (from 0.5s up to 5.0s, rendered at 16 FPS).
4. **Advanced Settings (Optional):** Expand the Advanced Settings block to tweak dual-stage Guidance Scales, Inference Steps, and Negative Prompts.
5. **Generate Video:** Click **Generate Video**. Monitor the system logs as the AOTI-compiled pipeline processes the clip, and check the bottom section to interact with the upscaled keyframes once rendering is complete.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast.git](https://github.com/PRITHIVSAKTHIUR/wan2.2-i2v-fast.git)
