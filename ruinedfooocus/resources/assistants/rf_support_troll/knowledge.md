# RuinedFooocus support notes

Layout: the top tabs are Main, Image browser, Chat bots and Settings. Main holds the prompt box and Generate button, plus the Generation, Models, Prompt tools, Image tools and Details tabs. The Models tab has two sub-tabs: Model and LoRAs.

Generation tab: Sampling (performance, steps, CFG, sampler and scheduler), Workflow settings, Size and upscale (resolution and upscaling), Style, and Negative prompt and seed. A seed of -1 picks a random seed. Image number 0 generates forever until Cancel is pressed. Ctrl+Enter is the same as pressing Generate.

Models tab, Model sub-tab: choose the image checkpoint. Image model components lists and downloads the text encoder and VAE files a checkpoint needs. Checkpoint tools reports a checkpoint's size and tensor types, checks its weights for damage (NaN/Inf), and can create a smaller FP16 copy. A missing text encoder or VAE, a damaged or incomplete file, or a file that is not an image checkpoint is reported by name when generation fails. Models go in models/checkpoints, LoRAs in models/loras, and chat models (GGUF) in models/llm. The running app sorts finished files in models/inbox into configured folders when their catalogue checksum or header identifies the model category. Online model metadata can also check Civitai by SHA-256. Unknown files, split weights and destination conflicts stay in the inbox.

LoRAs: pick them in Models > LoRAs, or type <lora:filename:0.8> in the prompt. Trigger words are read from a .txt file with the same name as the LoRA, or downloaded from Civitai when available.

Prompt syntax: **name** inserts a random line from wildcards/name.txt. <style:name> applies a style. Put --- on its own line to queue several prompts in one run. Pasting generation JSON into the prompt box reuses all of its settings, and dropping a RuinedFooocus PNG onto the main image restores its prompt and settings.

Prompt tools tab: One Button builds random prompts. Prompt format Auto matches the selected checkpoint: SD/SDXL tags, Illustrious/NoobAI/Pony anime tags, Anima tags with description, or natural language for Flux, Qwen, Krea, MOSS and similar models. Prompt enhancement and Rewrite prompt use the chat model loaded in Chat bots. Rewrite presets are text files in the llamas folder.

Image tools tab (PowerUp): ControlNet, image-to-image, inpainting, background removal (RemBG) and other tools that take an input image. Pick the tool, upload or select the source image, then press Generate. Adding an image with no tool selected picks Native image edit on checkpoints that can edit images (Qwen-Image, Mage-Flow, Fibo2, HiDream O1, MOSS with an edit format, Flux Kontext), otherwise Image to image with a Denoise slider. Switching to a checkpoint without native editing falls back to Image to image, and switching back restores native editing. The same applies to the Chat bots Image reference and to API image edits.

Image browser tab: search and view earlier generations from the outputs folder and reuse their settings.

Chat bots tab: chat with a local GGUF model through llama.cpp. Pick a bot, which sets the system prompt and greeting, or Normal for a plain assistant. Add bots as chatbots/<name>/info.json with name, greeting, system and an avatar.png, or drop a SillyTavern character card PNG into the chatbots folder. A bot's embed list adds reference text from text, file or url sources. Force image generation turns the next message into an image.

Settings tab: Chatbot settings sets the default chat model, the token limit for prompt rewrites and enhancement, and the rewrite reasoning level. llama.cpp backend chooses Vulkan, ROCm, CUDA or CPU for chat models; it is separate from the PyTorch backend used for images. Changes to defaults are saved in the settings folder.

Install: run RuinedFooocus.bat on Windows or ./RuinedFooocus.sh on Linux and WSL. They create the venv folder, install the matching PyTorch build and start the web UI. Python 3.11 to 3.14 and Git are required. The first start downloads several GB of packages and the default SDXL model.

GPU backends: the launcher detects the GPU and installs CUDA builds for NVIDIA, ROCm 10.1 for supported AMD Radeon RX 5000 to 9000 and Strix Halo (Windows, Linux and WSL), DirectML for other AMD or Intel GPUs on Windows, or CPU. Override detection with TORCH_PLATFORM (for example cu128, rocm10.1 or cpu) or the --rocm10, --cuda-nightly, --directml or --cpu flags. There is no config.yaml for this.

Reinstall: create an empty file named reinstall in the RuinedFooocus folder to reinstall all Python packages on the next start, reinstalltorch to reinstall PyTorch, or freezetorch to stop the launcher from changing PyTorch. Deleting the venv folder and starting again gives a completely fresh install.

Launch options: --listen opens the UI on the local network and needs --auth user/password. --port changes the port, --nobrowser skips opening the browser, and --lowvram, --novram or --cpu reduce GPU memory use. --fp32-vae or --cpu-vae can fix black or NaN images caused by VAE precision.

Common errors: MetadataIncompleteBuffer means a model file is corrupt or only partly downloaded; download it again. Out of memory errors: lower the resolution or batch size, close other GPU programs, or start with --lowvram. Black images usually come from VAE precision (try --fp32-vae) or a model that needs a different VAE. Always read the console window: the real error is printed there.

Chat model problems: if a chat bot gives no answer, check that a GGUF model is selected and loaded in Chat bots, and try another llama.cpp backend in Settings, for example Vulkan instead of ROCm. Large models need enough VRAM or system RAM; pick a smaller quantization (Q4) when memory is short.
