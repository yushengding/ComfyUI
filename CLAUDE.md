# ComfyUI Project

## Quick Reference

- **Server**: `python main.py --listen 0.0.0.0 --port 8188` | Check: `curl -s http://127.0.0.1:8188/system_stats`
- **GPU**: RTX 4080 (16GB VRAM)
- **Skill**: Use `/comfyui` skill for full operation manual (API usage, workflow templates, model list, auto-load patterns)
- **Memory**: See `project_comfyui_models.md` for model inventory, `reference_civitai_workflow.md` for CivitAI API patterns

## Available Pipelines

| Pipeline | Checkpoint/UNet | Use Case |
|----------|----------------|----------|
| FLUX txt2img | `flux1-dev-fp8-e4m3fn.safetensors` (UNETLoader + DualCLIPLoader) | High-quality generation |
| FLUX Kontext | `flux1-dev-kontext_fp8_scaled.safetensors` | Text-guided image editing |
| FLUX Fill | `flux1-fill-dev-fp8.safetensors` | Inpainting/outpainting |
| SD1.5 + LoRA | `dreamshaper_8.safetensors` (CheckpointLoaderSimple) | LoRA-based generation (512x768) |
| SDXL + LoRA | `animagine-xl-3.1.safetensors` (CheckpointLoaderSimple) | XL LoRA generation (768x1024) |
| Z-Image-Turbo | `z_image_turbo_bf16.safetensors` (UNETLoader) | 4-step fast generation |

## Key Rules

- **交付即可用**: 创建工作流后必须自动加载到浏览器（iframe + `app.loadGraphData` 注入），不要让用户手动拖文件
- **Auto-load pattern**: Create HTML in frontend static dir (`comfyui_frontend_package/static/`), serve from same origin `http://127.0.0.1:8188/<page>.html`
- **CivitAI API key**: User has one; pass as `?token=<key>` when downloading models
- **Sampler names**: CivitAI display names ≠ ComfyUI API names (e.g. "DPM++ 2M Karras" → `dpmpp_2m` + `karras`)
