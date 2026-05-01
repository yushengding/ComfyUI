"""ICEdit pants A->B->C test"""
import json, urllib.request, time, os, sys
from PIL import Image

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"

def upload(fp):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f: d = f.read()
    b = "----U"
    body = f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n".encode() + d + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{b}--\r\n".encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(f"{BASE}/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})).read())["name"]

def queue_wait(wf, timeout=600):
    data = json.dumps({"prompt": wf}).encode()
    try:
        resp = json.loads(urllib.request.urlopen(urllib.request.Request(f"{BASE}/prompt", data=data, headers={"Content-Type":"application/json"})).read())
    except Exception as e:
        print(f"  SUBMIT ERROR: {e}"); return None
    if resp.get("error"):
        print(f"  ERROR: {json.dumps(resp)[:800]}"); return None
    pid = resp["prompt_id"]
    print(f"  Queued: {pid}"); sys.stdout.flush()
    start = time.time()
    while time.time()-start < timeout:
        time.sleep(15); e = int(time.time()-start)
        try:
            hist = json.loads(urllib.request.urlopen(f"{BASE}/history/{pid}").read())
            entry = hist.get(pid)
            if not entry:
                vram = json.loads(urllib.request.urlopen(f"{BASE}/system_stats").read())
                free = vram["devices"][0].get("vram_free",0)/1e9
                print(f"  {e}s... (VRAM free: {free:.1f}GB)"); sys.stdout.flush(); continue
            if entry.get("status",{}).get("status_str")=="error":
                for m in entry["status"].get("messages",[]):
                    if isinstance(m,list) and len(m)>1 and isinstance(m[1],dict):
                        em=m[1].get("exception_message","")
                        if em: print(f"  ERROR: {em[:400]}")
                return None
            for n in entry.get("outputs",{}).values():
                for img in n.get("images",[]):
                    p=os.path.join(OUTPUT_DIR,img.get("subfolder",""),img["filename"])
                    print(f"  OK ({e}s): {p}"); return p
        except: pass
    print("  TIMEOUT"); return None

def build_icedit_wf(image_name, instruction, prefix, seed=42):
    """ICEdit workflow: LoadImage -> DiptychCreate -> ICEFConditioning -> KSampler -> crop right half"""
    return {
        # Load FLUX Fill model
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux1-fill-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        # Load LoRA
        "2": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": "pytorch_lora_weights.safetensors", "strength_model": 1.0}},
        # CLIP
        "3": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux"}},
        # VAE
        "4": {"class_type": "VAELoader",
              "inputs": {"vae_name": "ae.safetensors"}},
        # Load source image
        "10": {"class_type": "LoadImage",
               "inputs": {"image": image_name}},
        # Create diptych (side-by-side: original | blank for editing)
        "11": {"class_type": "DiptychCreate",
               "inputs": {"image": ["10", 0]}},
        # ICEdit prompt - the diptych format instruction
        "20": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["3", 0],
                          "text": f"A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}."}},
        # Negative
        "21": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["3", 0], "text": ""}},
        # ICEFConditioning
        "25": {"class_type": "ICEFConditioning",
               "inputs": {"In_context": ["20", 0], "negative": ["21", 0],
                          "vae": ["4", 0], "diptych": ["11", 0], "maskDiptych": ["11", 1]}},
        # KSampler
        "30": {"class_type": "KSampler",
               "inputs": {"model": ["2", 0], "positive": ["25", 0], "negative": ["25", 1],
                          "latent_image": ["25", 2], "seed": seed, "steps": 28, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        # Decode
        "40": {"class_type": "VAEDecode",
               "inputs": {"samples": ["30", 0], "vae": ["4", 0]}},
        # Crop right half (the edited image)
        "45": {"class_type": "ImageCrop",
               "inputs": {"image": ["40", 0], "width": 512, "height": 768, "x": 512, "y": 0}},
        # Save
        "50": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": prefix, "images": ["45", 0]}}
    }

# Resize source to 512x768 (ICEdit requires 512px width)
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((512, 768), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "icedit_A.png")
src_r.save(A_path)
A_name = upload(A_path)
print(f"A uploaded: {A_name}")

# A -> B: add pants
print("\n=== A -> B: Add black pants ==="); sys.stdout.flush()
B = queue_wait(build_icedit_wf(A_name,
    "the character is wearing long black pants instead of the green swimwear",
    "icedit_B", seed=500), timeout=600)

if B:
    # B -> C: remove pants
    print("\n=== B -> C: Remove pants ==="); sys.stdout.flush()
    B_name = upload(B)
    C = queue_wait(build_icedit_wf(B_name,
        "the character is wearing green speedo swimwear instead of the black pants",
        "icedit_C", seed=600), timeout=600)

    if C:
        # Report + Telegram
        html = '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>ICEdit Test</title>\n'
        html += '<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}'
        html += 'h1{color:#fff;margin-bottom:30px}.row{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}'
        html += '.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:220px}'
        html += '.item img{width:100%;display:block}.item p{padding:12px;text-align:center}'
        html += '.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:150px}</style></head><body>\n'
        html += '<h1>ICEdit: Pants A-B-C Test</h1><div class="row">\n'
        html += f'<div class="item"><img src="icedit_A.png"><p>A - Original</p></div>\n'
        html += f'<div class="arrow">-></div><div class="item"><img src="{os.path.basename(B)}"><p>B - +Pants</p></div>\n'
        html += f'<div class="arrow">-></div><div class="item"><img src="{os.path.basename(C)}"><p>C - Restored</p></div>\n'
        html += '</div></body></html>'
        with open(os.path.join(OUTPUT_DIR, "icedit_report.html"), "w", encoding="utf-8") as f: f.write(html)

        import subprocess
        BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
        CHAT = "755113426"
        API = f"https://api.telegram.org/bot{BOT}"
        for fp, cap in [(A_path,"ICEdit A: Original"), (B,"ICEdit B: +Pants"), (C,"ICEdit C: Restored")]:
            jpg = fp.replace(".png",".jpg")
            Image.open(fp).save(jpg, "JPEG", quality=90)
            subprocess.run(["curl","-s","-X","POST",f"{API}/sendPhoto","-F",f"chat_id={CHAT}","-F",f"photo=@{jpg}","-F",f"caption={cap}"], capture_output=True)
        print("Sent to Telegram!")

print("\nDone!")
os.startfile(os.path.join(OUTPUT_DIR, "icedit_report.html"))
