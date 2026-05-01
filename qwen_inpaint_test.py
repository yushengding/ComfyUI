"""Qwen-Image-Edit GGUF + DiffSynth Inpaint ControlNet A-B-C pants test
Uses QwenImageDiffsynthControlnet with mask to only edit the lower body area.
"""
import json, urllib.request, time, os, sys, struct, zlib
from PIL import Image

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"

def upload(fp):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f: d = f.read()
    b = "----U"
    body = ("--"+b+"\r\nContent-Disposition: form-data; name=\"image\"; filename=\""+fn+"\"\r\nContent-Type: image/png\r\n\r\n").encode() + d + ("\r\n--"+b+"\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--"+b+"--\r\n").encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/upload/image", data=body, headers={"Content-Type": "multipart/form-data; boundary="+b})).read())["name"]

def make_mask(w, h, regions):
    """Create a PNG mask with white rectangles on black background"""
    raw = []
    for y in range(h):
        row = b'\x00'  # filter byte
        for x in range(w):
            v = 0
            for (x1, y1, x2, y2) in regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    v = 255; break
            row += bytes([v])
        raw.append(row)
    data = b''.join(raw)
    def chunk(ct, d):
        c = ct + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    return (b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(data, 9)) +
            chunk(b'IEND', b''))

def queue_wait(wf, timeout=1200):
    data = json.dumps({"prompt": wf}).encode()
    try:
        resp = json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/prompt", data=data, headers={"Content-Type":"application/json"})).read())
    except Exception as e:
        try: err = e.read().decode()[:500]
        except: err = str(e)
        print("  SUBMIT ERROR: " + err); return None
    if resp.get("error"):
        print("  ERROR: " + json.dumps(resp)[:500]); return None
    pid = resp["prompt_id"]
    print("  Queued: " + pid); sys.stdout.flush()
    start = time.time()
    while time.time()-start < timeout:
        time.sleep(30); e = int(time.time()-start)
        try:
            hist = json.loads(urllib.request.urlopen(BASE+"/history/"+pid).read())
            entry = hist.get(pid)
            if not entry:
                vram = json.loads(urllib.request.urlopen(BASE+"/system_stats").read())
                free = vram["devices"][0].get("vram_free",0)/1e9
                print("  %ds... (VRAM: %.1fGB)" % (e, free)); sys.stdout.flush(); continue
            if entry.get("status",{}).get("status_str")=="error":
                for m in entry["status"].get("messages",[]):
                    if isinstance(m,list) and len(m)>1 and isinstance(m[1],dict):
                        em=m[1].get("exception_message","")
                        if em: print("  ERROR: "+em[:500])
                return None
            for n in entry.get("outputs",{}).values():
                for img in n.get("images",[]):
                    p=os.path.join(OUTPUT_DIR,img.get("subfolder",""),img["filename"])
                    print("  OK (%ds): %s" % (e,p)); return p
        except: pass
    print("  TIMEOUT"); return None


def build_qwen_inpaint(img_name, mask_name, prompt, neg_prompt, prefix, seed):
    """Qwen-Image-Edit GGUF + DiffSynth Inpaint ControlNet workflow"""
    return {
        # Load GGUF model
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {
            "unet_name": "qwen-image-edit-Q4_K_M.gguf"
        }},
        # Load Inpaint ControlNet patch
        "2": {"class_type": "ModelPatchLoader", "inputs": {
            "name": "qwen_image_inpaint_diffsynth_controlnet.safetensors"
        }},
        # Load CLIP
        "3": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image"
        }},
        # Load VAE
        "4": {"class_type": "VAELoader", "inputs": {
            "vae_name": "qwen_image_vae.safetensors"
        }},
        # Load source image
        "5": {"class_type": "LoadImage", "inputs": {
            "image": img_name
        }},
        # Load mask image
        "6": {"class_type": "LoadImage", "inputs": {
            "image": mask_name
        }},
        # Convert mask image to MASK type
        "7": {"class_type": "ImageToMask", "inputs": {
            "image": ["6", 0],
            "channel": "red"
        }},
        # QwenImageDiffsynthControlnet: apply inpaint control
        # Takes model + model_patch + vae + image + mask → patched MODEL
        "10": {"class_type": "QwenImageDiffsynthControlnet", "inputs": {
            "model": ["1", 0],
            "model_patch": ["2", 0],
            "vae": ["4", 0],
            "image": ["5", 0],
            "strength": 1.0,
            "mask": ["7", 0]
        }},
        # ModelSamplingAuraFlow
        "12": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": ["10", 0],
            "shift": 6.0
        }},
        # Positive conditioning with image context
        "20": {"class_type": "TextEncodeQwenImageEdit", "inputs": {
            "clip": ["3", 0],
            "prompt": prompt,
            "vae": ["4", 0],
            "image": ["5", 0]
        }},
        # Negative conditioning
        "21": {"class_type": "TextEncodeQwenImageEdit", "inputs": {
            "clip": ["3", 0],
            "prompt": neg_prompt
        }},
        # Empty latent
        "25": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": 512,
            "height": 768,
            "batch_size": 1
        }},
        # KSampler
        "30": {"class_type": "KSampler", "inputs": {
            "model": ["12", 0],
            "positive": ["20", 0],
            "negative": ["21", 0],
            "latent_image": ["25", 0],
            "seed": seed,
            "steps": 28,
            "cfg": 3.5,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0
        }},
        # VAE Decode
        "40": {"class_type": "VAEDecode", "inputs": {
            "samples": ["30", 0],
            "vae": ["4", 0]
        }},
        # Save
        "50": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": prefix,
            "images": ["40", 0]
        }}
    }


# Prepare A (512x768)
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((512, 768), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "qinp_A.png")
src_r.save(A_path)
A_name = upload(A_path)

# Create mask for lower body (pants region)
# 512x768 image: lower body roughly y=450-720, x=120-400
mask_data = make_mask(512, 768, [(100, 450, 420, 720)])
mask_path = os.path.join(OUTPUT_DIR, "qinp_mask.png")
with open(mask_path, "wb") as f: f.write(mask_data)
mask_name = upload(mask_path)
print("A: %s, Mask: %s" % (A_name, mask_name))

# === Test ===
print("\n" + "="*60)
print("Qwen-Image-Edit GGUF + Inpaint ControlNet: A-B-C Test")
print("="*60)

# A -> B: Add pants
print("\n=== A -> B: Add pants (masked lower body) ==="); sys.stdout.flush()
B = queue_wait(build_qwen_inpaint(A_name, mask_name,
    "Make the character wear long black pants in the masked area",
    "low quality, blurry, deformed",
    "qinp_B", 500))

if B:
    # B -> C: Remove pants
    print("\n=== B -> C: Remove pants (masked lower body) ==="); sys.stdout.flush()
    B_name = upload(B)
    C = queue_wait(build_qwen_inpaint(B_name, mask_name,
        "Remove the black pants, show bare legs with a small green speedo swimwear in the masked area",
        "low quality, blurry, deformed",
        "qinp_C", 600))

    # HTML Report
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Qwen Inpaint Test</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:10px}h2{color:#8b949e;margin-bottom:30px;font-weight:normal}
.row{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:200px}
.item img{width:100%%;display:block}.item p{padding:12px;text-align:center;font-size:14px}
.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:150px}
</style></head><body>
<h1>Qwen-Image-Edit + Inpaint ControlNet</h1>
<h2>Masked A-B-C Test | 512x768 | 28 steps</h2>
<div class="row">
<div class="item"><img src="qinp_mask.png"><p>Mask</p></div>
<div class="item"><img src="qinp_A.png"><p>A - Original</p></div>"""
    if B:
        html += '\n<div class="arrow">&gt;</div><div class="item"><img src="%s"><p>B - +Pants</p></div>' % os.path.basename(B)
    if C:
        html += '\n<div class="arrow">&gt;</div><div class="item"><img src="%s"><p>C - Restored</p></div>' % os.path.basename(C)
    html += "\n</div></body></html>"
    rpt = os.path.join(OUTPUT_DIR, "qwen_inpaint_report.html")
    with open(rpt, "w", encoding="utf-8") as f: f.write(html)
    print("\nReport: " + rpt)

    # Telegram
    import subprocess
    BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
    CHAT = "755113426"
    API = "https://api.telegram.org/bot" + BOT
    items = [(A_path, "Qwen Inpaint A: Original"), (mask_path, "Mask")]
    if B: items.append((B, "Qwen Inpaint B: +Pants"))
    if C: items.append((C, "Qwen Inpaint C: Restored"))
    for fp, cap in items:
        jpg = fp.replace(".png", ".jpg")
        Image.open(fp).save(jpg, "JPEG", quality=90)
        subprocess.run(["curl","-s","-X","POST",API+"/sendPhoto","-F","chat_id="+CHAT,"-F","photo=@"+jpg,"-F","caption="+cap], capture_output=True)
    print("Sent to Telegram!")
    os.startfile(rpt)

print("\nDone!")
