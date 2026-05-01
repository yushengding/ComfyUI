"""Qwen-Image-Edit GGUF Q4_K_M A-B-C pants test"""
import json, urllib.request, time, os, sys
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

def queue_wait(wf, timeout=600):
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
        time.sleep(15); e = int(time.time()-start)
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
                        if em: print("  ERROR: "+em[:400])
                return None
            for n in entry.get("outputs",{}).values():
                for img in n.get("images",[]):
                    p=os.path.join(OUTPUT_DIR,img.get("subfolder",""),img["filename"])
                    print("  OK (%ds): %s" % (e,p)); return p
        except: pass
    print("  TIMEOUT"); return None


def build_qwen_edit(img_name, prompt, neg_prompt, prefix, seed):
    """Qwen-Image-Edit GGUF workflow using TextEncodeQwenImageEdit"""
    return {
        # Load GGUF model
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {
            "unet_name": "qwen-image-edit-Q4_K_M.gguf"
        }},
        # Load CLIP (qwen_image type)
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
            "type": "qwen_image"
        }},
        # Load VAE
        "3": {"class_type": "VAELoader", "inputs": {
            "vae_name": "qwen_image_vae.safetensors"
        }},
        # Load source image
        "4": {"class_type": "LoadImage", "inputs": {
            "image": img_name
        }},
        # Positive conditioning: clip + prompt + vae + image
        "10": {"class_type": "TextEncodeQwenImageEdit", "inputs": {
            "clip": ["2", 0],
            "prompt": prompt,
            "vae": ["3", 0],
            "image": ["4", 0]
        }},
        # Negative conditioning: clip + neg prompt only
        "11": {"class_type": "TextEncodeQwenImageEdit", "inputs": {
            "clip": ["2", 0],
            "prompt": neg_prompt
        }},
        # ModelSamplingAuraFlow (sigma adjustment)
        "20": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": ["1", 0],
            "shift": 6.0
        }},
        # Empty latent at same resolution
        "25": {"class_type": "EmptySD3LatentImage", "inputs": {
            "width": 512,
            "height": 768,
            "batch_size": 1
        }},
        # KSampler
        "30": {"class_type": "KSampler", "inputs": {
            "model": ["20", 0],
            "positive": ["10", 0],
            "negative": ["11", 0],
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
            "vae": ["3", 0]
        }},
        # Save
        "50": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": prefix,
            "images": ["40", 0]
        }}
    }


# Prepare A (512x768 to fit VRAM)
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((512, 768), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "qwen_A.png")
src_r.save(A_path)
A_name = upload(A_path)
print("A: " + A_name)

# === A -> B: Add pants ===
print("\n" + "="*60)
print("Qwen-Image-Edit GGUF Q4_K_M: A-B-C Pants Test")
print("="*60)

print("\n=== A -> B: Add pants ==="); sys.stdout.flush()
B = queue_wait(build_qwen_edit(A_name,
    "Make the character wear long black pants",
    "low quality, blurry, deformed",
    "qwen_B", 500))

if B:
    print("\n=== B -> C: Remove pants ==="); sys.stdout.flush()
    B_name = upload(B)
    C = queue_wait(build_qwen_edit(B_name,
        "Remove the black pants, make the character wear a small green speedo swimwear",
        "low quality, blurry, deformed",
        "qwen_C", 600))

    # HTML Report
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Qwen Edit Test</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:30px}.row{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:220px}
.item img{width:100%%;display:block}.item p{padding:12px;text-align:center}
.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:150px}</style></head><body>
<h1>Qwen-Image-Edit GGUF Q4_K_M: Pants A-B-C</h1><div class="row">
<div class="item"><img src="qwen_A.png"><p>A - Original</p></div>"""
    if B:
        html += '\n<div class="arrow">-&gt;</div><div class="item"><img src="%s"><p>B - +Pants</p></div>' % os.path.basename(B)
    if C:
        html += '\n<div class="arrow">-&gt;</div><div class="item"><img src="%s"><p>C - Restored</p></div>' % os.path.basename(C)
    html += "\n</div></body></html>"
    rpt = os.path.join(OUTPUT_DIR, "qwen_edit_report.html")
    with open(rpt, "w", encoding="utf-8") as f: f.write(html)
    print("\nReport: " + rpt)

    # Telegram
    import subprocess
    BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
    CHAT = "755113426"
    API = "https://api.telegram.org/bot" + BOT
    items = [(A_path, "Qwen-Edit A: Original")]
    if B: items.append((B, "Qwen-Edit B: +Pants"))
    if C: items.append((C, "Qwen-Edit C: Restored"))
    for fp, cap in items:
        jpg = fp.replace(".png",".jpg")
        Image.open(fp).save(jpg, "JPEG", quality=90)
        subprocess.run(["curl","-s","-X","POST",API+"/sendPhoto","-F","chat_id="+CHAT,"-F","photo=@"+jpg,"-F","caption="+cap], capture_output=True)
    print("Sent to Telegram!")
    os.startfile(rpt)

print("\nDone!")
