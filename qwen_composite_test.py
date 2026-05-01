"""Qwen-Image-Edit GGUF + mask composite A-B-C pants test
Strategy: Use Qwen-Edit for full-image editing, then composite only the
masked region back onto the original to preserve non-edited areas.
"""
import json, urllib.request, time, os, sys
from PIL import Image, ImageFilter
import numpy as np

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"

def upload(fp):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f: d = f.read()
    b = "----U"
    body = ("--"+b+"\r\nContent-Disposition: form-data; name=\"image\"; filename=\""+fn+"\"\r\nContent-Type: image/png\r\n\r\n").encode() + d + ("\r\n--"+b+"\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--"+b+"--\r\n").encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(BASE+"/upload/image", data=body, headers={"Content-Type": "multipart/form-data; boundary="+b})).read())["name"]

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


def build_qwen_edit(img_name, prompt, neg_prompt, prefix, seed):
    """Qwen-Image-Edit GGUF workflow (full image edit, no ControlNet)"""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-Q4_K_M.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": img_name}},
        "10": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image": ["4", 0]}},
        "11": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["2", 0], "prompt": neg_prompt}},
        "20": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 6.0}},
        "25": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 512, "height": 768, "batch_size": 1}},
        "30": {"class_type": "KSampler", "inputs": {"model": ["20", 0], "positive": ["10", 0], "negative": ["11", 0], "latent_image": ["25", 0], "seed": seed, "steps": 28, "cfg": 3.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}},
        "50": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["40", 0]}}
    }


def composite_with_mask(original_path, edited_path, output_path, mask_region, feather=20):
    """Composite edited region onto original using a feathered mask.
    mask_region: (x1, y1, x2, y2) rectangle defining the edit area.
    feather: pixels of soft blending at the edges.
    """
    orig = Image.open(original_path).convert("RGB")
    edited = Image.open(edited_path).convert("RGB").resize(orig.size, Image.LANCZOS)

    # Create mask: white in edit region, black outside
    mask = Image.new("L", orig.size, 0)
    x1, y1, x2, y2 = mask_region
    for y in range(y1, min(y2+1, orig.size[1])):
        for x in range(x1, min(x2+1, orig.size[0])):
            mask.putpixel((x, y), 255)

    # Feather the mask edges
    mask = mask.filter(ImageFilter.GaussianBlur(feather))

    # Composite: use edited in mask region, original outside
    result = Image.composite(edited, orig, mask)
    result.save(output_path)
    print("  Composited: " + output_path)
    return output_path


# Prepare A (512x768)
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((512, 768), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "qcomp_A.png")
src_r.save(A_path)
A_name = upload(A_path)
print("A: " + A_name)

# Mask region for lower body (in 512x768 coords)
MASK_REGION = (100, 450, 420, 720)

print("\n" + "="*60)
print("Qwen-Image-Edit + Mask Composite: A-B-C Test")
print("="*60)

# A -> B_raw (full edit) -> B (composite)
print("\n=== A -> B: Add pants ==="); sys.stdout.flush()
B_raw = queue_wait(build_qwen_edit(A_name,
    "Make the character wear long black pants",
    "low quality, blurry, deformed",
    "qcomp_Braw", 500))

if B_raw:
    B_path = os.path.join(OUTPUT_DIR, "qcomp_B.png")
    composite_with_mask(A_path, B_raw, B_path, MASK_REGION, feather=25)

    # B -> C_raw (full edit) -> C (composite)
    print("\n=== B -> C: Remove pants ==="); sys.stdout.flush()
    B_name = upload(B_path)
    C_raw = queue_wait(build_qwen_edit(B_name,
        "Remove the black pants, make the character wear a small green speedo swimwear",
        "low quality, blurry, deformed",
        "qcomp_Craw", 600))

    if C_raw:
        C_path = os.path.join(OUTPUT_DIR, "qcomp_C.png")
        composite_with_mask(B_path, C_raw, C_path, MASK_REGION, feather=25)

    # HTML Report
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Qwen Composite Test</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:10px}h2{color:#8b949e;margin-bottom:30px;font-weight:normal}
.row{display:flex;gap:15px;flex-wrap:wrap;justify-content:center;margin-bottom:30px}
.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:180px}
.item img{width:100%%;display:block}.item p{padding:10px;text-align:center;font-size:13px}
.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:120px}
h3{color:#8b949e;text-align:center;margin:20px 0 10px}
</style></head><body>
<h1>Qwen-Image-Edit + Mask Composite</h1>
<h2>Only edit masked region, preserve rest | 512x768 | 28 steps</h2>
<h3>Final (composited)</h3>
<div class="row">
<div class="item"><img src="qcomp_A.png"><p>A - Original</p></div>"""
    if B_raw:
        html += '\n<div class="arrow">&gt;</div><div class="item"><img src="qcomp_B.png"><p>B - +Pants</p></div>'
    if C_raw:
        html += '\n<div class="arrow">&gt;</div><div class="item"><img src="qcomp_C.png"><p>C - Restored</p></div>'
    html += '\n</div>\n<h3>Raw model output (before composite)</h3>\n<div class="row">'
    html += '\n<div class="item"><img src="qcomp_A.png"><p>A - Original</p></div>'
    if B_raw:
        html += '\n<div class="arrow">&gt;</div><div class="item"><img src="%s"><p>B raw</p></div>' % os.path.basename(B_raw)
    if C_raw:
        html += '\n<div class="arrow">&gt;</div><div class="item"><img src="%s"><p>C raw</p></div>' % os.path.basename(C_raw)
    html += "\n</div></body></html>"
    rpt = os.path.join(OUTPUT_DIR, "qwen_composite_report.html")
    with open(rpt, "w", encoding="utf-8") as f: f.write(html)
    print("\nReport: " + rpt)

    # Telegram
    import subprocess
    BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
    CHAT = "755113426"
    API = "https://api.telegram.org/bot" + BOT
    items = [(A_path, "Composite A: Original")]
    if B_raw:
        items.append((B_path, "Composite B: +Pants (masked)"))
    if C_raw:
        items.append((C_path, "Composite C: Restored (masked)"))
    for fp, cap in items:
        jpg = fp.replace(".png", ".jpg")
        Image.open(fp).save(jpg, "JPEG", quality=90)
        subprocess.run(["curl","-s","-X","POST",API+"/sendPhoto","-F","chat_id="+CHAT,"-F","photo=@"+jpg,"-F","caption="+cap], capture_output=True)
    print("Sent to Telegram!")
    os.startfile(rpt)

print("\nDone!")
