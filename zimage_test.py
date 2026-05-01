"""Z-Image-Turbo inpainting pants A-B-C test"""
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
    raw = []
    for y in range(h):
        row = b'\x00'
        for x in range(w):
            v = 0
            for (x1,y1,x2,y2) in regions:
                if x1<=x<=x2 and y1<=y<=y2: v=255; break
            row += bytes([v])
        raw.append(row)
    data = b''.join(raw)
    def chunk(ct, d):
        c = ct+d; return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(data,9))+chunk(b'IEND',b'')

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

def build_zimage_inpaint(src_name, mask_name, prompt, prefix, seed):
    """Z-Image-Turbo inpainting workflow"""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": src_name}},
        "6": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "7": {"class_type": "ImageToMask", "inputs": {"image": ["6", 0], "channel": "red"}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "low quality, blurry, deformed"}},
        "15": {"class_type": "InpaintModelConditioning", "inputs": {"positive": ["10", 0], "negative": ["11", 0], "vae": ["3", 0], "pixels": ["4", 0], "mask": ["7", 0], "noise_mask": True}},
        "30": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["15", 0], "negative": ["15", 1], "latent_image": ["15", 2], "seed": seed, "steps": 8, "cfg": 1.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "40": {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}},
        "50": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["40", 0]}}
    }

# Prepare A (768x1152)
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((768, 1152), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "zimg_A.png")
src_r.save(A_path)
A_name = upload(A_path)

# Mask lower body
mask_data = make_mask(768, 1152, [(150, 700, 620, 1100)])
mask_path = os.path.join(OUTPUT_DIR, "zimg_mask_pants.png")
with open(mask_path, "wb") as f: f.write(mask_data)
mask_name = upload(mask_path)
print("A: %s, Mask: %s" % (A_name, mask_name))

# A -> B: add pants
print("\n=== Z-Image-Turbo: A -> B (add pants) ==="); sys.stdout.flush()
B = queue_wait(build_zimage_inpaint(A_name, mask_name,
    "anime character wearing long black pants, dark trousers, detailed, high quality",
    "zimg_B", 500))

if B:
    print("\n=== Z-Image-Turbo: B -> C (remove pants) ==="); sys.stdout.flush()
    B_name = upload(B)
    C = queue_wait(build_zimage_inpaint(B_name, mask_name,
        "anime character bare legs, small green speedo swimwear, muscular thighs, detailed, high quality",
        "zimg_C", 600))

    # Report + Telegram
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Z-Image Test</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:30px}.row{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:220px}
.item img{width:100%%;display:block}.item p{padding:12px;text-align:center}
.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:150px}</style></head><body>
<h1>Z-Image-Turbo: Pants A-B-C</h1><div class="row">
<div class="item"><img src="zimg_A.png"><p>A - Original</p></div>"""
    if B: html += '\n<div class="arrow">-&gt;</div><div class="item"><img src="%s"><p>B - +Pants</p></div>' % os.path.basename(B)
    if C: html += '\n<div class="arrow">-&gt;</div><div class="item"><img src="%s"><p>C - Restored</p></div>' % os.path.basename(C)
    html += "\n</div></body></html>"
    rpt = os.path.join(OUTPUT_DIR, "zimage_report.html")
    with open(rpt, "w", encoding="utf-8") as f: f.write(html)

    import subprocess
    BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
    CHAT = "755113426"
    API = "https://api.telegram.org/bot" + BOT
    items = [(A_path, "Z-Image A: Original")]
    if B: items.append((B, "Z-Image B: +Pants"))
    if C: items.append((C, "Z-Image C: Restored"))
    for fp, cap in items:
        jpg = fp.replace(".png",".jpg")
        Image.open(fp).save(jpg, "JPEG", quality=90)
        subprocess.run(["curl","-s","-X","POST",API+"/sendPhoto","-F","chat_id="+CHAT,"-F","photo=@"+jpg,"-F","caption="+cap], capture_output=True)
    print("Sent to Telegram!")
    os.startfile(rpt)

print("\nDone!")
