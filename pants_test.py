"""Pants add/remove A-B-C test"""
import json, urllib.request, time, os, sys, struct, zlib
from PIL import Image

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"
NEG = "low quality, worst quality, blurry, bad anatomy, deformed, ugly, extra limbs, missing limbs, watermark, text, extra fingers"

def upload(fp):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f: d = f.read()
    b = "----U"
    body = f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fn}\"\r\nContent-Type: image/png\r\n\r\n".encode() + d + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{b}--\r\n".encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(f"{BASE}/upload/image", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})).read())["name"]

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

def queue_wait(wf, timeout=300):
    data = json.dumps({"prompt": wf}).encode()
    resp = json.loads(urllib.request.urlopen(urllib.request.Request(f"{BASE}/prompt", data=data, headers={"Content-Type":"application/json"})).read())
    if resp.get("error"):
        print(f"  ERROR: {json.dumps(resp)[:500]}"); return None
    pid = resp["prompt_id"]
    print(f"  Queued: {pid}"); sys.stdout.flush()
    start = time.time()
    while time.time()-start < timeout:
        time.sleep(10); e = int(time.time()-start)
        try:
            hist = json.loads(urllib.request.urlopen(f"{BASE}/history/{pid}").read())
            entry = hist.get(pid)
            if not entry: print(f"  {e}s..."); sys.stdout.flush(); continue
            if entry.get("status",{}).get("status_str")=="error":
                for m in entry["status"].get("messages",[]):
                    if isinstance(m,list) and len(m)>1 and isinstance(m[1],dict):
                        em=m[1].get("exception_message","")
                        if em: print(f"  ERROR: {em[:300]}")
                return None
            for n in entry.get("outputs",{}).values():
                for img in n.get("images",[]):
                    p=os.path.join(OUTPUT_DIR,img.get("subfolder",""),img["filename"])
                    print(f"  OK ({e}s): {p}"); return p
        except: pass
    print("  TIMEOUT"); return None

def build_wf(src, mask, prompt, prefix, seed):
    return {
        "1": {"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"animagine-xl-3.1.safetensors"}},
        "2": {"class_type":"LoadImage","inputs":{"image":src}},
        "6": {"class_type":"LoadImage","inputs":{"image":mask}},
        "7": {"class_type":"ImageToMask","inputs":{"image":["6",0],"channel":"red"}},
        "8": {"class_type":"INPAINT_InpaintWithModel","inputs":{"inpaint_model":["9",0],"image":["2",0],"mask":["7",0],"seed":seed}},
        "9": {"class_type":"INPAINT_LoadInpaintModel","inputs":{"model_name":"big-lama.pt"}},
        "10": {"class_type":"INPAINT_LoadFooocusInpaint","inputs":{"head":"fooocus_inpaint_head.pth","patch":"inpaint_v26.fooocus.patch"}},
        "11": {"class_type":"INPAINT_ApplyFooocusInpaint","inputs":{"model":["1",0],"patch":["10",0],"latent":["15",0]}},
        "14": {"class_type":"VAEEncode","inputs":{"pixels":["8",0],"vae":["1",2]}},
        "15": {"class_type":"SetLatentNoiseMask","inputs":{"samples":["14",0],"mask":["7",0]}},
        "20": {"class_type":"CLIPTextEncode","inputs":{"clip":["1",1],"text":prompt}},
        "21": {"class_type":"CLIPTextEncode","inputs":{"clip":["1",1],"text":NEG}},
        "30": {"class_type":"KSampler","inputs":{"model":["11",0],"positive":["20",0],"negative":["21",0],"latent_image":["15",0],"seed":seed,"steps":30,"cfg":7.0,"sampler_name":"dpmpp_2m","scheduler":"karras","denoise":1.0}},
        "40": {"class_type":"VAEDecode","inputs":{"samples":["30",0],"vae":["1",2]}},
        "50": {"class_type":"SaveImage","inputs":{"filename_prefix":prefix,"images":["40",0]}}
    }

# Prepare A
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((768, 1152), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "pants_A.png")
src_r.save(A_path)
A_name = upload(A_path)
print(f"A uploaded: {A_name}")

# Mask: lower body y=700~1100, x=150~620
mask_data = make_mask(768, 1152, [(150, 700, 620, 1100)])
mask_path = os.path.join(OUTPUT_DIR, "mask_pants.png")
with open(mask_path, "wb") as f: f.write(mask_data)
mask_name = upload(mask_path)

# A -> B: add pants
print("\n=== A -> B: Add black pants ==="); sys.stdout.flush()
B = queue_wait(build_wf(A_name, mask_name,
    "1boy, wearing long black pants, dark trousers, belt, anime style, detailed, masterpiece, best quality",
    "pants_B", seed=500))

if B:
    # B -> C: remove pants
    print("\n=== B -> C: Remove pants, restore original ==="); sys.stdout.flush()
    B_name = upload(B)
    C = queue_wait(build_wf(B_name, mask_name,
        "1boy, bare legs, green speedo swimwear, muscular thighs, anime style, detailed, masterpiece, best quality",
        "pants_C", seed=600))

    if C:
        # Report
        html = '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Pants A-B-C</title>\n'
        html += '<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}'
        html += 'h1{color:#fff;margin-bottom:30px}.row{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}'
        html += '.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:260px}'
        html += '.item img{width:100%;display:block}.item p{padding:12px;text-align:center;font-size:0.95rem}'
        html += '.arrow{display:flex;align-items:center;font-size:2.5rem;color:#3fb950;padding-top:200px}</style></head><body>\n'
        html += '<h1>Pants Test: A vs B vs C</h1><div class="row">\n'
        html += '<div class="item"><img src="pants_A.png"><p>A - Original</p></div>\n'
        html += '<div class="arrow">-></div>\n'
        html += f'<div class="item"><img src="{os.path.basename(B)}"><p>B - +Pants</p></div>\n'
        html += '<div class="arrow">-></div>\n'
        html += f'<div class="item"><img src="{os.path.basename(C)}"><p>C - Restored</p></div>\n'
        html += '</div></body></html>'
        rpt = os.path.join(OUTPUT_DIR, "pants_report.html")
        with open(rpt, "w", encoding="utf-8") as f: f.write(html)
        print(f"\nReport: {rpt}")

        # Telegram
        import subprocess
        BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
        CHAT = "755113426"
        API = f"https://api.telegram.org/bot{BOT}"
        for fp, cap in [(A_path, "A: Original"), (B, "B: +Pants"), (C, "C: -Pants Restored")]:
            jpg = fp.replace(".png", ".jpg")
            Image.open(fp).save(jpg, "JPEG", quality=90)
            subprocess.run(["curl", "-s", "-X", "POST", f"{API}/sendPhoto",
                "-F", f"chat_id={CHAT}", "-F", f"photo=@{jpg}", "-F", f"caption={cap}"],
                capture_output=True)
        print("Sent to Telegram!")

os.startfile(os.path.join(OUTPUT_DIR, "pants_report.html"))
