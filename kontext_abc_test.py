"""FLUX Kontext pants A-B-C test"""
import json, urllib.request, time, os, sys
from PIL import Image

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"

def upload(fp):
    fn = os.path.basename(fp)
    with open(fp, "rb") as f: d = f.read()
    b = "----U"
    body = ("--" + b + "\r\nContent-Disposition: form-data; name=\"image\"; filename=\"" + fn + "\"\r\nContent-Type: image/png\r\n\r\n").encode() + d + ("\r\n--" + b + "\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--" + b + "--\r\n").encode()
    return json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/upload/image", data=body, headers={"Content-Type": "multipart/form-data; boundary=" + b})).read())["name"]

def queue_wait(wf, timeout=600):
    data = json.dumps({"prompt": wf}).encode()
    resp = json.loads(urllib.request.urlopen(urllib.request.Request(BASE + "/prompt", data=data, headers={"Content-Type":"application/json"})).read())
    if resp.get("error"):
        print("  ERROR: " + json.dumps(resp)[:500]); return None
    pid = resp["prompt_id"]
    print("  Queued: " + pid); sys.stdout.flush()
    start = time.time()
    while time.time()-start < timeout:
        time.sleep(15)
        e = int(time.time()-start)
        try:
            hist = json.loads(urllib.request.urlopen(BASE + "/history/" + pid).read())
            entry = hist.get(pid)
            if not entry:
                vram = json.loads(urllib.request.urlopen(BASE + "/system_stats").read())
                free = vram["devices"][0].get("vram_free",0)/1e9
                print("  %ds... (VRAM: %.1fGB free)" % (e, free)); sys.stdout.flush(); continue
            if entry.get("status",{}).get("status_str")=="error":
                for m in entry["status"].get("messages",[]):
                    if isinstance(m,list) and len(m)>1 and isinstance(m[1],dict):
                        em=m[1].get("exception_message","")
                        if em: print("  ERROR: " + em[:300])
                return None
            for n in entry.get("outputs",{}).values():
                for img in n.get("images",[]):
                    p = os.path.join(OUTPUT_DIR, img.get("subfolder",""), img["filename"])
                    print("  OK (%ds): %s" % (e, p)); return p
        except: pass
    print("  TIMEOUT"); return None

def build_kontext(img_name, instruction, prefix, seed):
    return {
        "1": {"class_type":"UNETLoader","inputs":{"unet_name":"flux1-dev-kontext_fp8_scaled.safetensors","weight_dtype":"fp8_e4m3fn"}},
        "2": {"class_type":"DualCLIPLoader","inputs":{"clip_name1":"clip_l.safetensors","clip_name2":"t5xxl_fp8_e4m3fn_scaled.safetensors","type":"flux"}},
        "3": {"class_type":"VAELoader","inputs":{"vae_name":"ae.safetensors"}},
        "4": {"class_type":"LoadImage","inputs":{"image":img_name}},
        "5": {"class_type":"ImageScale","inputs":{"image":["4",0],"upscale_method":"lanczos","width":512,"height":768,"crop":"center"}},
        "6": {"class_type":"VAEEncode","inputs":{"pixels":["5",0],"vae":["3",0]}},
        "10": {"class_type":"CLIPTextEncodeFlux","inputs":{"clip":["2",0],"clip_l":instruction,"t5xxl":instruction,"guidance":30.0}},
        "11": {"class_type":"ReferenceLatent","inputs":{"conditioning":["10",0],"latent":["6",0]}},
        "20": {"class_type":"EmptySD3LatentImage","inputs":{"width":512,"height":768,"batch_size":1}},
        "30": {"class_type":"KSampler","inputs":{"model":["1",0],"positive":["11",0],"negative":["10",0],"latent_image":["20",0],"seed":seed,"steps":12,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","denoise":1.0}},
        "40": {"class_type":"VAEDecode","inputs":{"samples":["30",0],"vae":["3",0]}},
        "50": {"class_type":"SaveImage","inputs":{"filename_prefix":prefix,"images":["40",0]}}
    }

# A
src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
src_r = src.resize((512, 768), Image.LANCZOS)
A_path = os.path.join(OUTPUT_DIR, "kontext_A.png")
src_r.save(A_path)
A_name = upload(A_path)
print("A: " + A_name)

# A -> B
print("\n=== A -> B: Add black pants ==="); sys.stdout.flush()
B = queue_wait(build_kontext(A_name,
    "The character is now wearing long black pants. Everything else stays exactly the same - same face, same pose, same upper body, same background.",
    "kontext_B", 500))

if B:
    # B -> C
    print("\n=== B -> C: Remove pants, restore green swimwear ==="); sys.stdout.flush()
    B_name = upload(B)
    C = queue_wait(build_kontext(B_name,
        "The character is now wearing a small green speedo swimwear instead of the black pants. Everything else stays exactly the same.",
        "kontext_C", 600))

    if C:
        html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Kontext ABC</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:30px}.row{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;width:220px}
.item img{width:100%%;display:block}.item p{padding:12px;text-align:center}
.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:150px}</style></head><body>
<h1>FLUX Kontext: Pants A-B-C</h1><div class="row">
<div class="item"><img src="kontext_A.png"><p>A - Original</p></div>
<div class="arrow">-&gt;</div><div class="item"><img src="%s"><p>B - +Pants</p></div>
<div class="arrow">-&gt;</div><div class="item"><img src="%s"><p>C - Restored</p></div>
</div></body></html>""" % (os.path.basename(B), os.path.basename(C))
        rpt = os.path.join(OUTPUT_DIR, "kontext_abc_report.html")
        with open(rpt, "w", encoding="utf-8") as f: f.write(html)

        import subprocess
        BOT = "8796302477:AAH0OX_fn9DIqL8O7vBupBCzhL6t3MdPIfI"
        CHAT = "755113426"
        API = "https://api.telegram.org/bot" + BOT
        for fp, cap in [(A_path,"Kontext A: Original"), (B,"Kontext B: +Pants"), (C,"Kontext C: Restored")]:
            jpg = fp.replace(".png",".jpg")
            Image.open(fp).save(jpg, "JPEG", quality=90)
            subprocess.run(["curl","-s","-X","POST",API+"/sendPhoto","-F","chat_id="+CHAT,"-F","photo=@"+jpg,"-F","caption="+cap], capture_output=True)
        print("Sent to Telegram!")
        os.startfile(rpt)

print("\nDone!")
