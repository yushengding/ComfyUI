"""
SDXL + Fooocus Inpaint + LaMa 高质量工作流验证测试
=====================================================
测试流程：
  Round 1: 原图 → 添加围巾 → 去除围巾
  Round 2: 原图 → 添加裹胸 → 去除裹胸
"""
import json, urllib.request, time, os, sys, struct, zlib

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"


def upload_image(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        img_data = f.read()
    boundary = "----UpBound"
    body = f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n".encode()
    body += img_data
    body += f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{BASE}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp["name"]


def create_mask_png(w, h, regions):
    raw_rows = []
    for y in range(h):
        row = b'\x00'
        for x in range(w):
            val = 0
            for (x1, y1, x2, y2) in regions:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    val = 255
                    break
            row += bytes([val])
        raw_rows.append(row)
    raw = b''.join(raw_rows)
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 9))
    png += chunk(b'IEND', b'')
    return png


def queue_and_wait(workflow, timeout=300):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    if resp.get("error"):
        print(f"  SUBMIT ERROR: {json.dumps(resp)[:800]}")
        return None
    pid = resp["prompt_id"]
    print(f"  Queued: {pid}")
    sys.stdout.flush()

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(10)
        elapsed = int(time.time() - start)
        try:
            hist = json.loads(urllib.request.urlopen(f"{BASE}/history/{pid}").read())
            entry = hist.get(pid)
            if not entry:
                print(f"  {elapsed}s: generating...")
                sys.stdout.flush()
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                for m in status.get("messages", []):
                    if isinstance(m, list) and len(m) > 1 and isinstance(m[1], dict):
                        em = m[1].get("exception_message", "")
                        if em:
                            print(f"  ERROR: {em[:400]}")
                            sys.stdout.flush()
                return None
            outputs = entry.get("outputs", {})
            if outputs:
                for node in outputs.values():
                    for img in node.get("images", []):
                        fpath = os.path.join(OUTPUT_DIR, img.get("subfolder", ""), img["filename"])
                        print(f"  SUCCESS ({elapsed}s): {fpath}")
                        sys.stdout.flush()
                        return fpath
        except:
            pass
    print(f"  TIMEOUT after {timeout}s")
    sys.stdout.flush()
    return None


def build_sdxl_fooocus_workflow(source_name, mask_name, prompt, neg_prompt, prefix, seed=42):
    """SDXL + Fooocus Inpaint + LaMa prefill workflow"""
    return {
        # Load SDXL checkpoint
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "animagine-xl-3.1.safetensors"}},
        # Load source image
        "2": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        # Load mask
        "6": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "7": {"class_type": "ImageToMask", "inputs": {"image": ["6", 0], "channel": "red"}},
        # LaMa prefill - fill masked area intelligently before inpainting
        "8": {"class_type": "INPAINT_InpaintWithModel",
              "inputs": {"inpaint_model": ["9", 0], "image": ["2", 0], "mask": ["7", 0], "seed": seed}},
        "9": {"class_type": "INPAINT_LoadInpaintModel",
              "inputs": {"model_name": "big-lama.pt"}},
        # Fooocus inpaint - transform SDXL into inpaint model
        "10": {"class_type": "INPAINT_LoadFooocusInpaint",
               "inputs": {"head": "fooocus_inpaint_head.pth", "patch": "inpaint_v26.fooocus.patch"}},
        "11": {"class_type": "INPAINT_ApplyFooocusInpaint",
               "inputs": {"model": ["1", 0], "patch": ["10", 0], "latent": ["15", 0]}},
        # Encode prefilled image to latent
        "14": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["8", 0], "vae": ["1", 2]}},
        # Set noise mask on latent
        "15": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["14", 0], "mask": ["7", 0]}},
        # CLIP encode prompts
        "20": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["1", 1], "text": prompt}},
        "21": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["1", 1], "text": neg_prompt}},
        # KSampler with Fooocus-patched model
        "30": {"class_type": "KSampler",
               "inputs": {"model": ["11", 0], "positive": ["20", 0], "negative": ["21", 0],
                          "latent_image": ["15", 0], "seed": seed, "steps": 25, "cfg": 7.0,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        # Decode
        "40": {"class_type": "VAEDecode",
               "inputs": {"samples": ["30", 0], "vae": ["1", 2]}},
        # Save
        "50": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": prefix, "images": ["40", 0]}}
    }


def main():
    print("=" * 60)
    print("SDXL + Fooocus Inpaint + LaMa - Validation Test")
    print("=" * 60)
    sys.stdout.flush()

    # Check server
    try:
        urllib.request.urlopen(f"{BASE}/system_stats", timeout=3)
    except:
        print("ComfyUI not running!")
        sys.exit(1)

    # Check nodes available
    resp = json.loads(urllib.request.urlopen(f"{BASE}/object_info").read())
    needed = ["INPAINT_InpaintWithModel", "INPAINT_LoadInpaintModel",
              "INPAINT_LoadFooocusInpaint", "INPAINT_ApplyFooocusInpaint"]
    missing = [n for n in needed if n not in resp]
    if missing:
        print(f"Missing nodes: {missing}")
        print("comfyui-inpaint-nodes plugin not loaded?")
        sys.exit(1)
    print("All required nodes available!")

    # Check models
    ckpts = resp["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    if "animagine-xl-3.1.safetensors" not in ckpts:
        print("animagine-xl-3.1.safetensors not found in checkpoints!")
        sys.exit(1)
    print("Animagine XL 3.1 model found!")
    sys.stdout.flush()

    # Resize source to 1024x1536 (SDXL native res)
    from PIL import Image
    src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
    src_resized = src.resize((768, 1152), Image.LANCZOS)
    resized_path = os.path.join(OUTPUT_DIR, "test_sdxl.png")
    src_resized.save(resized_path)
    source_name = upload_image(resized_path)
    print(f"Source uploaded: {source_name} (768x1152)")

    NEG = "low quality, worst quality, blurry, bad anatomy, deformed, ugly, extra limbs, missing limbs, watermark, text, extra fingers, mutated hands"

    # Masks for 768x1152
    # Neck/scarf: y=340~460, x=220~540
    mask_scarf = create_mask_png(768, 1152, [(220, 340, 540, 460)])
    scarf_path = os.path.join(OUTPUT_DIR, "mask_sdxl_scarf.png")
    with open(scarf_path, "wb") as f:
        f.write(mask_scarf)
    scarf_mask = upload_image(scarf_path)

    # Chest: y=420~650, x=200~570
    mask_chest = create_mask_png(768, 1152, [(200, 420, 570, 650)])
    chest_path = os.path.join(OUTPUT_DIR, "mask_sdxl_chest.png")
    with open(chest_path, "wb") as f:
        f.write(mask_chest)
    chest_mask = upload_image(chest_path)

    results = {}

    # Round 1A: Add scarf
    print("\n" + "=" * 60)
    print("Round 1A: Add a red scarf")
    print("=" * 60)
    sys.stdout.flush()
    wf1 = build_sdxl_fooocus_workflow(
        source_name, scarf_mask,
        "1boy, wearing a red knitted scarf around the neck, cozy winter scarf, anime style, detailed, masterpiece, best quality",
        NEG, "sdxl_add_scarf", seed=100)
    r1a = queue_and_wait(wf1, timeout=300)
    results["add_scarf"] = r1a

    if r1a:
        # Round 1B: Remove scarf
        print("\n" + "=" * 60)
        print("Round 1B: Remove the scarf")
        print("=" * 60)
        sys.stdout.flush()
        scarf_img = upload_image(r1a)
        wf1b = build_sdxl_fooocus_workflow(
            scarf_img, scarf_mask,
            "1boy, bare neck, no scarf, exposed skin, anime style, detailed, masterpiece, best quality",
            NEG, "sdxl_remove_scarf", seed=200)
        r1b = queue_and_wait(wf1b, timeout=300)
        results["remove_scarf"] = r1b

    # Round 2A: Add chest wrap
    print("\n" + "=" * 60)
    print("Round 2A: Add chest wrap")
    print("=" * 60)
    sys.stdout.flush()
    wf2 = build_sdxl_fooocus_workflow(
        source_name, chest_mask,
        "1boy, wearing a tight white bandage wrap on chest, cloth covering torso, anime style, detailed, masterpiece, best quality",
        NEG, "sdxl_add_chestwrap", seed=300)
    r2a = queue_and_wait(wf2, timeout=300)
    results["add_chestwrap"] = r2a

    if r2a:
        # Round 2B: Remove chest wrap
        print("\n" + "=" * 60)
        print("Round 2B: Remove chest wrap")
        print("=" * 60)
        sys.stdout.flush()
        chest_img = upload_image(r2a)
        wf2b = build_sdxl_fooocus_workflow(
            chest_img, chest_mask,
            "1boy, bare muscular chest, no clothing on torso, exposed muscles, anime style, detailed, masterpiece, best quality",
            NEG, "sdxl_remove_chestwrap", seed=400)
        r2b = queue_and_wait(wf2b, timeout=300)
        results["remove_chestwrap"] = r2b

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = "OK" if v else "FAILED"
        print(f"  {k}: {status} -> {v}")
    sys.stdout.flush()

    # Generate HTML report
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>SDXL Fooocus Inpaint Test</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:30px}h2{color:#58a6ff;margin:30px 0 15px;border-bottom:1px solid #21262d;padding-bottom:8px}
.row{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0}.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;flex:1;min-width:200px;max-width:300px}
.item img{width:100%;display:block}.item p{padding:10px;text-align:center;font-size:0.9rem;color:#8b949e}
.tag{display:inline-block;background:#1f6feb22;border:1px solid #1f6feb;color:#58a6ff;padding:2px 8px;border-radius:20px;font-size:0.75rem}
.arrow{display:flex;align-items:center;font-size:2rem;color:#3fb950;padding-top:150px}</style></head><body>
<h1>SDXL + Fooocus Inpaint + LaMa 验证报告</h1>"""

    html += '<h2>Round 1: 围巾测试 (添加 → 去除)</h2><div class="row">'
    html += '<div class="item"><img src="test_sdxl.png"><p>原图 768x1152</p></div>'
    html += '<div class="item"><img src="mask_sdxl_scarf.png"><p>蒙版 (颈部)</p></div>'
    if results.get("add_scarf"):
        html += f'<div class="arrow">→</div><div class="item"><img src="{os.path.basename(results["add_scarf"])}"><p><span class="tag">+围巾</span></p></div>'
    if results.get("remove_scarf"):
        html += f'<div class="arrow">→</div><div class="item"><img src="{os.path.basename(results["remove_scarf"])}"><p><span class="tag">-围巾(还原)</span></p></div>'
    html += '</div>'

    html += '<h2>Round 2: 裹胸测试 (添加 → 去除)</h2><div class="row">'
    html += '<div class="item"><img src="test_sdxl.png"><p>原图</p></div>'
    html += '<div class="item"><img src="mask_sdxl_chest.png"><p>蒙版 (胸部)</p></div>'
    if results.get("add_chestwrap"):
        html += f'<div class="arrow">→</div><div class="item"><img src="{os.path.basename(results["add_chestwrap"])}"><p><span class="tag">+裹胸</span></p></div>'
    if results.get("remove_chestwrap"):
        html += f'<div class="arrow">→</div><div class="item"><img src="{os.path.basename(results["remove_chestwrap"])}"><p><span class="tag">-裹胸(还原)</span></p></div>'
    html += '</div></body></html>'

    report_path = os.path.join(OUTPUT_DIR, "sdxl_inpaint_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport: {report_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
