"""
FLUX Fill Dev 自动化验证测试
=============================
测试流程：
  Round 1: 原图 → 添加围巾 → 去除围巾 → 对比原图
  Round 2: 原图 → 添加裹胸 → 去除裹胸 → 对比原图
"""
import json, urllib.request, time, os, sys, struct, zlib

BASE = "http://127.0.0.1:8188"
OUTPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/output"
INPUT_DIR = "C:/Users/Yusheng Ding/Desktop/projects/ComfyUI/input"

# ============================================================
# Helpers
# ============================================================
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
    """regions = [(x1,y1,x2,y2), ...], white = edit area"""
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


def queue_and_wait(workflow, timeout=600):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    if resp.get("error"):
        print(f"  SUBMIT ERROR: {json.dumps(resp)[:500]}")
        return None
    pid = resp["prompt_id"]
    print(f"  Queued: {pid}")

    start = time.time()
    while time.time() - start < timeout:
        time.sleep(10)
        elapsed = int(time.time() - start)
        try:
            hist = json.loads(urllib.request.urlopen(f"{BASE}/history/{pid}").read())
            entry = hist.get(pid)
            if not entry:
                print(f"  {elapsed}s: generating...")
                continue
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                for m in status.get("messages", []):
                    if isinstance(m, list) and len(m) > 1 and isinstance(m[1], dict):
                        em = m[1].get("exception_message", "")
                        if em:
                            print(f"  ERROR: {em[:300]}")
                return None
            outputs = entry.get("outputs", {})
            if outputs:
                for node in outputs.values():
                    for img in node.get("images", []):
                        fpath = os.path.join(OUTPUT_DIR, img.get("subfolder", ""), img["filename"])
                        print(f"  SUCCESS ({elapsed}s): {fpath}")
                        return fpath
        except:
            pass
    print(f"  TIMEOUT after {timeout}s")
    return None


def build_fill_workflow(source_name, mask_name, prompt, prefix, seed=42):
    """Build FLUX Fill inpainting workflow"""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "flux1-fill-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp8_e4m3fn_scaled.safetensors", "type": "flux"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "ae.safetensors"}},
        # Source image
        "4": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        # Mask image
        "6": {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        "7": {"class_type": "ImageToMask", "inputs": {"image": ["6", 0], "channel": "red"}},
        # Positive prompt
        "10": {"class_type": "CLIPTextEncodeFlux",
               "inputs": {"clip": ["2", 0], "clip_l": prompt, "t5xxl": prompt, "guidance": 30.0}},
        # Negative (same as positive for FLUX - no real negative)
        "10b": {"class_type": "CLIPTextEncodeFlux",
                "inputs": {"clip": ["2", 0], "clip_l": "", "t5xxl": "", "guidance": 30.0}},
        # InpaintModelConditioning - connects image + mask to conditioning
        "15": {"class_type": "InpaintModelConditioning",
               "inputs": {"positive": ["10", 0], "negative": ["10b", 0],
                          "vae": ["3", 0], "pixels": ["4", 0],
                          "mask": ["7", 0], "noise_mask": True}},
        # Empty latent matching source size (will be overridden by inpaint conditioning)
        "20": {"class_type": "EmptySD3LatentImage",
               "inputs": {"width": 512, "height": 768, "batch_size": 1}},
        # KSampler
        "30": {"class_type": "KSampler",
               "inputs": {"model": ["1", 0],
                          "positive": ["15", 0],   # from InpaintModelConditioning
                          "negative": ["15", 1],   # from InpaintModelConditioning
                          "latent_image": ["15", 2],  # from InpaintModelConditioning
                          "seed": seed, "steps": 20, "cfg": 1.0,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "40": {"class_type": "VAEDecode",
               "inputs": {"samples": ["30", 0], "vae": ["3", 0]}},
        "50": {"class_type": "SaveImage",
               "inputs": {"filename_prefix": prefix, "images": ["40", 0]}}
    }


# ============================================================
# Main Test
# ============================================================
def main():
    print("=" * 60)
    print("FLUX Fill Dev - Automated Validation Test")
    print("=" * 60)

    # Check server
    try:
        urllib.request.urlopen(f"{BASE}/system_stats", timeout=3)
    except:
        print("ComfyUI not running!")
        sys.exit(1)

    # Resize source to 512x768 for speed
    print("\n[Prep] Resizing source image to 512x768...")
    from PIL import Image
    src = Image.open(os.path.join(INPUT_DIR, "test_source.png"))
    src_resized = src.resize((512, 768), Image.LANCZOS)
    resized_path = os.path.join(OUTPUT_DIR, "test_512x768.png")
    src_resized.save(resized_path)

    source_name = upload_image(resized_path)
    print(f"  Uploaded: {source_name}")

    # Create masks
    # Neck/scarf area: roughly y=230~310, x=150~360
    print("\n[Prep] Creating scarf mask (neck area)...")
    mask_scarf = create_mask_png(512, 768, [(150, 230, 360, 310)])
    scarf_mask_path = os.path.join(OUTPUT_DIR, "mask_scarf.png")
    with open(scarf_mask_path, "wb") as f:
        f.write(mask_scarf)
    scarf_mask_name = upload_image(scarf_mask_path)

    # Chest area: roughly y=280~420, x=130~380
    print("[Prep] Creating chest wrap mask...")
    mask_chest = create_mask_png(512, 768, [(130, 280, 380, 420)])
    chest_mask_path = os.path.join(OUTPUT_DIR, "mask_chest.png")
    with open(chest_mask_path, "wb") as f:
        f.write(mask_chest)
    chest_mask_name = upload_image(chest_mask_path)

    results = {}

    # ---- Round 1: Add scarf ----
    print("\n" + "=" * 60)
    print("Round 1A: Add a scarf")
    print("=" * 60)
    wf1 = build_fill_workflow(source_name, scarf_mask_name,
                              "wearing a red knitted scarf around the neck, cozy winter scarf, detailed fabric texture",
                              "fill_add_scarf", seed=100)
    r1a = queue_and_wait(wf1, timeout=600)
    results["add_scarf"] = r1a

    if r1a:
        # Round 1B: Remove scarf from generated image
        print("\n" + "=" * 60)
        print("Round 1B: Remove the scarf (restore original)")
        print("=" * 60)
        scarf_img_name = upload_image(r1a)
        wf1b = build_fill_workflow(scarf_img_name, scarf_mask_name,
                                   "bare neck, no scarf, original skin, clean neck area matching the rest of the body",
                                   "fill_remove_scarf", seed=200)
        r1b = queue_and_wait(wf1b, timeout=600)
        results["remove_scarf"] = r1b

    # ---- Round 2: Add chest wrap ----
    print("\n" + "=" * 60)
    print("Round 2A: Add a chest wrap / tube top")
    print("=" * 60)
    wf2 = build_fill_workflow(source_name, chest_mask_name,
                              "wearing a tight white tube top, bandage chest wrap, cloth covering chest",
                              "fill_add_chestwrap", seed=300)
    r2a = queue_and_wait(wf2, timeout=600)
    results["add_chestwrap"] = r2a

    if r2a:
        # Round 2B: Remove chest wrap
        print("\n" + "=" * 60)
        print("Round 2B: Remove the chest wrap (restore original)")
        print("=" * 60)
        chest_img_name = upload_image(r2a)
        wf2b = build_fill_workflow(chest_img_name, chest_mask_name,
                                   "bare muscular chest, no clothing, original skin tone, matching the rest of the body",
                                   "fill_remove_chestwrap", seed=400)
        r2b = queue_and_wait(wf2b, timeout=600)
        results["remove_chestwrap"] = r2b

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = "OK" if v else "FAILED"
        print(f"  {k}: {status} -> {v}")

    # Generate HTML report
    print("\nGenerating comparison report...")
    html = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>FLUX Fill Test</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:40px}
h1{color:#fff;margin-bottom:30px}h2{color:#58a6ff;margin:30px 0 15px;border-bottom:1px solid #21262d;padding-bottom:8px}
.row{display:flex;gap:16px;flex-wrap:wrap;margin:10px 0}.item{background:#161b22;border:1px solid #30363d;border-radius:12px;overflow:hidden;flex:1;min-width:200px}
.item img{width:100%;display:block}.item p{padding:10px;text-align:center;font-size:0.9rem;color:#8b949e}
.pass{color:#3fb950}.fail{color:#f85149}</style></head><body>
<h1>FLUX Fill Dev - Inpaint Validation Report</h1>"""

    html += '<h2>Round 1: Scarf Test</h2><div class="row">'
    html += f'<div class="item"><img src="test_512x768.png"><p>Original</p></div>'
    html += f'<div class="item"><img src="mask_scarf.png"><p>Mask (neck)</p></div>'
    if results.get("add_scarf"):
        html += f'<div class="item"><img src="{os.path.basename(results["add_scarf"])}"><p>+ Scarf</p></div>'
    if results.get("remove_scarf"):
        html += f'<div class="item"><img src="{os.path.basename(results["remove_scarf"])}"><p>- Scarf (restored)</p></div>'
    html += '</div>'

    html += '<h2>Round 2: Chest Wrap Test</h2><div class="row">'
    html += f'<div class="item"><img src="test_512x768.png"><p>Original</p></div>'
    html += f'<div class="item"><img src="mask_chest.png"><p>Mask (chest)</p></div>'
    if results.get("add_chestwrap"):
        html += f'<div class="item"><img src="{os.path.basename(results["add_chestwrap"])}"><p>+ Chest Wrap</p></div>'
    if results.get("remove_chestwrap"):
        html += f'<div class="item"><img src="{os.path.basename(results["remove_chestwrap"])}"><p>- Chest Wrap (restored)</p></div>'
    html += '</div>'

    html += '</body></html>'

    report_path = os.path.join(OUTPUT_DIR, "flux_fill_test_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
