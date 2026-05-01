"""
Generate T-pose hero characters via ComfyUI API + split into parts via OpenCV.
Pipeline: ComfyUI (animagine-xl-3.1, 768x768) -> background removal -> 6-part split -> copy to game project.
"""

import json
import os
import shutil
import time
import urllib.request
import urllib.error
import uuid

import cv2
import numpy as np

COMFYUI_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = r"C:\Users\Yusheng Ding\Desktop\projects\ComfyUI\output"
PARTS_DIR = os.path.join(OUTPUT_DIR, "character_parts")
GAME_DST = r"C:\Users\Yusheng Ding\Desktop\projects\dungeon_shopkeeper\dungeon-shopkeeper\assets\heroes\parts_v2"

NEGATIVE = "lowres, bad anatomy, extra limbs, multiple characters, text, watermark, blurry, jpeg artifacts, complex background"

HEROES = {
    "ling": "1girl, elf mage, long silver hair, blue eyes, flowing white-blue robe, crystal staff, T-pose, arms spread wide, full body, standing straight, facing viewer, simple flat color, white background, clean lineart, game character, masterpiece",
    "moye": "1boy, berserker warrior, wild red hair, face paint, leather and fur armor, large axe, T-pose, arms spread wide, full body, standing straight, facing viewer, simple flat color, white background, clean lineart, game character, masterpiece",
    "sha": "1girl, assassin ninja, short black hair, dark outfit, twin daggers, T-pose, arms spread wide, full body, standing straight, facing viewer, simple flat color, white background, clean lineart, game character, masterpiece",
    "tie": "1boy, knight paladin, silver plate armor, shield, helmet under arm, blonde hair, T-pose, arms spread wide, full body, standing straight, facing viewer, simple flat color, white background, clean lineart, game character, masterpiece",
    "xiao_gong": "1boy, young archer, messy brown hair, green tunic, longbow on back, quiver, T-pose, arms spread wide, full body, standing straight, facing viewer, simple flat color, white background, clean lineart, game character, masterpiece",
}


def build_workflow(prompt: str, negative: str, filename_prefix: str, seed: int) -> dict:
    """Build SDXL txt2img workflow for ComfyUI API."""
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": 28,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "animagine-xl-3.1.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 768, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": filename_prefix, "images": ["8", 0]},
        },
    }


def queue_prompt(workflow: dict) -> str:
    """Queue a prompt and return the prompt_id."""
    client_id = str(uuid.uuid4())
    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    return data["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 300) -> dict:
    """Poll history until prompt completes."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}")
            history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.URLError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Prompt {prompt_id} did not complete within {timeout}s")


def get_output_image_path(history: dict) -> str:
    """Extract the saved image path from history outputs."""
    for node_id, node_output in history["outputs"].items():
        if "images" in node_output:
            img_info = node_output["images"][0]
            return os.path.join(OUTPUT_DIR, img_info["subfolder"], img_info["filename"])
    raise ValueError("No image output found in history")


def remove_background(img_bgr: np.ndarray) -> np.ndarray:
    """Remove near-white background, return BGRA image with transparency."""
    # Convert to grayscale to detect white background
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold: white background pixels (>235) become 0 in mask
    # Use adaptive approach: sample corners to find actual bg color
    h, w = img_bgr.shape[:2]
    corners = [
        img_bgr[0, 0], img_bgr[0, w - 1],
        img_bgr[h - 1, 0], img_bgr[h - 1, w - 1],
    ]
    bg_color = np.mean(corners, axis=0).astype(np.uint8)

    # Create mask based on color distance from background
    diff = np.sqrt(np.sum((img_bgr.astype(float) - bg_color.astype(float)) ** 2, axis=2))
    # Foreground = pixels sufficiently different from background
    fg_mask = (diff > 30).astype(np.uint8) * 255

    # Clean up mask with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Add alpha channel
    bgra = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = fg_mask

    return bgra


def split_character(bgra: np.ndarray, name: str, out_dir: str):
    """Split T-pose character into 6 parts: head, body, arm_l, arm_r, leg_l, leg_r + full."""
    h, w = bgra.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    # Save full with transparency
    cv2.imwrite(os.path.join(out_dir, "full.png"), bgra)

    # Find character bounding box from alpha
    alpha = bgra[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any():
        print(f"  WARNING: No foreground found for {name}!")
        return

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    char_h = y_max - y_min
    char_w = x_max - x_min
    cx = (x_min + x_max) // 2  # character center x

    # Split ratios (relative to character bounding box)
    # Head: top ~23% of character
    # Body: ~23% to ~65% vertically, central region
    # Arms: ~20% to ~55% vertically, left/right thirds
    # Legs: ~55% to bottom, left/right halves

    head_bottom = y_min + int(char_h * 0.30)
    body_top = y_min + int(char_h * 0.18)
    body_bottom = y_min + int(char_h * 0.65)
    arm_top = y_min + int(char_h * 0.15)
    arm_bottom = y_min + int(char_h * 0.55)
    leg_top = y_min + int(char_h * 0.55)
    leg_bottom = y_max + 5

    # Overlap margins
    margin = 15

    def crop_and_save(region_bgra: np.ndarray, filename: str):
        """Crop to non-transparent bounding box and save."""
        a = region_bgra[:, :, 3]
        r = np.any(a > 0, axis=1)
        c = np.any(a > 0, axis=0)
        if not r.any() or not c.any():
            print(f"  WARNING: Empty region for {filename}")
            # Save a small placeholder
            cv2.imwrite(os.path.join(out_dir, filename), region_bgra[:10, :10])
            return
        ry_min, ry_max = np.where(r)[0][[0, -1]]
        rx_min, rx_max = np.where(c)[0][[0, -1]]
        # Add small padding
        pad = 3
        ry_min = max(0, ry_min - pad)
        ry_max = min(region_bgra.shape[0] - 1, ry_max + pad)
        rx_min = max(0, rx_min - pad)
        rx_max = min(region_bgra.shape[1] - 1, rx_max + pad)
        cropped = region_bgra[ry_min:ry_max + 1, rx_min:rx_max + 1]
        cv2.imwrite(os.path.join(out_dir, filename), cropped)
        print(f"  {filename}: {cropped.shape[1]}x{cropped.shape[0]}")

    # Head region
    head_region = bgra[max(0, y_min - margin):head_bottom, :, :].copy()
    # Mask out pixels below head line except center column (for neck)
    crop_and_save(head_region, "head.png")

    # Body region (center strip)
    body_left = cx - int(char_w * 0.30)
    body_right = cx + int(char_w * 0.30)
    body_region = bgra[body_top:body_bottom, max(0, body_left):min(w, body_right), :].copy()
    crop_and_save(body_region, "body.png")

    # Left arm (viewer's left = image left side)
    arm_l_region = bgra[arm_top:arm_bottom, max(0, x_min - margin):cx - int(char_w * 0.10), :].copy()
    crop_and_save(arm_l_region, "arm_l.png")

    # Right arm
    arm_r_region = bgra[arm_top:arm_bottom, cx + int(char_w * 0.10):min(w, x_max + margin), :].copy()
    crop_and_save(arm_r_region, "arm_r.png")

    # Left leg
    leg_l_region = bgra[leg_top:min(h, leg_bottom):, max(0, x_min - margin):cx + margin, :].copy()
    crop_and_save(leg_l_region, "leg_l.png")

    # Right leg
    leg_r_region = bgra[leg_top:min(h, leg_bottom), cx - margin:min(w, x_max + margin), :].copy()
    crop_and_save(leg_r_region, "leg_r.png")


def main():
    print("=" * 60)
    print("Hero T-Pose Generation Pipeline")
    print("=" * 60)

    for hero_name, prompt in HEROES.items():
        print(f"\n--- Generating: {hero_name} ---")
        prefix = f"tpose_{hero_name}"
        seed = hash(hero_name) % (2**32)

        # Build and queue workflow
        workflow = build_workflow(prompt, NEGATIVE, prefix, seed)
        prompt_id = queue_prompt(workflow)
        print(f"  Queued: {prompt_id}")

        # Wait for completion
        history = wait_for_completion(prompt_id)
        img_path = get_output_image_path(history)
        print(f"  Generated: {img_path}")

        # Load and process
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  ERROR: Could not read {img_path}")
            continue

        # Remove background
        bgra = remove_background(img_bgr)
        print(f"  Background removed")

        # Split into parts
        hero_parts_dir = os.path.join(PARTS_DIR, hero_name)
        split_character(bgra, hero_name, hero_parts_dir)
        print(f"  Parts saved to: {hero_parts_dir}")

    # Copy to game project
    print(f"\n--- Copying to game project ---")
    os.makedirs(GAME_DST, exist_ok=True)
    for hero_name in HEROES:
        src = os.path.join(PARTS_DIR, hero_name)
        dst = os.path.join(GAME_DST, hero_name)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  Copied {hero_name} -> {dst}")
        else:
            print(f"  WARNING: {src} not found, skipping")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
