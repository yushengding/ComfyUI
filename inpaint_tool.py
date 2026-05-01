"""
ComfyUI 局部重绘工具 — 支持参考图引用
=========================================
功能：
  1. 对源图片的指定区域进行重绘（Inpainting）
  2. 用文本 prompt 控制修改方向
  3. 可引用另一张图片的局部区域作为参考（IP-Adapter）
  4. 保持修改区域与原图的一致性

用法：
  # 基础：只用 prompt 控制重绘
  python inpaint_tool.py --source photo.png --mask mask.png --prompt "muscular chest"

  # 进阶：用参考图的某个区域引导重绘
  python inpaint_tool.py --source photo.png --mask mask.png --prompt "muscular chest" \
      --reference ref.png --ref-crop 100,200,256,256 --ref-weight 0.7

  # 手动创建蒙版（打开画笔工具标记要修改的区域）
  python inpaint_tool.py --source photo.png --draw-mask

依赖：pip install Pillow
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
import io
import struct
import zlib

BASE_URL = "http://127.0.0.1:8188"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


# =============================================================================
# ComfyUI API
# =============================================================================

def check_server():
    try:
        urllib.request.urlopen(f"{BASE_URL}/system_stats", timeout=3)
        return True
    except:
        return False


def upload_image(filepath, subfolder="", image_type="input"):
    """上传图片到 ComfyUI，返回上传后的文件名"""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        img_data = f.read()

    boundary = "----ComfyUploadBoundary"
    parts = []

    # image field
    parts.append(f"--{boundary}\r\n")
    parts.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n')
    parts.append("Content-Type: image/png\r\n\r\n")

    # subfolder field
    sub_part = f"--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n{subfolder}\r\n"

    # type field
    type_part = f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\n{image_type}\r\n"

    # closing
    close_part = f"--{boundary}--\r\n"

    body = b""
    for p in parts:
        body += p.encode()
    body += img_data
    body += f"\r\n{sub_part}{type_part}{close_part}".encode()

    req = urllib.request.Request(
        f"{BASE_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    return resp.get("name", filename)


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if "error" in resp:
        print(f"ComfyUI Error: {resp['error']}")
        if "node_errors" in resp:
            for node_id, err in resp["node_errors"].items():
                print(f"  Node {node_id}: {err}")
        sys.exit(1)
    return resp.get("prompt_id", "unknown")


def wait_result(prompt_id, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(2)
        try:
            raw = urllib.request.urlopen(f"{BASE_URL}/history/{prompt_id}").read()
            history = json.loads(raw)
            entry = history.get(prompt_id)
            if not entry:
                continue
            images = []
            for node in entry.get("outputs", {}).values():
                for img in node.get("images", []):
                    p = os.path.join(OUTPUT_DIR, img.get("subfolder", ""), img["filename"])
                    images.append(p)
            if images:
                return images
        except:
            pass
    raise TimeoutError(f"Generation timed out after {timeout}s")


# =============================================================================
# Mask 工具
# =============================================================================

def create_blank_mask(width, height, save_path):
    """创建全黑蒙版 PNG（无需 Pillow）"""
    # 简单的纯黑 PNG
    def create_png(w, h):
        def chunk(chunk_type, data):
            c = chunk_type + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

        header = b'\x89PNG\r\n\x1a\n'
        ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0))
        raw = b''
        for y in range(h):
            raw += b'\x00' + b'\x00' * w
        idat = chunk(b'IDAT', zlib.compress(raw))
        iend = chunk(b'IEND', b'')
        return header + ihdr + idat + iend

    with open(save_path, 'wb') as f:
        f.write(create_png(width, height))


def draw_mask_interactive(source_path, mask_path):
    """用 tkinter 简单画笔画蒙版"""
    try:
        from PIL import Image, ImageTk
        import tkinter as tk
    except ImportError:
        print("需要 Pillow: pip install Pillow")
        sys.exit(1)

    img = Image.open(source_path).convert("RGB")
    w, h = img.size

    # 缩放显示（最大800px）
    scale = min(800 / w, 800 / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    root = tk.Tk()
    root.title("画蒙版 — 白色=修改区域 | 左键画 | 右键橡皮 | 滚轮笔刷大小 | Enter保存")

    canvas = tk.Canvas(root, width=dw, height=dh)
    canvas.pack()

    display_img = img.resize((dw, dh), Image.LANCZOS)
    photo = ImageTk.PhotoImage(display_img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # 蒙版数据
    mask = Image.new("L", (w, h), 0)  # 全黑
    brush_size = [30]
    drawing = [False]
    erasing = [False]

    def paint(event, erase=False):
        # 转换到原图坐标
        ox, oy = int(event.x / scale), int(event.y / scale)
        r = int(brush_size[0] / scale)
        color = 0 if erase else 255
        # 画圆
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=color)
        # 画布上显示
        cr = brush_size[0]
        c = "black" if erase else "white"
        canvas.create_oval(event.x - cr, event.y - cr, event.x + cr, event.y + cr,
                          fill=c, outline=c, stipple="gray50" if not erase else "gray25")

    def on_press(event):
        drawing[0] = True
        paint(event, erasing[0])

    def on_move(event):
        if drawing[0]:
            paint(event, erasing[0])

    def on_release(event):
        drawing[0] = False

    def on_right_press(event):
        erasing[0] = True
        drawing[0] = True
        paint(event, True)

    def on_right_release(event):
        erasing[0] = False
        drawing[0] = False

    def on_scroll(event):
        if event.delta > 0:
            brush_size[0] = min(100, brush_size[0] + 5)
        else:
            brush_size[0] = max(5, brush_size[0] - 5)
        root.title(f"笔刷大小: {brush_size[0]} | Enter保存")

    def on_save(event):
        mask.save(mask_path)
        print(f"蒙版已保存: {mask_path}")
        root.destroy()

    canvas.bind("<Button-1>", on_press)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<Button-3>", on_right_press)
    canvas.bind("<B3-Motion>", lambda e: paint(e, True) if drawing[0] else None)
    canvas.bind("<ButtonRelease-3>", on_right_release)
    root.bind("<MouseWheel>", on_scroll)
    root.bind("<Return>", on_save)
    root.bind("<Escape>", lambda e: root.destroy())

    root.mainloop()

    if not os.path.exists(mask_path):
        print("未保存蒙版，已取消")
        sys.exit(0)


# =============================================================================
# 工作流构建
# =============================================================================

def build_inpaint_workflow(
    source_name,
    mask_name,
    prompt,
    negative_prompt,
    reference_name=None,
    ref_crop=None,       # (x, y, w, h)
    ref_weight=0.65,
    denoise=0.85,
    grow_mask=8,
    steps=25,
    cfg=7.0,
    seed=-1,
    model="dreamshaper_8.safetensors",
    prefix="inpaint_result",
):
    if seed < 0:
        import random
        seed = random.randint(0, 2**31)

    wf = {}

    # --- 基础节点 ---
    wf["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": model},
    }
    wf["2"] = {
        "class_type": "LoadImage",
        "inputs": {"image": source_name},
    }
    wf["4"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": prompt},
    }
    wf["5"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["1", 1], "text": negative_prompt},
    }

    # --- Mask 加载 ---
    wf["6"] = {
        "class_type": "LoadImage",
        "inputs": {"image": mask_name},
    }

    # --- 编码源图 + Mask ---
    # 使用 ImageToMask 将 mask 图片转为 MASK 类型
    wf["7"] = {
        "class_type": "ImageToMask",
        "inputs": {"image": ["6", 0], "channel": "red"},
    }
    wf["20"] = {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": ["2", 0],
            "vae": ["1", 2],
            "mask": ["7", 0],
            "grow_mask_by": grow_mask,
        },
    }

    # --- 模型来源：是否用 IP-Adapter ---
    model_source = ["1", 0]  # 默认直接用 checkpoint model

    if reference_name:
        wf["3"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_name},
        }
        wf["10"] = {
            "class_type": "IPAdapterModelLoader",
            "inputs": {"ipadapter_file": "ip-adapter-plus_sd15.safetensors"},
        }
        wf["11"] = {
            "class_type": "CLIPVisionLoader",
            "inputs": {"clip_name": "clip_vision_h14.safetensors"},
        }

        # 裁剪参考图的局部区域
        ref_image_source = ["3", 0]
        if ref_crop:
            rx, ry, rw, rh = ref_crop
            wf["12"] = {
                "class_type": "ImageCrop",
                "inputs": {
                    "image": ["3", 0],
                    "x": rx, "y": ry,
                    "width": rw, "height": rh,
                },
            }
            ref_image_source = ["12", 0]

        # IP-Adapter Advanced
        wf["14"] = {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["10", 0],
                "image": ref_image_source,
                "weight": ref_weight,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only",
                "clip_vision": ["11", 0],
            },
        }
        model_source = ["14", 0]

    # --- KSampler ---
    wf["30"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_source,
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["20", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": denoise,
        },
    }

    # --- 解码 & 保存 ---
    wf["40"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["30", 0], "vae": ["1", 2]},
    }
    wf["50"] = {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": prefix, "images": ["40", 0]},
    }

    return wf


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ComfyUI 局部重绘工具 — 支持参考图引用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础重绘（手动画蒙版）
  python inpaint_tool.py --source photo.png --draw-mask --prompt "muscular arms"

  # 提供现有蒙版
  python inpaint_tool.py --source photo.png --mask mask.png --prompt "six pack abs"

  # 用参考图引导（整张参考图）
  python inpaint_tool.py --source photo.png --mask mask.png --prompt "muscular chest" \\
      --reference ref.png --ref-weight 0.7

  # 用参考图的局部区域引导
  python inpaint_tool.py --source photo.png --mask mask.png --prompt "muscular chest" \\
      --reference ref.png --ref-crop 100,200,256,256 --ref-weight 0.7

  # 微调参数
  python inpaint_tool.py --source photo.png --mask mask.png --prompt "..." \\
      --denoise 0.6 --steps 30 --cfg 8 --seed 12345
        """,
    )

    parser.add_argument("--source", required=True, help="源图片路径（要修改的图）")
    parser.add_argument("--mask", help="蒙版路径（白色=修改区域，黑色=保留区域）")
    parser.add_argument("--draw-mask", action="store_true", help="打开画笔工具手动画蒙版")
    parser.add_argument("--prompt", required=True, help="正面提示词（控制修改方向）")
    parser.add_argument("--negative", default="low quality, blurry, bad anatomy, deformed, ugly, watermark, text",
                       help="负面提示词")

    # 参考图选项
    parser.add_argument("--reference", help="参考图路径（可选，提供外观/风格参考）")
    parser.add_argument("--ref-crop", help="裁剪参考图的区域: x,y,width,height（如 100,200,256,256）")
    parser.add_argument("--ref-weight", type=float, default=0.65,
                       help="参考图权重 0.0-1.0（默认0.65，越高越接近参考图）")

    # 生成参数
    parser.add_argument("--denoise", type=float, default=0.85,
                       help="去噪强度 0.0-1.0（默认0.85，越高改动越大）")
    parser.add_argument("--grow-mask", type=int, default=8,
                       help="蒙版边缘扩展像素（默认8，用于平滑过渡）")
    parser.add_argument("--steps", type=int, default=25, help="采样步数（默认25）")
    parser.add_argument("--cfg", type=float, default=7.0, help="CFG 比例（默认7.0）")
    parser.add_argument("--seed", type=int, default=-1, help="随机种子（-1=随机）")
    parser.add_argument("--model", default="dreamshaper_8.safetensors", help="模型名")
    parser.add_argument("--prefix", default="inpaint_result", help="输出文件名前缀")

    args = parser.parse_args()

    # 检查服务器
    if not check_server():
        print("ComfyUI 未运行！请先启动：")
        print('  cd "C:\\Users\\Yusheng Ding\\Desktop\\projects\\ComfyUI"')
        print("  py -3.12 main.py --listen 127.0.0.1 --port 8188")
        sys.exit(1)

    print(f"ComfyUI 已连接: {BASE_URL}")

    # 检查源图
    if not os.path.exists(args.source):
        print(f"源图片不存在: {args.source}")
        sys.exit(1)

    # 处理蒙版
    mask_path = args.mask
    if args.draw_mask:
        mask_path = os.path.splitext(args.source)[0] + "_mask.png"
        print("打开蒙版编辑器...")
        print("  左键画白色（修改区域）| 右键擦除 | 滚轮调笔刷 | Enter保存")
        draw_mask_interactive(args.source, mask_path)

    if not mask_path or not os.path.exists(mask_path):
        print("需要蒙版文件！用 --mask 指定或 --draw-mask 手动画")
        sys.exit(1)

    # 上传图片
    print(f"上传源图片: {args.source}")
    source_name = upload_image(args.source)

    print(f"上传蒙版: {mask_path}")
    mask_name = upload_image(mask_path)

    ref_name = None
    ref_crop = None
    if args.reference:
        if not os.path.exists(args.reference):
            print(f"参考图不存在: {args.reference}")
            sys.exit(1)
        print(f"上传参考图: {args.reference}")
        ref_name = upload_image(args.reference)

        if args.ref_crop:
            parts = [int(x) for x in args.ref_crop.split(",")]
            if len(parts) != 4:
                print("--ref-crop 格式: x,y,width,height")
                sys.exit(1)
            ref_crop = tuple(parts)
            print(f"裁剪参考图区域: x={parts[0]}, y={parts[1]}, {parts[2]}x{parts[3]}")

    # 构建工作流
    print(f"\n--- 生成参数 ---")
    print(f"  Prompt:    {args.prompt}")
    print(f"  Denoise:   {args.denoise}")
    print(f"  Steps:     {args.steps}")
    print(f"  CFG:       {args.cfg}")
    print(f"  Seed:      {args.seed if args.seed >= 0 else '随机'}")
    if ref_name:
        print(f"  参考图:    {args.reference}")
        print(f"  参考权重:  {args.ref_weight}")
    print()

    workflow = build_inpaint_workflow(
        source_name=source_name,
        mask_name=mask_name,
        prompt=args.prompt,
        negative_prompt=args.negative,
        reference_name=ref_name,
        ref_crop=ref_crop,
        ref_weight=args.ref_weight,
        denoise=args.denoise,
        grow_mask=args.grow_mask,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        model=args.model,
        prefix=args.prefix,
    )

    # 提交生成
    print("提交生成任务...")
    prompt_id = queue_prompt(workflow)
    print(f"任务 ID: {prompt_id}")
    print("等待生成完成...")

    images = wait_result(prompt_id)

    print(f"\n生成完成！输出 {len(images)} 张图片:")
    for img in images:
        print(f"  {img}")


if __name__ == "__main__":
    main()
