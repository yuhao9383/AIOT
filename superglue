import torch
import numpy as np
import time
import argparse
from pathlib import Path
import matplotlib.cm as cm
from PIL import Image
import json
import os
import sys

# 加入 SuperGlue 模組路徑
superglue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "superglue_lib"))
sys.path.append(superglue_path)

from models.matching import Matching
from models.utils import read_image, make_matching_plot


def init_model():
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024,
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 初始化模型中...（device: {device}）")
    matching = Matching(config).eval().to(device)
    return matching, device


def save_json_and_optional_plot(image0, image1, kpts0, kpts1,
                                 mkpts0, mkpts1, mconf,
                                 output_path, draw_plot=False, top_k='all',
                                 scale0=(1.0, 1.0), scale1=(1.0, 1.0)):
    viz_path = Path(output_path)

    # 還原原圖座標
    mkpts0_orig = mkpts0.copy()
    mkpts1_orig = mkpts1.copy()
    mkpts0_orig[:, 0] *= scale0[0]
    mkpts0_orig[:, 1] *= scale0[1]
    mkpts1_orig[:, 0] *= scale1[0]
    mkpts1_orig[:, 1] *= scale1[1]

    # 依信心值排序
    sorted_indices = np.argsort(-mconf)
    mkpts0_orig = mkpts0_orig[sorted_indices]
    mkpts1_orig = mkpts1_orig[sorted_indices]
    mconf_sorted = mconf[sorted_indices]

    # 輸出 JSON
    json_path = viz_path.with_suffix('.json')
    match_data = []
    for pt0, pt1, conf in zip(mkpts0_orig, mkpts1_orig, mconf_sorted):
        match_data.append({
            'x0': round(float(pt0[0]), 2),
            'y0': round(float(pt0[1]), 2),
            'x1': round(float(pt1[0]), 2),
            'y1': round(float(pt1[1]), 2),
            'confidence': round(float(conf), 4)
        })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(match_data, f, indent=2, ensure_ascii=False)
    print(f"📝 匹配資訊已輸出（JSON）：{json_path}")

    # ➕ 新增一份備份或指定名稱的 JSON 檔案
    # 例如：原始是 imgA_imgB_matches.json，備份是 imgA_imgB_pixel_pairs.json
    json_path_alt = json_path.with_name(json_path.stem.replace("_matches", "_pixel_pairs") + ".json")
    with open(json_path_alt, 'w', encoding='utf-8') as f:
        json.dump(match_data, f, indent=2, ensure_ascii=False)
    print(f"📁 匹配資料也儲存至備份檔：{json_path_alt}")
    # 可選：畫圖
    if draw_plot:
        if isinstance(top_k, int) and top_k < len(mconf):
            top_indices = sorted_indices[:top_k]
            mkpts0_vis = mkpts0[top_indices]
            mkpts1_vis = mkpts1[top_indices]
            mconf_vis = mconf[top_indices]
        else:
            mkpts0_vis = mkpts0
            mkpts1_vis = mkpts1
            mconf_vis = mconf

        color = cm.jet(mconf_vis)
        text = [
            'SuperGlue',
            f'Matches shown: {len(mkpts0_vis)}'
        ]
        small_text = [
            'Keypoint Threshold: 0.005',
            'Match Threshold: 0.2',
        ]
        make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0_vis, mkpts1_vis, color,
            text, viz_path, show_keypoints=False,
            fast_viz=False, opencv_display=False,
            opencv_title='Matches', small_text=small_text
        )
        print(f"🖼️ 匹配圖已輸出：{viz_path}")


def run_matching(matching, device, img0_path, img1_path,
                 enable_viz=False, top_k='all', output_dir=Path("output")):
    print(f"\n🔍 開始匹配：\n  圖片1: {img0_path}\n  圖片2: {img1_path}")

    # ✅ 確保 output_dir 存在（如果沒有就建立）
    output_dir.mkdir(parents=True, exist_ok=True)
    w0, h0 = Image.open(img0_path).size
    w1, h1 = Image.open(img1_path).size

    resize_wh = [640, 480]
    image0, inp0, _ = read_image(img0_path, device, resize_wh, 0, False)
    image1, inp1, _ = read_image(img1_path, device, resize_wh, 0, False)

    if image0 is None or image1 is None:
        print("❌ 圖片讀取失敗")
        return

    scale0 = (w0 / resize_wh[0], h0 / resize_wh[1])
    scale1 = (w1 / resize_wh[0], h1 / resize_wh[1])

    start_time = time.time()
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
    elapsed = time.time() - start_time

    matches = pred['matches0']
    conf = pred['matching_scores0']
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    print(f"✅ 匹配完成！成功匹配點數：{len(mkpts0)}")
    print(f"⏱️ 匹配時間：{elapsed:.4f} 秒")

    img0_name = Path(img0_path).stem
    img1_name = Path(img1_path).stem
    output_path = output_dir / f"{img0_name}_{img1_name}_matches.png"

    save_json_and_optional_plot(
        image0, image1, kpts0, kpts1,
        mkpts0, mkpts1, mconf,
        output_path, draw_plot=enable_viz, top_k=top_k,
        scale0=scale0, scale1=scale1
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', action='store_true', help='是否輸出視覺化圖（.png）')
    args = parser.parse_args()

    matching, device = init_model()
    print("✅ 模型初始化完成，開始互動模式！")

    # 輸出資料夾設定
    output_dir_input = input("請輸入輸出資料夾路徑（預設為 output）：").strip()
    if output_dir_input == '':
        output_dir = Path("output")
    else:
        output_dir = Path(output_dir_input)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top-k 設定
    top_k_input = input("請輸入要顯示的匹配線條數量（輸入 all 表示顯示全部）： ").strip()
    if top_k_input.lower() == 'all':
        top_k = 'all'
    elif top_k_input.isdigit():
        top_k = int(top_k_input)
    else:
        print("⚠️ 輸入錯誤，預設為 all")
        top_k = 'all'

    while True:
        img0 = input("\n請輸入圖片1路徑（或輸入 'q' 離開）： ").strip()
        if img0.lower() == 'q':
            break
        img1 = input("請輸入圖片2路徑： ").strip()
        run_matching(matching, device, img0, img1,
                     enable_viz=args.viz, top_k=top_k, output_dir=output_dir)
