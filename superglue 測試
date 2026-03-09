import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.cm as cm

# === 加入 superglue 模型路徑 ===
superglue_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "superglue_lib"))
sys.path.append(superglue_path)

from models.matching import Matching
from models.utils import read_image, make_matching_plot

# === 設定圖片與輸出路徑 ===
img0_path = Path(r"D:\vscode\D-project\test\SuperGluePretrainedNetwork-master\test_file\test_photo\match_test_respiberry.jpg")
img1_path = Path(r"D:\vscode\D-project\test\SuperGluePretrainedNetwork-master\test_file\test_photo\match_test_cesium.png")
output_path = Path(r"D:\vscode\D-project\test\SuperGluePretrainedNetwork-master\output\SuperGluetest.png")
output_path.parent.mkdir(exist_ok=True)

# === 初始化模型 ===
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
print(f"✅ 使用設備：{device}")

matching = Matching(config).eval().to(device)

# === 讀圖並 resize 成 640x480 ===
image0, inp0, _ = read_image(img0_path, device, [640, 480], 0, False)
image1, inp1, _ = read_image(img1_path, device, [640, 480], 0, False)

# === 開始匹配 ===
pred = matching({'image0': inp0, 'image1': inp1})
pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}

kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
matches = pred['matches0']
conf = pred['matching_scores0']

# === 篩選有效匹配點 ===
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
mconf = conf[valid]

# === 印出前 10 高信心值的匹配點 ===
top_k = 10
top_idx = np.argsort(-mconf)[:top_k]

print(f"\n✅ 匹配完成，總匹配點數：{len(mkpts0)}，前 {top_k} 筆如下：")
for i in top_idx:
    print(f"[{i+1}] img1: {mkpts0[i]} -> img2: {mkpts1[i]} | confidence: {mconf[i]:.4f}")

# === 匯出視覺化圖片 ===
color = cm.jet(mconf[top_idx])
make_matching_plot(
    image0, image1, kpts0, kpts1, mkpts0[top_idx], mkpts1[top_idx], color,
    ['SuperGlue Matching'], output_path,
    show_keypoints=False, fast_viz=False, opencv_display=False
)

print(f"\n🖼️ 匹配圖已儲存至：{output_path.resolve()}")
