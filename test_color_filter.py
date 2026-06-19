import os
import cv2
import numpy as np
import fitz
from pathlib import Path

# 出力先ディレクトリ
DEBUG_DIR = Path("c:/Users/km26805/Documents/ocr/debug_output")
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def pdf_to_image(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[0]
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img

def crop_top_left(img):
    h, w = img.shape[:2]
    x1, x2 = int(w * 0.0), int(w * 0.30)
    y1, y2 = int(h * 0.04), int(h * 0.25)
    return img[y1:y2, x1:x2]

def remove_black_and_red(cropped_img):
    """
    HSV色空間を用いて、黒系（手書き文字など）と赤系（赤スタンプなど）を白色で塗りつぶす
    """
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 1. 黒系マスク (明度が低く、かつ青系の鮮やかな色ではない部分)
    # 青系のHSV範囲: H = 85〜135 程度。彩度 S が高い青は除外したい。
    # ここでは単純に明度 V が低く (V < 120)、かつ青の範囲 (H=85-135, S>70) ではない領域を黒とする
    is_blue = (h >= 85) & (h <= 135) & (s >= 70)
    black_mask = (v < 130) & (~is_blue)
    
    # 2. 赤系マスク (Hが0-10付近または170-180付近で、一定の鮮やかさがある部分)
    red_mask1 = (h >= 0) & (h <= 10) & (s >= 50) & (v >= 50)
    red_mask2 = (h >= 170) & (h <= 180) & (s >= 50) & (v >= 50)
    red_mask = red_mask1 | red_mask2
    
    # マスクの結合
    removal_mask = black_mask | red_mask
    
    # マスク箇所を白色 [255, 255, 255] で塗りつぶす
    filtered_img = cropped_img.copy()
    filtered_img[removal_mask > 0] = [255, 255, 255]
    
    return filtered_img, black_mask, red_mask, removal_mask

def main():
    test_files = ["30000.pdf", "30001.pdf"]
    
    for filename in test_files:
        pdf_path = os.path.join("./image", filename)
        if not os.path.exists(pdf_path):
            print(f"ファイルが見つかりません: {pdf_path}")
            continue
            
        print(f"\n--- カラーフィルタ処理中: {filename} ---")
        img = pdf_to_image(pdf_path)
        cropped = crop_top_left(img)
        
        stem = Path(filename).stem
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_color_1_cropped.png"), cropped)
        
        # 黒系・赤系の除去
        filtered, black_m, red_m, total_m = remove_black_and_red(cropped)
        
        # デバッグ保存
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_color_2a_black_mask.png"), (black_m * 255).astype(np.uint8))
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_color_2b_red_mask.png"), (red_m * 255).astype(np.uint8))
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_color_3_filtered.png"), filtered)
        
        # フィルタ後の画像を二値化（適応的二値化で青文字を抽出）
        filtered_gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        filtered_blurred = cv2.GaussianBlur(filtered_gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            filtered_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )
        
        # 枠線（青線）は残るが、Hough変換や輪郭フィルタで枠を消す処理と組み合わせることが可能
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_color_4_binary.png"), binary)
        
        print(f"{filename} のカラーフィルタ画像を保存しました。")

if __name__ == "__main__":
    main()
