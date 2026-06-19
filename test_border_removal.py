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
    # 大雑把に数字の位置（および下部の手書き黒文字等）をクロップするため y_end=0.25 に拡大
    x1, x2 = int(w * 0.0), int(w * 0.30)
    y1, y2 = int(h * 0.04), int(h * 0.25)
    return img[y1:y2, x1:x2]

def keep_only_blue(cropped_img):
    """
    HSV色空間を用いて、青系（シアン〜ブルー）以外の色をすべて白色で塗りつぶす
    """
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 青系（シアン〜ブルー）の範囲を定義
    # H: 75〜140 (シアン〜ブルー)
    # S: 35〜255 (ある程度の鮮やかさ。滲みも考慮して少し低めからカバー)
    # V: 50〜255 (ある程度の明るさ)
    is_blue = (h >= 75) & (h <= 145) & (s >= 35) & (v >= 50)
    
    # 白ベースの画像を作成し、青いピクセルだけをコピーする
    filtered_img = np.ones_like(cropped_img) * 255
    filtered_img[is_blue] = cropped_img[is_blue]
    
    return filtered_img

def remove_borders_hough(cropped_img):
    """
    ハフ変換等を利用した枠線（直線）の直接検出と除去
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # 二値化 (適応的二値化で文字を浮かび上がらせる)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    # エッジ検出を行い、ハフ変換で長い直線（枠線候補）を検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=25, maxLineGap=10)
    
    mask = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 水平（角度が0付近）または垂直（角度が90付近）な線のみを対象とする
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10 or angle > 80:
                # 検出された直線部分をマスクに太さ3で描画
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
                
    # 1. 二値化画像からマスクを引く（単純な除去）
    binary_removed = cv2.subtract(binary, mask)
    
    # 2. 原画像に対してインペイント（修復）を行い、再二値化する（滑らかな除去）
    inpainted = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)
    inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
    inpainted_blurred = cv2.GaussianBlur(inpainted_gray, (5, 5), 0)
    inpainted_binary = cv2.adaptiveThreshold(
        inpainted_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    return binary, mask, binary_removed, inpainted, inpainted_binary

def align_and_subtract(cropped_img, template_gray):
    """
    特徴点マッチング (ORB) を使用して正確な位置合わせを行った後のテンプレート差分
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    # テンプレートを一旦同じ解像度にリサイズ
    template_resized = cv2.resize(template_gray, (gray.shape[1], gray.shape[0]))
    
    # ORB特徴量によるマッチングと射影変換
    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(template_resized, None)
    kp2, des2 = orb.detectAndCompute(gray, None)
    
    if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = matches[:50]
        if len(good_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 射影変換でアライメント
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                aligned_template = cv2.warpPerspective(template_resized, H, (gray.shape[1], gray.shape[0]), borderValue=255)
                diff = cv2.absdiff(gray, aligned_template)
                return aligned_template, diff

    # フォールバック (マッチング失敗時は普通にリサイズ差分)
    diff = cv2.absdiff(gray, template_resized)
    return template_resized, diff

def crop_digits_from_binary(cropped_img, binary_img, stem):
    """
    枠線除去済みの二値化画像 (binary_img) から数字領域を検出してクロップする
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 輪郭検出
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * 0.005   # クロップ範囲の拡大に合わせて閾値を緩和
    max_area = (h * w) * 0.8

    normal_regions = []
    wide_regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect_ratio = ch / max(cw, 1)

        if not (min_area < area < max_area and ch > h * 0.15):
            continue
        if aspect_ratio >= 5.0:
            continue  # 細長い縦線を除外

        if aspect_ratio >= 0.5:
            normal_regions.append((x, y, cw, ch))
        elif aspect_ratio >= 0.20:
            wide_regions.append((x, y, cw, ch))

    # 横長領域の分割処理 (extract_number.py から移植)
    for (wx, wy, ww, wh) in wide_regions:
        contained = [r for r in normal_regions if r[0] >= wx and r[0] + r[2] <= wx + ww]

        if len(contained) >= 2:
            pass
        else:
            for c in contained:
                if c in normal_regions:
                    normal_regions.remove(c)

            if normal_regions:
                widths = sorted([r[2] for r in normal_regions])
                median_width = widths[len(widths) // 2]
            else:
                median_width = ww // 2

            region_binary = binary_img[wy:wy + wh, wx:wx + ww]
            v_proj = np.sum(region_binary > 0, axis=0)

            threshold = max(v_proj) * 0.20 if max(v_proj) > 0 else 0
            in_valley = v_proj <= threshold
            split_points = []
            valley_start = None
            for col_idx in range(len(in_valley)):
                if in_valley[col_idx] and valley_start is None:
                    valley_start = col_idx
                elif not in_valley[col_idx] and valley_start is not None:
                    split_points.append((valley_start + col_idx) // 2)
                    valley_start = None

            if split_points:
                boundaries = [0] + split_points + [ww]
                for j in range(len(boundaries) - 1):
                    sx = boundaries[j]
                    sw = boundaries[j + 1] - sx
                    if sw > median_width * 0.3:
                        normal_regions.append((wx + sx, wy, sw, wh))
            else:
                if median_width > 0:
                    num_digits = round(ww / median_width)
                    num_digits = max(2, min(num_digits, 5))
                else:
                    num_digits = 2
                single_w = ww // num_digits
                for k in range(num_digits):
                    split_x = wx + k * single_w
                    split_w = single_w if k < num_digits - 1 else (ww - k * single_w)
                    normal_regions.append((split_x, wy, split_w, wh))

    digit_regions = normal_regions

    # NMS（非最大値抑制）による重複マージ
    digit_regions.sort(key=lambda r: r[0])
    merged = []
    for region in digit_regions:
        rx, ry, rw, rh = region
        if not merged:
            merged.append(region)
            continue

        px, py, pw, ph = merged[-1]
        overlap_x = max(0, min(px + pw, rx + rw) - max(px, rx))
        min_width = min(pw, rw)

        if overlap_x > min_width * 0.3:
            prev_score = abs(ph / max(pw, 1) - 1.5)
            curr_score = abs(rh / max(rw, 1) - 1.5)
            if curr_score < prev_score:
                merged[-1] = region
        else:
            merged.append(region)

    digit_regions = merged
    digit_regions.sort(key=lambda r: r[0])

    # 1. 検出領域を描画したデバッグ画像を保存
    debug_img = cropped_img.copy()
    for i, (x, y, cw, ch) in enumerate(digit_regions):
        cv2.rectangle(debug_img, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
        cv2.putText(debug_img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.imwrite(str(DEBUG_DIR / f"{stem}_2f_detected_regions.png"), debug_img)

    # 2. 各数字をクロップして保存
    for i, (x, y, cw, ch) in enumerate(digit_regions):
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        
        # グレースケールと二値化の両方を保存
        digit_crop_gray = gray[y1:y2, x1:x2]
        digit_crop_bin = binary_img[y1:y2, x1:x2]
        
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2g_digit_{i}_gray.png"), digit_crop_gray)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2g_digit_{i}_bin.png"), digit_crop_bin)

    return len(digit_regions)

def main():
    print("テンプレート画像を読み込み中...")
    template_img = pdf_to_image("template.pdf")
    template_cropped = crop_top_left(template_img)
    template_gray = cv2.cvtColor(template_cropped, cv2.COLOR_BGR2GRAY)
    
    # テスト対象のPDFファイル
    test_files = ["30000.pdf", "30001.pdf"]
    
    for filename in test_files:
        pdf_path = os.path.join("./image", filename)
        if not os.path.exists(pdf_path):
            print(f"ファイルが見つかりません: {pdf_path}")
            continue
            
        print(f"\n--- 処理中: {filename} ---")
        img = pdf_to_image(pdf_path)
        
        # クロップ前全体画像を保存
        stem = Path(filename).stem
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_0_raw.png"), img)
        
        # クロップ
        cropped = crop_top_left(img)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_1_cropped.png"), cropped)
        
        # カラーフィルタ (青系以外の色を除去)
        filtered = keep_only_blue(cropped)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_1b_color_filtered.png"), filtered)
        
        # アルゴリズム1: ハフ変換（直線検出）による除去 (カラーフィルタ適用後の画像を使用)
        binary, mask, bin_removed, inpainted, inpainted_binary = remove_borders_hough(filtered)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2a_adaptive_binary.png"), binary)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2b_border_mask.png"), mask)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2c_subtract_removed.png"), bin_removed)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2d_inpainted_color.png"), inpainted)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_2e_inpainted_binary.png"), inpainted_binary)
        
        # 追加: 2eの結果から数字を切り出す (カラーフィルタ適用後の cropped を使用)
        num_cropped = crop_digits_from_binary(filtered, inpainted_binary, stem)
        print(f"  -> [2e] から {num_cropped} 個の数字をクロップしました。")
        
        # アルゴリズム2: 特徴点アライメント+差分 (カラーフィルタ適用後の画像を使用)
        aligned_template, diff = align_and_subtract(filtered, template_gray)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_3a_aligned_template.png"), aligned_template)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_3b_aligned_diff.png"), diff)
        
        # 差分画像の二値化 (大津の二値化)
        _, diff_binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(str(DEBUG_DIR / f"{stem}_3c_aligned_diff_binary.png"), diff_binary)
        
        print(f"{filename} のデバッグ画像を出力しました。")

if __name__ == "__main__":
    main()
