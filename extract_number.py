"""
画像・PDFの左上にある登録番号を OCR で読み取り、
その番号をファイル名にして画像を保存するスクリプト。

TensorFlow を使用（MNIST ベースの数字認識）。

処理フロー:
  1. PDF の場合は PyMuPDF で画像に変換
  2. OpenCV で画像左上を切り出し
  3. 二値化・輪郭検出で個々の数字領域を抽出
  4. TensorFlow/Keras の MNIST 学習済みモデルで各数字を分類
  5. 認識した番号をファイル名にして保存

使い方:
  python extract_number.py image.jpg
  python extract_number.py document.pdf
  python extract_number.py --input-dir ./images
  python extract_number.py --input-dir ./images --output-dir ./renamed

必要なライブラリ:
  pip install tensorflow opencv-python-headless Pillow numpy PyMuPDF
"""

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
import time

import cv2
import numpy as np

# XLA JIT コンパイルとオートチューニングを無効化 (RTX 4060 Ti + cuDNN 9.x での autotuner エラー回避)
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Linux/WSLでTensorFlowがCUDAライブラリを見つけられるように、環境変数を自動設定して再起動する
if os.name != "nt" and not os.environ.get("TF_GPU_RESOLVED"):
    try:
        import nvidia
        nvidia_base = Path(nvidia.__file__).parent
        lib_paths = [str(p) for p in nvidia_base.glob("*/lib") if p.is_dir()]
        if lib_paths:
            env = os.environ.copy()
            old_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(lib_paths) + (f":{old_ld}" if old_ld else "")
            env["TF_GPU_RESOLVED"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
    except ImportError:
        pass

# TensorFlow のログを抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# サポートする画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# サポートするファイル拡張子（PDF含む）
ALL_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

# MNIST モデルのキャッシュ
_model = None


def get_mnist_model():
    """
    MNIST で学習した数字認識モデルを取得（初回のみ学習）。
    """
    global _model
    if _model is not None:
        return _model

    model_path = Path(__file__).parent / "stamp_mnist_model.keras"

    if model_path.exists():
        print("  学習済みスタンプモデルをロード中...")
        _model = tf.keras.models.load_model(str(model_path))
    else:
        print("  MNIST モデルを学習中（初回のみ）...")
        # MNIST データのロード
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # 前処理: 正規化 + 次元追加
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

        # CNN モデル構築（深層版）
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),

            # ブロック1: 32フィルタ x2
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # ブロック2: 64フィルタ x2
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # ブロック3: 128フィルタ x2
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # 全結合層
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(x_train, y_train, epochs=10, batch_size=128,
                  validation_data=(x_test, y_test), verbose=1)

        # 精度確認
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"  モデル精度: {acc:.4f}")

        # モデル保存
        model.save(str(model_path))
        print(f"  モデルを保存: {model_path}")

        _model = model

    return _model

def crop_top_left(img: np.ndarray,
                   x_start: float = 0.0, x_end: float = 0.35,
                   y_start: float = 0.04, y_end: float = 0.25):
    """
    画像の左上部分（登録番号エリア）を切り出す。
    登録番号のすぐ下の手書き黒文字等を含む大雑把な位置で切り出すため、
    y_end をデフォルトで 25% まで広げて取得する。
    """
    h, w = img.shape[:2]
    x1 = int(w * x_start)
    x2 = int(w * x_end)
    y1 = int(h * y_start)
    y2 = int(h * y_end)
    cropped = img[y1:y2, x1:x2]
    return cropped

def keep_blue_and_black(cropped_img):
    """
    HSV色空間を用いて、青系（シアン〜ブルー）と黒系以外の色をすべて白色で塗りつぶす（赤系を抜く）
    """
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 青系（シアン〜ブルー）の範囲を定義 (H:75-145, S:35-255, V:50-255)
    is_blue = (h >= 75) & (h <= 145) & (s >= 35) & (v >= 50)
    
    # 黒系の範囲を定義 (明度Vが低いピクセル)
    is_black = (v < 100)
    
    # 青系・黒系以外の色（赤系・白に近い背景色など）を白色 [255, 255, 255] に置き換える
    keep_mask = is_blue | is_black
    filtered_img = np.ones_like(cropped_img) * 255
    filtered_img[keep_mask] = cropped_img[keep_mask]
    
    return filtered_img

def extract_digit_regions(cropped_img, debug_dir: Path | None = None,
                          image_name: str = "", template_gray=None,
                          min_area_ratio: float = 0.005,
                          disable_border_removal: bool = False,
                          hough_threshold: int = 30):
    """
    切り出し画像から直線検出（Hough変換）を用いて枠線マスクを作成し、
    二値化画像からマスクを直接 subtract することで枠線を除去し、
    個々の数字の領域を検出してクロップする。
    """
    # 青系・黒系以外の色（赤系など）をカラーフィルタで白色に塗りつぶす
    filtered_img = keep_blue_and_black(cropped_img)
    
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 二値化
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    # 1. 数字行の自動検出 (水平投影)
    h_proj = np.sum(binary > 0, axis=1)
    if np.max(h_proj) > 0:
        y_peak = np.argmax(h_proj)
        thresh = np.max(h_proj) * 0.10
        y_start_digit = y_peak
        while y_start_digit > 0 and h_proj[y_start_digit] > thresh:
            y_start_digit -= 1
        y_end_digit = y_peak
        while y_end_digit < h - 1 and h_proj[y_end_digit] > thresh:
            y_end_digit += 1
        y_center = (y_start_digit + y_end_digit) // 2
    else:
        y_center = h // 2
        y_start_digit = 0
        y_end_digit = h

    # Houghマスクの作成
    mask = np.zeros_like(gray)
    
    if not disable_border_removal:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=25, maxLineGap=10)
        
        if lines is not None:
            # 数字行の範囲内を通過する水平線を除外（数字の横棒や上下の曲線を保護）
            y_start_exclude = y_start_digit - 3
            y_end_exclude = y_end_digit + 3
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10:  # 水平線
                    line_y_center = (y1 + y2) / 2
                    if y_start_exclude <= line_y_center <= y_end_exclude:
                        continue
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
                elif angle > 80:  # 垂直線
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

    # 枠線除去: 二値化画像からマスクを引き算する
    binary_removed = cv2.subtract(binary, mask)

    # 2. 輪郭検出と数字領域の特定
    contours, _ = cv2.findContours(binary_removed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (h * w) * min_area_ratio
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
            continue

        if aspect_ratio >= 0.5:
            normal_regions.append((x, y, cw, ch))
        elif aspect_ratio >= 0.20:
            wide_regions.append((x, y, cw, ch))

    # Calculate median width of normal regions to guide split/peaks
    if normal_regions:
        median_width = np.median([r[2] for r in normal_regions])
    else:
        median_width = w * 0.05

    # 横長領域の分割処理 (投影ベースのピーク検出による高精度切り出し)
    for (wx, wy, ww, wh) in wide_regions:
        region_binary = binary_removed[wy:wy + wh, wx:wx + ww]
        v_proj = np.sum(region_binary > 0, axis=0)

        threshold = max(v_proj) * 0.15 if max(v_proj) > 0 else 0
        
        # 閾値を超える連続部分（ピークブロック）を抽出
        peaks = []
        in_peak = False
        peak_start = 0
        for col_idx in range(len(v_proj)):
            if v_proj[col_idx] > threshold:
                if not in_peak:
                    peak_start = col_idx
                    in_peak = True
            else:
                if in_peak:
                    peaks.append((peak_start, col_idx))
                    in_peak = False
        if in_peak:
            peaks.append((peak_start, len(v_proj)))

        # 各ピークブロックを処理
        for ps, pe in peaks:
            sw = pe - ps
            if sw > median_width * 1.4:
                # 複数数字が接触している場合は等分割
                num_splits = max(2, round(sw / median_width))
                split_w = sw / num_splits
                for k in range(num_splits):
                    normal_regions.append((wx + ps + int(k * split_w), wy, int(split_w), wh))
            elif sw > median_width * 0.3:
                normal_regions.append((wx + ps, wy, sw, wh))

    digit_regions = normal_regions
    digit_regions.sort(key=lambda r: r[0])

    # NMSによる重複マージ (縦横両方の重なりをチェックして異なる行の数字の誤マージを防ぐ)
    merged = []
    for region in digit_regions:
        rx, ry, rw, rh = region
        if not merged:
            merged.append(region)
            continue

        px, py, pw, ph = merged[-1]
        overlap_x = max(0, min(px + pw, rx + rw) - max(px, rx))
        overlap_y = max(0, min(py + ph, ry + rh) - max(py, ry))
        min_width = min(pw, rw)
        min_height = min(ph, rh)

        if overlap_x > min_width * 0.3 and overlap_y > min_height * 0.3:
            prev_score = abs(ph / max(pw, 1) - 1.5)
            curr_score = abs(rh / max(rw, 1) - 1.5)
            if curr_score < prev_score:
                merged[-1] = region
        else:
            merged.append(region)

    digit_regions = merged
    digit_regions.sort(key=lambda r: r[0])

    # Union-Find (Disjoint-Set) アルゴリズムを用いた位置近傍クラスタリング
    # 異なる行や手書き文字などの外れ値ノイズを頑健に排除し、メインのスタンプ文字行だけを抽出する
    if digit_regions:
        heights = [r[3] for r in digit_regions]
        median_height = np.median(heights)
        
        n = len(digit_regions)
        parent = list(range(n))
        
        def find_root(i):
            if parent[i] == i:
                return i
            parent[i] = find_root(parent[i])
            return parent[i]
            
        def union_nodes(i, j):
            root_i = find_root(i)
            root_j = find_root(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        # 縦横両方で近接している領域同士を連結
        for i in range(n):
            for j in range(i + 1, n):
                r1 = digit_regions[i]
                r2 = digit_regions[j]
                
                # 横方向の隙間
                x1_min, x1_max = r1[0], r1[0] + r1[2]
                x2_min, x2_max = r2[0], r2[0] + r2[2]
                gap_x = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
                
                # 縦方向の中心座標ズレ
                cy1 = r1[1] + r1[3]/2
                cy2 = r2[1] + r2[3]/2
                v_diff = abs(cy1 - cy2)
                
                if gap_x <= 1.5 * median_height and v_diff <= 0.4 * median_height:
                    union_nodes(i, j)
                    
        # 親ノード（ルート）ごとにクラスタを統合
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find_root(i)].append(digit_regions[i])
            
        clusters = list(groups.values())
        
        # 最も要素数（桁数）が多いクラスタを採用し、同数の場合はより左側（x最小）を優先する
        clusters.sort(key=lambda c: (len(c), -c[0][0]), reverse=True)
        digit_regions = clusters[0]
        digit_regions.sort(key=lambda r: r[0])

    # 左右端および隙間のインクスキャンによる未処理インク検出
    ink_warning = False
    if digit_regions:
        x_left = digit_regions[0][0]
        x_right = digit_regions[-1][0] + digit_regions[-1][2]
        
        v_proj = np.sum(binary_removed > 0, axis=0)
        
        # 1. 左側のスキャン (0 から x_left - 8)
        left_margin = max(0, x_left - 8)
        for col in range(left_margin):
            if v_proj[col] >= h * 0.15:
                ink_warning = True
                break
                
        # 2. 右側のスキャン (x_right + 8 から w)
        if not ink_warning:
            right_margin = min(w, x_right + 8)
            for col in range(right_margin, w):
                if v_proj[col] >= h * 0.15:
                    ink_warning = True
                    break
                    
        # 3. 隙間（内側）のスキャン
        if not ink_warning:
            for i in range(len(digit_regions) - 1):
                gap_left = digit_regions[i][0] + digit_regions[i][2] + 4
                gap_right = digit_regions[i+1][0] - 4
                if gap_left < gap_right:
                    for col in range(gap_left, gap_right):
                        if v_proj[col] >= h * 0.15:
                            ink_warning = True
                            break
                    if ink_warning:
                        break
    else:
        ink_warning = True

    # 実際の数字画像を切り出し
    digit_images = []
    for i, (x, y, cw, ch) in enumerate(digit_regions):
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        digit_img = gray[y1:y2, x1:x2]
        digit_images.append(digit_img)

    return digit_images, digit_regions, ink_warning, binary_removed

def preprocess_for_mnist(digit_img):
    """
    数字画像を MNIST 形式 (28x28, 白文字黒背景) に変換する。
    """
    # 二値化（白文字 on 黒背景）
    _, binary = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 白のピクセルが少なければ反転が不要（既に白文字黒背景）
    white_ratio = np.sum(binary > 128) / binary.size
    if white_ratio > 0.5:
        binary = 255 - binary

    # 数字部分を中央に配置するためにバウンディングボックスで切り出し
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    digit_crop = binary[y:y+h, x:x+w]

    # 正方形にパディング
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2
    padded = cv2.copyMakeBorder(
        digit_crop, pad_h, pad_h, pad_w, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )

    # さらに余白を追加（MNIST は数字の周りに余白がある）
    margin = max(max_dim // 4, 4)
    padded = cv2.copyMakeBorder(
        padded, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=0
    )

    # 28x28 にリサイズ
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    # 正規化
    normalized = resized.astype("float32") / 255.0

    return normalized

def extract_number(image_path: str, model, debug_dir: Path | None = None,
                   template_gray=None) -> tuple[str, dict] | None:
    """
    画像の左上から登録番号（数字）を抽出する（マルチパス・リトライロジック）。
    戻り値: (認識した番号文字列, ベストパスの付加情報辞書) または None
    """
    image_name = Path(image_path).name
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ❌ 画像を読み込めませんでした: {image_path}")
        return None

    # 通常方向と180度回転方向を定義
    orientations = [
        {"name": "Normal", "img": img, "rotated": False},
        {"name": "Rotated_180", "img": cv2.rotate(img, cv2.ROTATE_180), "rotated": True}
    ]

    # リトライパラメータの定義
    passes = [
        # Pass 1: 標準パラメータ (バイナリサブトラクト)
        {"y_start": 0.04, "x_end": 0.35, "min_area_ratio": 0.005, "disable_border_removal": False, "hough_threshold": 30},
        # Pass 2: クロップ幅拡張 + 枠線除去緩和
        {"y_start": 0.04, "x_end": 0.38, "min_area_ratio": 0.003, "disable_border_removal": False, "hough_threshold": 45},
        # Pass 3: 枠線除去完全無効化 (y_start は 0.04 に設定して先頭「3」の脱落を防止)
        {"y_start": 0.04, "x_end": 0.40, "min_area_ratio": 0.003, "disable_border_removal": True, "hough_threshold": 30}
    ]
    
    candidates = []
    
    for orient in orientations:
        current_img = orient["img"]
        is_rotated = orient["rotated"]
        orient_name = orient["name"]
        
        for idx, params in enumerate(passes):
            try:
                cropped = crop_top_left(current_img, y_start=params["y_start"], x_end=params["x_end"])
            except Exception as e:
                print(f"  [{orient_name} Pass {idx+1}] クロップエラー: {e}")
                continue
                
            digit_images, digit_regions, ink_warning, binary_removed = extract_digit_regions(
                cropped, debug_dir=None, image_name=image_name,
                min_area_ratio=params["min_area_ratio"],
                disable_border_removal=params["disable_border_removal"],
                hough_threshold=params["hough_threshold"]
            )
            
            if not digit_images:
                continue
                
            recognized_digits = []
            confidences = []
            has_low_conf = False
            
            for i, digit_img in enumerate(digit_images):
                processed = preprocess_for_mnist(digit_img)
                if processed is None:
                    continue
                    
                input_data = np.expand_dims(np.expand_dims(processed, axis=-1), axis=0)
                prediction = model.predict(input_data, verbose=0)
                digit = np.argmax(prediction[0])
                confidence = prediction[0][digit]
                
                recognized_digits.append(str(digit))
                confidences.append(confidence)
                
                if confidence <= 0.35:
                    has_low_conf = True
                    
            if not recognized_digits:
                continue
                
            num_digits = len(recognized_digits)
            avg_confidence = np.mean(confidences)
            result_str = "".join(recognized_digits)
            
            candidates.append({
                "orient_name": orient_name,
                "pass_idx": idx + 1,
                "number": result_str,
                "num_digits": num_digits,
                "avg_confidence": avg_confidence,
                "has_low_conf": has_low_conf,
                "ink_warning": ink_warning,
                "cropped": cropped,
                "digit_images": digit_images,
                "digit_regions": digit_regions,
                "binary_removed": binary_removed,
                "confidences": confidences,
                "rotated": is_rotated
            })
        
    if not candidates:
        return None
        
    # ソートして最良の候補を選択
    # 優先順位:
    # 1. 検出された文字数 (num_digits) が6以下であること (最大桁数制約)
    # 2. 空間チェック（インクスキャン）警告がないこと (余計なノイズや端切れを回避)
    # 3. 低信頼度の数字（信頼度 <= 0.35）が含まれていないこと
    # 4. 検出された文字数が多いこと (途中切れ・切り捨てを防止)
    # 5. 平均信頼度が高いこと
    candidates.sort(key=lambda c: (
        c["num_digits"] <= 6,
        c["num_digits"],
        not c["ink_warning"],
        c["avg_confidence"]
    ), reverse=True)
    best = candidates[0]
    
    print(f"  => Best Result: {best['number']} ({best['orient_name']} Pass {best['pass_idx']}, 桁数: {best['num_digits']}, 平均信頼度: {best['avg_confidence']:.4f}, 低信頼度有: {best['has_low_conf']}, インク警告: {best['ink_warning']})")
    
    # ベスト候補のデバッグ画像を保存
    if debug_dir is not None:
        base = Path(image_name).stem
        
        # 検出領域
        debug_img = best["cropped"].copy()
        for i, (x, y, cw, ch) in enumerate(best["digit_regions"]):
            cv2.rectangle(debug_img, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            cv2.putText(debug_img, f"{best['number'][i]}({best['confidences'][i]:.2f})", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(debug_dir / f"{base}_3_detected_regions.png"), debug_img)
        
        # MNIST入力画像の保存
        for i, digit_img in enumerate(best["digit_images"]):
            processed = preprocess_for_mnist(digit_img)
            if processed is not None:
                mnist_img = (processed * 255).astype(np.uint8)
                cv2.imwrite(str(debug_dir / f"{base}_5_mnist_input_{i}.png"), mnist_img)
                
        # バイナリ消去結果の保存
        cv2.imwrite(str(debug_dir / f"{base}_2c_subtract_removed.png"), best["binary_removed"])
        
    return best["number"], {
        "avg_confidence": best["avg_confidence"],
        "ink_warning": best["ink_warning"],
        "num_digits": best["num_digits"],
        "has_low_conf": best["has_low_conf"],
        "rotated": best["rotated"]
    }

def save_rotated(src_path: Path, dest_path: Path):
    """
    ファイルを180度回転させて保存する。
    PDFの場合は各ページを180度回転、画像の場合は cv2 で回転して保存。
    """
    ext = src_path.suffix.lower()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(str(src_path))
        for page in doc:
            page.set_rotation((page.rotation + 180) % 360)
        doc.save(str(dest_path))
        doc.close()
        print(f"  🔄 [回転保存(PDF)] {dest_path.name} (元: {src_path.name})")
    elif ext in IMAGE_EXTENSIONS:
        img = cv2.imread(str(src_path))
        if img is not None:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(dest_path), rotated)
            print(f"  🔄 [回転保存(画像)] {dest_path.name} (元: {src_path.name})")
        else:
            shutil.copy2(src_path, dest_path)
            print(f"  ⚠️ [回転保存失敗(通常コピー)] {dest_path.name} (元: {src_path.name})")
    else:
        shutil.copy2(src_path, dest_path)

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """
    PDF ファイルを画像ファイルに変換する（PyMuPDF 使用）。
    一時ディレクトリに PNG として保存し、パスのリストを返す。
    """
    import fitz  # PyMuPDF

    temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
    image_paths = []

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        # DPI に応じたズーム倍率
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        stem = Path(pdf_path).stem
        output_path = os.path.join(temp_dir, f"{stem}_page{page_num + 1:04d}.png")
        pix.save(output_path)
        image_paths.append(output_path)

    doc.close()
    return image_paths


def process_image(image_path: Path, output_dir: Path, model,
                  debug_dir: Path | None = None, output_ext: str | None = None,
                  template_gray=None) -> bool:
    """
    1枚の画像を処理: 番号を読み取り → 番号.拡張子 で保存。
    （後方互換性および単一ファイル処理用のフォールバック）
    """
    print(f"処理中: {image_path.name}")

    try:
        res = extract_number(str(image_path), model, debug_dir=debug_dir,
                            template_gray=template_gray)
        if res is None:
            print("  ❌ 番号を検出できませんでした")
            return False
        number, _ = res
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return False

    ext = output_ext if output_ext else image_path.suffix.lower()
    output_path = output_dir / f"{number}{ext}"

    if output_path.exists():
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{number}_{counter}{ext}"
            counter += 1

    shutil.copy2(image_path, output_path)
    print(f"✅ → {output_path.name}  (検出番号: {number})")
    return True


def test_case(ocr_results: list[dict], output_dir: Path):
    """
    入力ファイル名（正解ラベル）とOCR認識結果を比較して、詳細な精度レポートを表示する。
    入力ファイル名に含まれる枝番（例: 344_1）は自動的に除去して比較する。
    """
    if not ocr_results:
        print("\nテスト対象の結果が存在しません。")
        return

    # 入力ファイル名が正解ラベル（数字）として適切か自動判定する
    # 最初の結果のファイル名を確認
    first_path = ocr_results[0]["original_path"]
    first_stem = first_path.stem
    # 枝番などを考慮して数字のみにクリーンアップして判定
    clean_stem = first_stem.split('_')[0]
    if not clean_stem.isdigit():
        print("\n[INFO] 入力ファイル名が数字ではないため、正解ラベルテストはスキップします。")
        return

    total_count = len(ocr_results)
    match_count = 0
    mismatch_count = 0
    unresolved_count = 0
    
    output_ok_count = 0
    
    details = [] # list of dict (file_name, expected, actual, status)

    for res in ocr_results:
        orig_path = res["original_path"]
        orig_name = orig_path.name
        expected_num = orig_path.stem.split('_')[0] # 枝番（_1など）の除去
        actual_num = res["number"]
        
        status = ""
        if actual_num is None or actual_num == "":
            status = "検出不可"
            unresolved_count += 1
        elif expected_num == actual_num:
            status = "一致 (Match)"
            match_count += 1
        else:
            status = "誤認識 (Mismatch)"
            mismatch_count += 1
            
        # 最終出力ファイル（保存先）の確認（サイズ比較で判定）
        src_size = orig_path.stat().st_size
        dest_exists = False
        for out_file in output_dir.glob(f"*{orig_path.suffix}"):
            if out_file.stat().st_size == src_size:
                dest_exists = True
                break
        
        if dest_exists:
            output_ok_count += 1
            
        details.append({
            "name": orig_name,
            "expected": expected_num,
            "actual": actual_num if actual_num else "(検出不可)",
            "status": status
        })

    accuracy = (match_count / total_count) * 100
    output_rate = (output_ok_count / total_count) * 100

    print(f"\n======================================================================")
    print(f"                     OCR 認識・出力 精度レポート")
    print(f"======================================================================")
    print(f"【統計情報】")
    print(f"  ・処理した入力ファイル数:  {total_count:4d} 件")
    print(f"  ・OCR認識成功 (正解):      {match_count:4d} 件")
    print(f"  ・OCR認識失敗 (誤認識):     {mismatch_count:4d} 件")
    print(f"  ・OCR認識不可 (検出なし):   {unresolved_count:4d} 件")
    print(f"  ・最終出力成功 (保存済):   {output_ok_count:4d} 件")
    print()
    
    # 簡易星評価
    stars = "★" * round(accuracy / 20) + "☆" * (5 - round(accuracy / 20))
    print(f"【総合認識精度 (Accuracy)】")
    print(f"  {stars}  {accuracy:.2f} %")
    print(f"【最終出力成功率 (Output Rate)】")
    print(f"  {output_rate:.2f} %")

    # 失敗したファイルの詳細を表示
    failures = [d for d in details if d["status"] != "一致 (Match)"]
    if failures:
        print(f"----------------------------------------------------------------------")
        print(f"【詳細：誤認識または検出不可のファイル (計 {len(failures)} 件)】")
        print(f"  %-35s --> %-15s (%s)" % ("[入力ファイル名]", "[OCR認識結果]", "[ステータス]"))
        for f in sorted(failures, key=lambda x: x["name"]):
            print(f"  %-35s --> %-15s (%s)" % (f["name"], f["actual"], f["status"]))
            
    print(f"======================================================================")


def delete_output_folder(output_folder: str):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"出力フォルダを削除しました: {output_folder}")
    else:
        print(f"出力フォルダが存在しません: {output_folder}")


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="画像左上の登録番号を読み取り、番号をファイル名にして保存する (TensorFlow 2.16.1)"
    )
    parser.add_argument("files", nargs="*", help="処理する画像ファイル（複数指定可）")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="画像が格納されたフォルダ")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="保存先フォルダ（デフォルト: ./output）")
    parser.add_argument("--template", type=str, default=None,
                        help="空欄カードのPDFテンプレート（背景差分用）")
    parser.add_argument("--debug", type=str, default=None,
                        help="testケースの実行")
    args = parser.parse_args()
    delete_output_folder(args.output_dir)  # 出力フォルダを削除してから処理を開始
    # 処理対象のファイル一覧を構築
    image_files: list[Path] = []

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"エラー: フォルダが見つかりません: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        image_files = [
            f for f in sorted(input_dir.iterdir())
            if f.suffix.lower() in ALL_EXTENSIONS
        ]
    elif args.files:
        image_files = [Path(f) for f in args.files]
    else:
        parser.print_help()
        sys.exit(1)

    if not image_files:
        print("処理対象の画像ファイルがありません。")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバッグ出力フォルダ
    debug_dir = Path("./debug_output")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # テンプレートは使用しません（ハフ変換による自動枠線除去）
    template_gray = None

    # MNIST モデルの準備
    print("OCR エンジンを初期化中 (TensorFlow 2.16.1)...")
    model = get_mnist_model()

    # PDF と画像を分離
    pdf_files = [f for f in image_files if f.suffix.lower() == ".pdf"]
    img_only_files = [f for f in image_files if f.suffix.lower() != ".pdf"]

    # PDF を画像に変換
    temp_dirs = []
    pdf_image_map: list[tuple[Path, Path]] = []  # (変換後画像, 元PDF)
    if pdf_files:
        print(f"PDF ファイルを画像に変換中 ({len(pdf_files)} 件)...")
        for pdf_path in pdf_files:
            try:
                converted = pdf_to_images(str(pdf_path))
                temp_dirs.append(Path(converted[0]).parent)
                for img_path_str in converted:
                    pdf_image_map.append((Path(img_path_str), pdf_path))
                print(f"  {pdf_path.name} → {len(converted)} ページ")
            except Exception as e:
                print(f"  ❌ PDF 変換エラー ({pdf_path.name}): {e}")

    total_count = len(img_only_files) + len(pdf_image_map)

    # 処理実行
    print(f"\n{'='*50}")
    print(f"対象ファイル数: {total_count} (画像: {len(img_only_files)}, PDF→画像: {len(pdf_image_map)})")
    print(f"出力先: {output_dir.resolve()}")
    print(f"デバッグ出力: {debug_dir.resolve()}")
    print(f"{'='*50}\n")

    ocr_results = []

    # 通常の画像ファイルを処理
    for img_path in img_only_files:
        print(f"処理中: {img_path.name}")
        try:
            res = extract_number(str(img_path), model, debug_dir=debug_dir, template_gray=template_gray)
            if res is not None:
                number, best_info = res
                score_key = (not best_info["ink_warning"], not best_info["has_low_conf"], best_info["num_digits"], best_info["avg_confidence"])
                ocr_results.append({
                    "original_path": img_path,
                    "ext": img_path.suffix.lower(),
                    "number": number,
                    "score_key": score_key,
                    "avg_confidence": best_info["avg_confidence"],
                    "ink_warning": best_info["ink_warning"],
                    "rotated": best_info["rotated"]
                })
            else:
                print("  ❌ 番号を検出できませんでした")
        except Exception as e:
            print(f"  ❌ エラー: {e}")

    # PDF から変換した画像を処理（元の PDF をコピーして保存）
    for converted_img, original_pdf in pdf_image_map:
        print(f"処理中: {original_pdf.name} (→ {converted_img.name})")
        try:
            res = extract_number(str(converted_img), model, debug_dir=debug_dir, template_gray=template_gray)
            if res is not None:
                number, best_info = res
                score_key = (not best_info["ink_warning"], not best_info["has_low_conf"], best_info["num_digits"], best_info["avg_confidence"])
                ocr_results.append({
                    "original_path": original_pdf,
                    "ext": original_pdf.suffix.lower(),
                    "number": number,
                    "score_key": score_key,
                    "avg_confidence": best_info["avg_confidence"],
                    "ink_warning": best_info["ink_warning"],
                    "rotated": best_info["rotated"]
                })
            else:
                print("  ❌ 番号を検出できませんでした")
        except Exception as e:
            print(f"  ❌ エラー: {e}")

    # 衝突解決とファイルコピーの実行
    from collections import defaultdict
    grouped_results = defaultdict(list)
    for res in ocr_results:
        grouped_results[res["number"]].append(res)

    success = 0
    fail = total_count - len(ocr_results)

    print(f"\n--- 衝突解決とファイル出力中 ---")
    for number, items in grouped_results.items():
        if len(items) == 1:
            item = items[0]
            output_path = output_dir / f"{number}{item['ext']}"
            if item.get("rotated"):
                save_rotated(item["original_path"], output_path)
            else:
                shutil.copy2(item["original_path"], output_path)
                print(f"✅ → {output_path.name}  (元: {item['original_path'].name})")
            success += 1
        else:
            # 衝突発生：スコアの降順でソートして最良のものをオリジナル名、残りは _1, _2 サフィックス
            items.sort(key=lambda x: x["score_key"], reverse=True)
            source_names = [item["original_path"].name for item in items]
            print(f"⚠️ [COLLISION] 複数のファイルが '{number}' と認識されました: {source_names}")
            
            best_item = items[0]
            output_path = output_dir / f"{number}{best_item['ext']}"
            if best_item.get("rotated"):
                save_rotated(best_item["original_path"], output_path)
            else:
                shutil.copy2(best_item["original_path"], output_path)
                print(f"✅ → {output_path.name} (最良スコア採用, 元: {best_item['original_path'].name}, conf: {best_item['avg_confidence']:.4f})")
            success += 1

            for idx, item in enumerate(items[1:]):
                suffix_path = output_dir / f"{number}_{idx+1}{item['ext']}"
                if item.get("rotated"):
                    save_rotated(item["original_path"], suffix_path)
                else:
                    shutil.copy2(item["original_path"], suffix_path)
                    print(f"⚠️ → {suffix_path.name} (衝突退避, 元: {item['original_path'].name}, conf: {item['avg_confidence']:.4f})")
                success += 1

    # 一時ファイルのクリーンアップ
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"完了: 成功 {success} / 失敗 {fail} / 合計 {success + fail}")
    print(f"{'='*50}")
    test_case(ocr_results, output_dir)
    end = time.perf_counter()
    print("処理時間: {:.2f} 秒".format((end - start)))
if __name__ == "__main__":
    main()
