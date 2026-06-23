"""
スタンプフォント（Century, Times New Roman, MS明朝）の合成データと
MNIST手書き文字データセットをブレンドして、CNNモデルを再学習するスクリプト。
"""

import os
import random
import sys
from pathlib import Path

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

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# 出力モデル名
MODEL_PATH = Path(__file__).parent / "stamp_mnist_model.keras"

# Windows/WSL標準のフォントディレクトリとフォント候補の動的解決
def _find_fonts():
    if os.name == "nt":
        font_dir = Path("C:/Windows/Fonts")
    else:
        # WSLでWindowsのフォントディレクトリがマウントされている場所を最優先
        wsl_fonts = Path("/mnt/c/Windows/Fonts")
        if wsl_fonts.is_dir():
            font_dir = wsl_fonts
        else:
            font_dir = Path("/usr/share/fonts")
            
    font_files = {}
    targets = {
        "Century": "century",
        "Times": "times",
        "MSMincho": "msmincho"
    }
    
    if font_dir.is_dir():
        # 大文字小文字を区別せず再帰的に検索
        for path in font_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in (".ttf", ".ttc", ".otf"):
                stem_lower = path.stem.lower()
                for key, target_prefix in targets.items():
                    if key not in font_files and stem_lower == target_prefix:
                        font_files[key] = str(path)
                        
    return str(font_dir), font_files

FONT_DIR, FONT_FILES = _find_fonts()

def keep_only_blue(cropped_img):
    """
    HSV色空間を用いて、青系（シアン〜ブルー）以外の色をすべて白色で塗りつぶす（赤系と黒系を抜く）
    """
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    is_blue = (h >= 75) & (h <= 145) & (s >= 35) & (v >= 50)
    filtered_img = np.ones_like(cropped_img) * 255
    filtered_img[is_blue] = cropped_img[is_blue]
    return filtered_img

def generate_synthetic_digit(digit: int, font_path: str, size: int = 40, damage_level: str = "none") -> np.ndarray:
    """
    Pipeline-Consistent Rendering:
    1. Render a blue digit on a white background.
    2. Apply keep_only_blue filter.
    3. Apply Gaussian blur & adaptive threshold.
    4. Apply Hough mask subtraction simulation (if damage_level != 'none').
    5. Crop, pad, and resize to 28x28.
    """
    canvas_size = 64
    img_pil = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        font = ImageFont.load_default()
        
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # 描画位置に少しゆらぎを加える
    dx = random.randint(-4, 4)
    dy = random.randint(-4, 4)
    
    # BGR/RGBでの青スタンプの色（R: 30-100, G: 80-150, B: 180-255）
    blue_color = (random.randint(30, 100), random.randint(80, 150), random.randint(180, 255)) # RGB
    draw.text(((canvas_size - text_w) // 2 - bbox[0] + dx, (canvas_size - text_h) // 2 - bbox[1] + dy), 
              text, fill=blue_color, font=font)
    
    np_img_rgb = np.array(img_pil)
    np_img_bgr = cv2.cvtColor(np_img_rgb, cv2.COLOR_RGB2BGR)
    
    # 1. 回転 (回転角 -10度 〜 10度)
    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), angle, 1.0)
    np_img_bgr = cv2.warpAffine(np_img_bgr, matrix, (canvas_size, canvas_size), borderValue=(255, 255, 255))
    
    # 2. keep_only_blue
    filtered = keep_only_blue(np_img_bgr)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    # 3. adaptiveThreshold (二値化)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )
    
    # 4. 枠線消去シミュレーション
    if damage_level == "damaged":
        # 4a. Hough線消去による削れの再現
        if random.random() < 0.4:
            # 水平線を描画して消去
            line_y = random.randint(15, canvas_size - 15)
            cv2.line(binary, (0, line_y), (canvas_size, line_y), 0, random.randint(2, 3))
        if random.random() < 0.3:
            # 垂直線を描画して消去
            line_x = random.randint(15, canvas_size - 15)
            cv2.line(binary, (line_x, 0), (line_x, canvas_size), 0, random.randint(2, 3))
            
        # 4b. 左右端削れ (特に 5, 8 の縦線消失などのシミュレーション)
        prob_erase = 0.5 if digit in [3, 5, 8] else 0.25
        if random.random() < prob_erase:
            side = random.choice(['left', 'right'])
            strip_w = random.randint(2, 7)
            if side == 'right':
                binary[:, canvas_size - strip_w:] = 0
            else:
                binary[:, :strip_w] = 0
                
        # 4c. 上下端削れ
        if random.random() < 0.2:
            side = random.choice(['top', 'bottom'])
            strip_h = random.randint(2, 6)
            if side == 'top':
                binary[:strip_h, :] = 0
            else:
                binary[canvas_size - strip_h:, :] = 0
                
        # 4d. ガウシアンブラーによるボケ・太さ変化
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            binary = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)
            _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
            
    # 5. MNIST 形式への正規化 (28x28, 白文字黒背景)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = binary[y:y+h, x:x+w]
        
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padded = cv2.copyMakeBorder(
            digit_crop, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        
        margin = max(max_dim // 4, 4)
        padded = cv2.copyMakeBorder(
            padded, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=0
        )
        
        resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        resized = np.zeros((28, 28), dtype=np.uint8)
        
    normalized = resized.astype("float32") / 255.0
    return normalized

def build_dataset(num_samples_per_digit=3000, damage_level="none", oversample_pairs=False):
    """
    指定されたフォント群からスタンプ合成データセットを構築する
    """
    x_synth = []
    y_synth = []
    
    available_fonts = []
    for name, path in FONT_FILES.items():
        if os.path.exists(path):
            available_fonts.append((name, path))
            
    if not available_fonts:
        raise FileNotFoundError(f"利用可能なフォントが一つも見つかりませんでした。")

    for digit in range(10):
        # 混同されやすい数字(3, 5, 8, 2, 7, 4, 0)をオーバーサンプリング
        count = num_samples_per_digit
        if oversample_pairs and digit in [3, 5, 8, 2, 7, 4, 0]:
            count = int(num_samples_per_digit * 2.0)
            
        for _ in range(count):
            _, font_path = random.choice(available_fonts)
            font_size = random.randint(32, 45)
            
            digit_img = generate_synthetic_digit(digit, font_path, font_size, damage_level=damage_level)
            x_synth.append(digit_img)
            y_synth.append(digit)
            
    return np.array(x_synth), np.array(y_synth)

def load_real_crops(real_data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    training_data/real から実データクロップを読み込み、MNIST用に前処理して返す
    """
    from extract_number import preprocess_for_mnist
    x_real = []
    y_real = []
    
    for digit in range(10):
        digit_dir = real_data_dir / f"digit_{digit}"
        if not digit_dir.exists():
            continue
        for img_path in digit_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            processed = preprocess_for_mnist(img)
            if processed is not None:
                x_real.append(processed)
                y_real.append(digit)
                
    return np.array(x_real), np.array(y_real)

def augment_real_crop(crop: np.ndarray) -> np.ndarray:
    """
    実データ画像に対するデータ拡張
    """
    img = (crop * 255).astype(np.uint8)
    
    # 1. 回転
    angle = random.uniform(-6, 6)
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    img = cv2.warpAffine(img, matrix, (28, 28), flags=cv2.INTER_AREA)
    
    # 2. 並行移動
    tx = random.randint(-1, 1)
    ty = random.randint(-1, 1)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_trans, (28, 28), flags=cv2.INTER_AREA)
    
    # 3. 膨張/収縮
    if random.random() < 0.3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.choice([True, False]):
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)
            
    return img.astype("float32") / 255.0

def main():
    print("MNIST データをロード中...")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist = x_train_mnist.astype("float32") / 255.0
    x_test_mnist = x_test_mnist.astype("float32") / 255.0
    
    # 5%の極小比率で手書きMNISTを混ぜる (バイアスのための正則化)
    mnist_train_count = int(len(x_train_mnist) * 0.05)
    mnist_test_count = int(len(x_test_mnist) * 0.05)
    x_train_mnist = x_train_mnist[:mnist_train_count]
    y_train_mnist = y_train_mnist[:mnist_train_count]
    x_test_mnist = x_test_mnist[:mnist_test_count]
    y_test_mnist = y_test_mnist[:mnist_test_count]

    # モデル定義 (CNN構造)
    print("モデル構築中...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        
        # ブロック1: 32フィルタ
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # ブロック2: 64フィルタ
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # ブロック3: 128フィルタ
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
    
    # -------------------------------------------------------------------------
    # Stage 1: クリーンな Pipeline-Consistent 合成データの学習
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Stage 1: クリーンな合成データの学習を開始します")
    print("="*50)
    
    x_synth_s1, y_synth_s1 = build_dataset(num_samples_per_digit=4000, damage_level="none", oversample_pairs=False)
    x_train_s1 = np.concatenate([x_synth_s1, x_train_mnist], axis=0)
    y_train_s1 = np.concatenate([y_synth_s1, y_train_mnist], axis=0)
    
    x_train_s1 = np.expand_dims(x_train_s1, axis=-1)
    
    # シャッフル
    idx_s1 = np.arange(len(x_train_s1))
    np.random.shuffle(idx_s1)
    x_train_s1 = x_train_s1[idx_s1]
    y_train_s1 = y_train_s1[idx_s1]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    model.fit(
        x_train_s1, y_train_s1,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )
    
    # -------------------------------------------------------------------------
    # Stage 2: 損傷加工した合成データ + 混同しやすい数字のオーバーサンプリング
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Stage 2: 損傷加工合成データと混同ペア重点学習を開始します")
    print("="*50)
    
    x_synth_s2, y_synth_s2 = build_dataset(num_samples_per_digit=3000, damage_level="damaged", oversample_pairs=True)
    x_train_s2 = np.concatenate([x_synth_s2, x_train_mnist], axis=0)
    y_train_s2 = np.concatenate([y_synth_s2, y_train_mnist], axis=0)
    
    x_train_s2 = np.expand_dims(x_train_s2, axis=-1)
    
    # シャッフル
    idx_s2 = np.arange(len(x_train_s2))
    np.random.shuffle(idx_s2)
    x_train_s2 = x_train_s2[idx_s2]
    y_train_s2 = y_train_s2[idx_s2]
    
    # 低い学習率で微調整
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    model.fit(
        x_train_s2, y_train_s2,
        epochs=12,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # -------------------------------------------------------------------------
    # Stage 3: 実データファインチューニング
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Stage 3: 実データのファインチューニングを開始します")
    print("="*50)
    
    real_data_dir = Path("./training_data/real")
    if not real_data_dir.exists():
        print("⚠️ 実データのクロップフォルダが見つかりません。Stage 3 はスキップします。")
        model.save(str(MODEL_PATH))
        return
        
    x_real, y_real = load_real_crops(real_data_dir)
    print(f"ロードした実データクロップ数: {len(x_real)}")
    
    if len(x_real) < 10:
        print("⚠️ 実データクロップが少なすぎます。Stage 3 はスキップします。")
        model.save(str(MODEL_PATH))
        return
        
    # Numpyによる手動の層化分割 (Stratified Split)
    train_idx = []
    val_idx = []
    for digit in range(10):
        indices = np.where(y_real == digit)[0]
        np.random.seed(42)
        np.random.shuffle(indices)
        split_point = int(len(indices) * 0.8)
        train_idx.extend(indices[:split_point])
        val_idx.extend(indices[split_point:])
        
    x_real_train, y_real_train = x_real[train_idx], y_real[train_idx]
    x_real_val, y_real_val = x_real[val_idx], y_real[val_idx]
    
    # 実トレーニングデータをデータ拡張で 10 倍に水増し
    x_real_augmented = []
    y_real_augmented = []
    for crop, label in zip(x_real_train, y_real_train):
        x_real_augmented.append(crop)
        y_real_augmented.append(label)
        for _ in range(10):
            aug_crop = augment_real_crop(crop)
            x_real_augmented.append(aug_crop)
            y_real_augmented.append(label)
            
    x_real_augmented = np.array(x_real_augmented)
    y_real_augmented = np.array(y_real_augmented)
    
    # 過学習を防ぐために、一部の損傷合成データを Stage 3 にも混ぜる (2000サンプル)
    x_synth_s3, y_synth_s3 = build_dataset(num_samples_per_digit=200, damage_level="damaged", oversample_pairs=True)
    
    x_train_s3 = np.concatenate([x_real_augmented, x_synth_s3], axis=0)
    y_train_s3 = np.concatenate([y_real_augmented, y_synth_s3], axis=0)
    
    x_train_s3 = np.expand_dims(x_train_s3, axis=-1)
    x_real_val = np.expand_dims(x_real_val, axis=-1)
    
    # シャッフル
    idx_s3 = np.arange(len(x_train_s3))
    np.random.shuffle(idx_s3)
    x_train_s3 = x_train_s3[idx_s3]
    y_train_s3 = y_train_s3[idx_s3]
    
    # 畳み込みブロック 1 と 2 をフリーズ (下位特徴を保護し、過学習を防止)
    # model.layers 内のインデックス:
    # 0, 1, 2, 3, 4, 5 は Block 1
    # 6, 7, 8, 9, 10, 11 は Block 2
    for layer in model.layers[:12]:
        layer.trainable = False
        
    print(f"Stage 3 訓練サンプル数: {len(x_train_s3)}")
    print(f"Stage 3 検証サンプル数 (実データHoldout): {len(x_real_val)}")
    
    # 非常に低い学習率で微調整
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    model.fit(
        x_train_s3, y_train_s3,
        epochs=15,
        batch_size=32,
        validation_data=(x_real_val, y_real_val),
        verbose=1
    )
    
    # 7. モデル保存
    model.save(str(MODEL_PATH))
    print(f"\n🎉 3ステージ学習完了！モデルを保存しました: {MODEL_PATH}")

if __name__ == "__main__":
    main()
