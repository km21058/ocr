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

def generate_synthetic_digit(digit: int, font_path: str, size: int = 40) -> np.ndarray:
    """
    Pillow を使用して、指定されたフォントで数字画像を生成し、MNIST形式 (28x28) に変換する。
    """
    # 余裕を持たせた少し大きめのキャンバスに描画 (アンチエイリアス用)
    canvas_size = 64
    img = Image.new("L", (canvas_size, canvas_size), 0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, size)
    except IOError:
        # フォントが読み込めない場合はデフォルトを使用
        font = ImageFont.load_default()
        
    # 文字描画位置のセンタリング
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    draw.text(((canvas_size - text_w) // 2 - bbox[0], (canvas_size - text_h) // 2 - bbox[1]), 
              text, fill=255, font=font)
    
    # numpy 配列に変換
    np_img = np.array(img)
    
    # --- データ拡張 (Data Augmentation) の適用 ---
    
    # 1. 回転 (回転角 -15度 〜 15度)
    angle = random.uniform(-15, 15)
    matrix = cv2.getRotationMatrix2D((canvas_size // 2, canvas_size // 2), angle, 1.0)
    np_img = cv2.warpAffine(np_img, matrix, (canvas_size, canvas_size), flags=cv2.INTER_AREA)
    
    # 2. 枠線消去シミュレーション (ランダムな切り欠き/消しゴム効果)
    # 文字の端や一部をランダムに黒(0)で塗りつぶし、削れを再現
    if random.random() < 0.4:  # 40%の確率で適用
        h_crop = random.randint(3, 8)
        w_crop = random.randint(3, 8)
        cx = random.randint(10, canvas_size - 10)
        cy = random.randint(10, canvas_size - 10)
        np_img[cy:cy+h_crop, cx:cx+w_crop] = 0

    # 2b. 枠線除去シミュレーション（左右端のストライプ消去）
    # 実際のスタンプでは枠線除去の際に数字の左右端が削れることがある
    # 特に 5→3, 8→3 の誤認は右側の縦線が消えることが原因
    if random.random() < 0.3:  # 30%の確率で左右端消去
        side = random.choice(['left', 'right'])
        strip_w = random.randint(3, 10)
        if side == 'right':
            np_img[:, canvas_size - strip_w:] = 0
        else:
            np_img[:, :strip_w] = 0

    # 2c. 上下端のストライプ消去（7の上部横棒が消えるケースなど）
    if random.random() < 0.2:  # 20%の確率で上下端消去
        side = random.choice(['top', 'bottom'])
        strip_h = random.randint(3, 8)
        if side == 'top':
            np_img[:strip_h, :] = 0
        else:
            np_img[canvas_size - strip_h:, :] = 0

    # 3. ノイズ・かすれ効果の追加 (ガウシアンノイズ)
    if random.random() < 0.3:
        noise = np.random.normal(0, 15, np_img.shape).astype(np.uint8)
        np_img = cv2.add(np_img, noise)
        
    # 4. 太さのランダム変更 (膨張・収縮)
    if random.random() < 0.3:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        if random.choice([True, False]):
            np_img = cv2.dilate(np_img, kernel, iterations=1)
        else:
            np_img = cv2.erode(np_img, kernel, iterations=1)

    # 5. バウンディングボックスの切り出しと 28x28 へのフィッティング (MNISTと同様)
    coords = cv2.findNonZero(np_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = np_img[y:y+h, x:x+w]
        
        # 正方形パディング
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padded = cv2.copyMakeBorder(
            digit_crop, pad_h, pad_h, pad_w, pad_w,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 外周に余白を追加 (MNISTの余白比率に合わせる)
        margin = max(max_dim // 4, 4)
        padded = cv2.copyMakeBorder(
            padded, margin, margin, margin, margin,
            cv2.BORDER_CONSTANT, value=0
        )
        
        # 28x28 にリサイズ
        np_img = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # 万が一文字が消えてしまった場合は 28x28 のゼロ行列
        np_img = cv2.resize(np_img, (28, 28), interpolation=cv2.INTER_AREA)
        
    # 正規化
    normalized = np_img.astype("float32") / 255.0
    return normalized

def build_dataset(num_samples_per_digit=3000):
    """
    指定されたフォント群からスタンプ合成データセットを構築する
    """
    print("スタンプ合成データを生成中...")
    x_synth = []
    y_synth = []
    
    # 利用可能なフォントのみをフィルタリング
    available_fonts = []
    for name, path in FONT_FILES.items():
        if os.path.exists(path):
            available_fonts.append((name, path))
            print(f"  フォント使用可能: {name} ({path})")
        else:
            print(f"  Warning: フォントが見つかりません: {name} ({path})")
            
    if not available_fonts:
        raise FileNotFoundError(f"利用可能なフォントが一つも見つかりませんでした。フォントディレクトリ（{FONT_DIR}）の中身を確認してください。")

    for digit in range(10):
        for _ in range(num_samples_per_digit):
            # ランダムにフォントを選択
            _, font_path = random.choice(available_fonts)
            # 文字サイズも少し揺らす
            font_size = random.randint(32, 45)
            
            digit_img = generate_synthetic_digit(digit, font_path, font_size)
            x_synth.append(digit_img)
            y_synth.append(digit)
            
    return np.array(x_synth), np.array(y_synth)

def main():
    # 1. MNIST 手書きデータセットをロード
    print("MNIST データをロード中...")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    
    x_train_mnist = x_train_mnist.astype("float32") / 255.0
    x_test_mnist = x_test_mnist.astype("float32") / 255.0
    
    # 2. スタンプ合成データを生成 (数字ごとに 3000 サンプル、計 30,000 サンプル)
    x_train_synth, y_train_synth = build_dataset(num_samples_per_digit=3000)
    x_test_synth, y_test_synth = build_dataset(num_samples_per_digit=500)
    
    # 3. データの結合とシャッフル
    x_train = np.concatenate([x_train_mnist, x_train_synth], axis=0)
    y_train = np.concatenate([y_train_mnist, y_train_synth], axis=0)
    x_test = np.concatenate([x_test_mnist, x_test_synth], axis=0)
    y_test = np.concatenate([y_test_mnist, y_test_synth], axis=0)
    
    # CNN用の次元追加 (28x28 -> 28x28x1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    print(f"訓練データ数: {x_train.shape[0]} (MNIST: {x_train_mnist.shape[0]}, 合成: {x_train_synth.shape[0]})")
    print(f"テストデータ数: {x_test.shape[0]} (MNIST: {x_test_mnist.shape[0]}, 合成: {x_test_synth.shape[0]})")
    
    # 4. シャッフル
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # 5. モデル構築 (extract_number.py と同じCNN構造)
    print("モデル構築中...")
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
    
    # 6. トレーニング
    print("モデルの学習を開始します（15 Epochs）...")
    model.fit(
        x_train, y_train, 
        epochs=15, 
        batch_size=128,
        validation_data=(x_test, y_test), 
        verbose=1
    )
    
    # 7. モデル保存
    model.save(str(MODEL_PATH))
    print(f"\n🎉 学習完了！モデルを保存しました: {MODEL_PATH}")

if __name__ == "__main__":
    main()
