"""
画像の左上にある登録番号を OCR で読み取り、
その番号をファイル名にして画像を保存するスクリプト。

TensorFlow 2.16.1 を使用（MNIST ベースの数字認識）。

処理フロー:
  1. OpenCV で画像左上を切り出し
  2. 二値化・輪郭検出で個々の数字領域を抽出
  3. TensorFlow/Keras の MNIST 学習済みモデルで各数字を分類
  4. 認識した番号をファイル名にして保存

使い方:
  python extract_number.py image.jpg
  python extract_number.py image1.jpg image2.jpg
  python extract_number.py --input-dir ./images
  python extract_number.py --input-dir ./images --output-dir ./renamed

必要なライブラリ:
  pip install tensorflow==2.16.1 opencv-python-headless Pillow numpy
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# TensorFlow のログを抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# サポートする画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# MNIST モデルのキャッシュ
_model = None


def get_mnist_model():
    """
    MNIST で学習した数字認識モデルを取得（初回のみ学習）。
    """
    global _model
    if _model is not None:
        return _model

    model_path = Path(__file__).parent / "mnist_model.keras"

    if model_path.exists():
        print("  学習済みモデルをロード中...")
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

        # CNN モデル構築
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(x_train, y_train, epochs=5, batch_size=128,
                  validation_data=(x_test, y_test), verbose=1)

        # 精度確認
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"  モデル精度: {acc:.4f}")

        # モデル保存
        model.save(str(model_path))
        print(f"  モデルを保存: {model_path}")

        _model = model

    return _model


def crop_top_left(image_path: str,
                   x_start: float = 0.0, x_end: float = 0.30,
                   y_start: float = 0.04, y_end: float = 0.18):
    """
    画像の左上部分（登録番号エリア）を切り出す。
    登録番号のすぐ下には別の手書き数字（分類番号等）があるため、
    縦方向は登録番号のみが含まれるように絞り（4%〜18%）、
    横方向はズレを考慮して広め（30%）に取得する。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像を読み込めません: {image_path}")

    h, w = img.shape[:2]
    x1 = int(w * x_start)
    x2 = int(w * x_end)
    y1 = int(h * y_start)
    y2 = int(h * y_end)
    cropped = img[y1:y2, x1:x2]
    return cropped


def extract_digit_regions(cropped_img, debug_dir: Path | None = None, image_name: str = ""):
    """
    切り出し画像から個々の数字の領域を検出し、
    左から右の順で返す。
    debug_dir が指定されていれば、中間画像を保存する。
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # ガウシアンブラーでノイズ除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 適応的二値化（背景のムラに強い）
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10
    )

    # デバッグ: 切り出し画像と二値化画像を保存
    if debug_dir is not None:
        base = Path(image_name).stem
        cv2.imwrite(str(debug_dir / f"{base}_1_cropped.png"), cropped_img)
        cv2.imwrite(str(debug_dir / f"{base}_2_binary.png"), binary)

    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    min_area = (h * w) * 0.01   # ノイズを除外
    max_area = (h * w) * 0.8

    digit_regions = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect_ratio = ch / max(cw, 1)

        # 数字らしい領域のフィルタリング:
        # 1. 面積制限
        # 2. アスペクト比制限
        # 3. 高さ制限（切り出し画像全体の高さの30%以上あること。漢字ラベルと細かなノイズ対策）
        # 4. 開始位置制限
        is_digit = (
            min_area < area < max_area and
            0.5 < aspect_ratio < 5.0 and
            ch > h * 0.30 and
            y > h * 0.05
        )

        if is_digit:
            digit_regions.append((x, y, cw, ch))

    # 左から右にソート
    digit_regions.sort(key=lambda r: r[0])

    # デバッグ: 検出領域を矩形で描画した画像を保存
    if debug_dir is not None:
        base = Path(image_name).stem
        debug_img = cropped_img.copy()
        for i, (x, y, cw, ch) in enumerate(digit_regions):
            cv2.rectangle(debug_img, (x, y), (x + cw, y + ch), (0, 0, 255), 2)
            cv2.putText(debug_img, str(i), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(debug_dir / f"{base}_3_detected_regions.png"), debug_img)

    # 実際の数字画像を切り出し
    digit_images = []
    for i, (x, y, cw, ch) in enumerate(digit_regions):
        # 少しパディングを追加
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + cw + pad)
        y2 = min(h, y + ch + pad)
        digit_img = gray[y1:y2, x1:x2]
        digit_images.append(digit_img)

        # デバッグ: 各数字領域を個別に保存
        if debug_dir is not None:
            base = Path(image_name).stem
            cv2.imwrite(str(debug_dir / f"{base}_4_digit_{i}.png"), digit_img)

    return digit_images


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


def extract_number(image_path: str, model, debug_dir: Path | None = None) -> str | None:
    """
    画像の左上から登録番号（数字）を抽出する。
    """
    image_name = Path(image_path).name
    cropped = crop_top_left(image_path)
    digit_images = extract_digit_regions(cropped, debug_dir=debug_dir, image_name=image_name)

    if not digit_images:
        return None

    recognized_digits = []
    for i, digit_img in enumerate(digit_images):
        processed = preprocess_for_mnist(digit_img)
        if processed is None:
            continue

        # デバッグ: MNIST 入力画像を保存
        if debug_dir is not None:
            base = Path(image_name).stem
            mnist_img = (processed * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"{base}_5_mnist_input_{i}.png"), mnist_img)

        # モデルで推論
        input_data = np.expand_dims(np.expand_dims(processed, axis=-1), axis=0)
        prediction = model.predict(input_data, verbose=0)
        digit = np.argmax(prediction[0])
        confidence = prediction[0][digit]

        print(f"[digit {i}] 認識: {digit}  信頼度: {confidence:.4f}")

        # 信頼度が低い場合はスキップ
        if confidence > 0.35:
            recognized_digits.append(str(digit))

    if not recognized_digits:
        return None

    return "".join(recognized_digits)


def process_image(image_path: Path, output_dir: Path, model, debug_dir: Path | None = None) -> bool:
    """
    1枚の画像を処理: 番号を読み取り → 番号.拡張子 で保存。
    """
    print(f"処理中: {image_path.name}")

    try:
        number = extract_number(str(image_path), model, debug_dir=debug_dir)
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return False

    if number is None:
        print("  ❌ 番号を検出できませんでした")
        return False

    ext = image_path.suffix.lower()
    output_path = output_dir / f"{number}{ext}"

    # 同名ファイルが既にある場合はサフィックスを付与
    if output_path.exists():
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{number}_{counter}{ext}"
            counter += 1

    shutil.copy2(image_path, output_path)
    print(f"✅ → {output_path.name}  (検出番号: {number})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="画像左上の登録番号を読み取り、番号をファイル名にして保存する (TensorFlow 2.16.1)"
    )
    parser.add_argument("files", nargs="*", help="処理する画像ファイル（複数指定可）")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="画像が格納されたフォルダ")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="保存先フォルダ（デフォルト: ./output）")
    args = parser.parse_args()

    # 処理対象のファイル一覧を構築
    image_files: list[Path] = []

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"エラー: フォルダが見つかりません: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        image_files = [
            f for f in sorted(input_dir.iterdir())
            if f.suffix.lower() in IMAGE_EXTENSIONS
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

    # MNIST モデルの準備
    print("OCR エンジンを初期化中 (TensorFlow 2.16.1)...")
    model = get_mnist_model()

    # 処理実行
    print(f"\n{'='*50}")
    print(f"対象ファイル数: {len(image_files)}")
    print(f"出力先: {output_dir.resolve()}")
    print(f"デバッグ出力: {debug_dir.resolve()}")
    print(f"{'='*50}\n")

    success = 0
    fail = 0
    for img_path in image_files:
        if process_image(img_path, output_dir, model, debug_dir=debug_dir):
            success += 1
        else:
            fail += 1

    print(f"\n{'='*50}")
    print(f"完了: 成功 {success} / 失敗 {fail} / 合計 {success + fail}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
