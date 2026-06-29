"""
画像・PDFの左上にある登録番号を EasyOCR で読み取り、
その番号をファイル名にして画像を保存するスクリプト。

EasyOCR（PyTorchベースのOCRエンジン）を使用。

処理フロー:
  1. PDF の場合は PyMuPDF で画像に変換
  2. OpenCV で画像左上をクロップ（大雑把な位置）
  3. HSVカラーフィルタで青（シアン〜ブルー）と黒以外の赤を白色化して除去
  4. ハフ変換による枠線検出＋インペイントで青枠線を消去
  5. 消去後のクリーンなカラー画像を EasyOCR に入力して数字を検出・認識
  6. 認識した番号をファイル名にして保存

使い方:
  python extract_number_easyocr.py image.jpg
  python extract_number_easyocr.py document.pdf
  python extract_number_easyocr.py --input-dir ./images
  python extract_number_easyocr.py --input-dir ./images --output-dir ./renamed

必要なライブラリ:
  pip install easyocr PyMuPDF opencv-python-headless numpy Pillow
"""

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# サポートする画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
# サポートするファイル拡張子（PDF含む）
ALL_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

# EasyOCR Reader のキャッシュ
_reader = None

def get_easyocr_reader():
    """
    EasyOCR Readerを初期化（初回のみロード）。
    GPUが利用可能な場合は自動的にGPUを使用します。
    """
    global _reader
    if _reader is not None:
        return _reader

    import easyocr
    print("  EasyOCR エンジンを初期化中...")
    # 英語(en)モデルをロード。数字のみの読み取りなので en で十分。
    # cuDNNバージョン不一致エラーを回避するため、GPUを使用せずCPUモードで動作させます。
    _reader = easyocr.Reader(["en"], gpu=False)
    return _reader

def crop_top_left(image_path: str,
                   x_start: float = 0.0, x_end: float = 0.30,
                   y_start: float = 0.04, y_end: float = 0.25):
    """
    画像の左上部分（登録番号エリア）を切り出す。
    下部の手書き黒文字などをカラーフィルタで消すため、y_end は広め（25%）に取得。
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

def remove_borders_hough(cropped_img):
    """
    ハフ変換等を利用した枠線（直線）の検出とインペイント消去
    """
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
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
                cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
                
    # 原画像に対してインペイントを適用して枠線を消去
    inpainted = cv2.inpaint(cropped_img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def extract_number(image_path: str, reader, debug_dir: Path | None = None) -> str | None:
    """
    画像の左上から登録番号（数字）を EasyOCR で抽出する。
    """
    image_name = Path(image_path).name
    
    # 1. 大雑把なクロップ
    cropped = crop_top_left(image_path)
    
    # 2. 青以外（黒・赤など）を除去するカラーフィルタ適用
    filtered = keep_blue_and_black(cropped)
    
    # 3. ハフ変換による枠線除去
    inpainted = remove_borders_hough(filtered)
    
    # デバッグ用中間画像の保存
    if debug_dir is not None:
        base = Path(image_name).stem
        cv2.imwrite(str(debug_dir / f"{base}_1_cropped.png"), cropped)
        cv2.imwrite(str(debug_dir / f"{base}_2_filtered.png"), filtered)
        cv2.imwrite(str(debug_dir / f"{base}_3_inpainted.png"), inpainted)

    # 4. EasyOCRによる数字読み取り
    # allowlistで数字 (0-9) のみに限定し、誤検出や不要な文字読み込みを防ぐ
    results = reader.readtext(inpainted, allowlist="0123456789")
    
    if not results:
        return None

    # 検出結果を左から右の順にソート (bboxの左上X座標で並び替え)
    results.sort(key=lambda r: r[0][0][0])
    
    recognized_digits = []
    for bbox, text, prob in results:
        # デバッグ出力
        print(f"    [OCR 検出] テキスト: {text}  信頼度: {prob:.4f}")
        # 信頼度がある程度高いものを採用
        if prob > 0.20:
            recognized_digits.append(text)

    if not recognized_digits:
        return None

    return "".join(recognized_digits)

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    """
    PDF ファイルを画像ファイルに変換する（PyMuPDF 使用）。
    一時ディレクトリに PNG として保存し、パスのリストを返す。
    """
    import fitz  # PyMuPDF
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="ocr_pdf_")
    image_paths = []

    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        stem = Path(pdf_path).stem
        output_path = os.path.join(temp_dir, f"{stem}_page{page_num + 1:04d}.png")
        pix.save(output_path)
        image_paths.append(output_path)

    doc.close()
    return image_paths

def process_image(image_path: Path, output_dir: Path, reader,
                   debug_dir: Path | None = None, output_ext: str | None = None) -> bool:
    """
    1枚の画像を処理: 番号を読み取り → 番号.拡張子 で保存。
    """
    print(f"処理中: {image_path.name}")

    try:
        number = extract_number(str(image_path), reader, debug_dir=debug_dir)
    except Exception as e:
        print(f"  ❌ エラー: {e}")
        return False

    if number is None:
        print("  ❌ 番号を検出できませんでした")
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

def test_case(test_folder: str, true_folder: str):
    test_files = os.listdir(test_folder)
    true_files = os.listdir(true_folder)
    test_counter = Counter(test_files)
    true_counter = Counter(true_files)

    common = list((test_counter & true_counter).elements())
    missing = list((true_counter - test_counter).elements())
    extra = list((test_counter - true_counter).elements())

    matched_count = len(common)
    missing_count = len(missing)
    extra_count = len(extra)

    print(f"期待ファイル数: {len(true_files)}")
    print(f"実際出力ファイル数: {len(test_files)}")
    print(f"一致したファイル数: {matched_count}")

    if missing_count:
        print("\n期待にあって出力にないファイル:")
        for filename in sorted(missing):
            print(f"  {filename}")

    if extra_count:
        print("\n出力にあって期待にないファイル:")
        for filename in sorted(extra):
            print(f"  {filename}")

    match_rate = matched_count / len(true_files) * 100 if true_files else 0.0
    print(f"一致率: {match_rate:.2f}%")

def delete_output_folder(output_folder: str):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"出力フォルダを削除しました: {output_folder}")
    else:
        print(f"出力フォルダが存在しません: {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="画像左上の登録番号を読み取り、番号をファイル名にして保存する (EasyOCR)"
    )
    parser.add_argument("files", nargs="*", help="処理する画像ファイル（複数指定可）")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="画像が格納されたフォルダ")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="保存先フォルダ（デフォルト: ./output）")
    args = parser.parse_args()
    
    delete_output_folder(args.output_dir)
    
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

    # EasyOCR Readerの初期化
    reader = get_easyocr_reader()

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

    print(f"\n{'='*50}")
    print(f"対象ファイル数: {total_count} (画像: {len(img_only_files)}, PDF→画像: {len(pdf_image_map)})")
    print(f"出力先: {output_dir.resolve()}")
    print(f"デバッグ出力: {debug_dir.resolve()}")
    print(f"{'='*50}\n")

    success = 0
    fail = 0

    # 通常の画像ファイルを処理
    for img_path in img_only_files:
        if process_image(img_path, output_dir, reader, debug_dir=debug_dir):
            success += 1
        else:
            fail += 1

    # PDF から変換した画像を処理
    for converted_img, original_pdf in pdf_image_map:
        print(f"処理中: {original_pdf.name} (→ {converted_img.name})")
        try:
            number = extract_number(str(converted_img), reader, debug_dir=debug_dir)
        except Exception as e:
            print(f"  ❌ エラー: {e}")
            fail += 1
            continue

        if number is None:
            print("  ❌ 番号を検出できませんでした")
            fail += 1
            continue

        # 元の PDF を番号付きでコピー
        ext = original_pdf.suffix.lower()
        output_path = output_dir / f"{number}{ext}"
        if output_path.exists():
            counter = 1
            while output_path.exists():
                output_path = output_dir / f"{number}_{counter}{ext}"
                counter += 1
        shutil.copy2(original_pdf, output_path)
        print(f"✅ → {output_path.name}  (検出番号: {number})")
        success += 1

    # 一時ファイルのクリーンアップ
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n{'='*50}")
    print(f"完了: 成功 {success} / 失敗 {fail} / 合計 {success + fail}")
    print(f"{'='*50}")
    test_case("./output", "./image")

if __name__ == "__main__":
    main()
