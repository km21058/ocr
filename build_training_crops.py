import os
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

# extract_number.py から関数をインポート
from extract_number import crop_top_left, extract_digit_regions, pdf_to_images

def main():
    image_dir = Path("./image")
    output_base = Path("./training_data/real")
    
    # クリーンアップとフォルダ作成
    if output_base.exists():
        shutil.rmtree(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (output_base / f"digit_{i}").mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(list(image_dir.glob("*.pdf")))
    print(f"実データからクロップを抽出中 ({len(pdf_files)} 件のPDF)...")

    # リトライパラメータの定義 (Pass 1 を最優先にし、必要に応じて Pass 2/3)
    passes = [
        {"y_start": 0.04, "x_end": 0.35, "min_area_ratio": 0.005, "disable_border_removal": False, "hough_threshold": 30},
        {"y_start": 0.04, "x_end": 0.38, "min_area_ratio": 0.003, "disable_border_removal": False, "hough_threshold": 45},
        {"y_start": 0.05, "x_end": 0.40, "min_area_ratio": 0.003, "disable_border_removal": True, "hough_threshold": 30}
    ]

    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        # ファイル名が期待する正解ラベル（例: 30000）
        expected_label = stem
        expected_len = len(expected_label)
        
        # PDFを一時画像に変換
        try:
            converted = pdf_to_images(str(pdf_path))
            img_path = converted[0]
            temp_dir = Path(img_path).parent
        except Exception as e:
            print(f"  ❌ PDF 変換エラー ({pdf_path.name}): {e}")
            continue

        best_digit_images = None
        
        # 各パスを実行し、期待する桁数と一致するものを探す
        for idx, params in enumerate(passes):
            try:
                cropped = crop_top_left(img_path, y_start=params["y_start"], x_end=params["x_end"])
                digit_images, digit_regions, ink_warning, _ = extract_digit_regions(
                    cropped, debug_dir=None, image_name=pdf_path.name,
                    min_area_ratio=params["min_area_ratio"],
                    disable_border_removal=params["disable_border_removal"],
                    hough_threshold=params["hough_threshold"]
                )
                
                if digit_images and len(digit_images) == expected_len:
                    best_digit_images = digit_images
                    break
            except Exception as e:
                continue

        # 一時ディレクトリのクリーンアップ
        shutil.rmtree(temp_dir, ignore_errors=True)

        if best_digit_images is not None:
            # 切り出した各数字を保存
            for i, digit_img in enumerate(best_digit_images):
                digit_val = int(expected_label[i])
                dest_path = output_base / f"digit_{digit_val}" / f"{stem}_{i}.png"
                cv2.imwrite(str(dest_path), digit_img)
            success_count += 1
        else:
            print(f"  ⚠️ {pdf_path.name} は桁数が一致しないためスキップされました (期待桁数: {expected_len})")
            fail_count += 1

    print(f"\n実データからの抽出完了: 成功 {success_count} 件 / 失敗 {fail_count} 件")
    print(f"保存先: {output_base.resolve()}")

if __name__ == "__main__":
    main()
