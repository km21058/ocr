# 登録番号 OCR ツール (extract_number.py)

画像の左上部分にある登録番号を検出し、その番号をファイル名にして自動保存（コピー）するスクリプトです。

---

## 1. 環境のアクティベート

まず、ツールを実行するために仮想環境を有効化（アクティベート）します。

```bash
cd /home/mikamipc3/projectmanage
source venv/bin/activate
```

※ もし仮想環境に必要なライブラリが不足している場合は、以下を実行してください。
```bash
pip install tensorflow==2.16.1 tf-keras==2.16.0 opencv-python-headless pillow numpy
```

---

## 2. 実行方法

### パターンA：特定の画像を指定して実行する
処理したい画像（複数指定可能）を引数に渡します。

```bash
# 1枚だけ処理する場合
python extract_number.py image.png

# 複数枚をまとめて処理する場合
python extract_number.py image.png test.png
```

### パターンB：フォルダ内の画像をまとめて処理する
`--input-dir` オプションでフォルダを指定すると、その中の画像をすべて自動で処理します。

```bash
python extract_number.py --input-dir ./input_images
```

* `--output-dir`: 保存先のフォルダ（デフォルトは `./output`）を変更したい場合に指定します。
  ```bash
  python extract_number.py --input-dir ./input_images --output-dir ./results
  ```

---

## 3. 出力ファイルについて

1. **結果の保存 (`output/` フォルダ)**
   - 認識された番号（例: `62`, `63`）をファイル名にして、画像がコピーされます。
   - 同名のファイルが既に存在する場合は、`62_1.png` のように自動で連番が付与されます。

2. **デバッグ用中間画像 (`debug_output/` フォルダ)**
   - 処理中の画像が保存されます。数字がうまく認識できない場合の確認用にご利用ください。
   - `*_1_cropped.png`: 左上の切り出し範囲
   - `*_2_binary.png`: 二値化（白黒）画像
   - `*_3_detected_regions.png`: 数字として認識され、切り出された領域（赤枠で表示されます）
