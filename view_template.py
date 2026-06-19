"""テンプレートPDFを画像に変換して確認する"""
import fitz
import os

pdf_path = "./template.pdf"
doc = fitz.open(pdf_path)
page = doc[0]
zoom = 300 / 72
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat)
pix.save("./debug_output/template_full.png")
doc.close()
print(f"Template saved: {pix.width}x{pix.height}")
