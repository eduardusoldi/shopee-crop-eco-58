import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import io
import streamlit as st

def crop_and_rearrange_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        return None

    output_pdf = fitz.open()

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, dpi=300)

        # Convert to numpy
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # Detect content
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Bounding box union
        x0 = min(cv2.boundingRect(c)[0] for c in contours)
        y0 = min(cv2.boundingRect(c)[1] for c in contours)
        x1 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours)
        y1 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours)

        padding = 10
        x0 = max(x0 - padding, 0)
        y0 = max(y0 - padding, 0)
        x1 = min(x1 + padding, img_np.shape[1])
        y1 = min(y1 + padding, img_np.shape[0])

        cropped = img_np[y0:y1, x0:x1]

        # --- Split into top and bottom halves ---
        h, w, _ = cropped.shape
        mid = h // 2
        top_half = cropped[:mid, :]
        bottom_half = cropped[mid:, :]

        # --- Create new side-by-side image ---
        new_w = w * 2
        new_h = mid
        combined = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        combined[:, :w] = top_half
        combined[:, w:] = bottom_half[:new_h, :]

        combined_pil = Image.fromarray(combined)

        # Save combined page to memory as PDF
        img_bytes = io.BytesIO()
        combined_pil.save(img_bytes, format="PDF")
        img_bytes.seek(0)

        img_pdf = fitz.open("pdf", img_bytes.read())
        output_pdf.insert_pdf(img_pdf)

    # Return final PDF as bytes
    buffer = io.BytesIO()
    output_pdf.save(buffer)
    output_pdf.close()
    return buffer.getvalue()

# --- Streamlit UI ---
st.title("📄 PDF Crop & Rearrange Tool")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing..."):
        result_bytes = crop_and_rearrange_pdf_bytes(uploaded_file.read())

    if result_bytes:
        st.success("✅ PDF cropped & rearranged successfully!")
        st.download_button(
            label="⬇️ Download Rearranged PDF",
            data=result_bytes,
            file_name=uploaded_file.name.replace(".pdf", "-rearranged.pdf"),
            mime="application/pdf"
        )
    else:
        st.error("❌ No content detected in the PDF.")
