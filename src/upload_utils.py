import io
import re
from pathlib import Path

import fitz
import pytesseract
import os

from PIL import Image
from werkzeug.utils import secure_filename


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


TESSERACT_PATH = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


ALLOWED_IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}

ALLOWED_PDF_EXTENSIONS = {
    "pdf"
}


def allowed_file(filename: str) -> bool:
    """
    Check whether the uploaded file has a supported extension.
    """

    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return (
        extension in ALLOWED_IMAGE_EXTENSIONS
        or extension in ALLOWED_PDF_EXTENSIONS
    )


def get_extension(filename: str) -> str:
    """
    Return lowercase file extension.
    """

    return filename.rsplit(".", 1)[1].lower()


def sanitize_filename(filename: str) -> str:
    """
    Safely sanitize an uploaded filename.
    """

    return secure_filename(filename)


def extract_text_from_image(file_bytes: bytes) -> str:
    """
    Extract text from an image using Tesseract OCR.
    """

    image = Image.open(
        io.BytesIO(file_bytes)
    )

    # Convert to RGB for OCR compatibility
    image = image.convert("RGB")

    text = pytesseract.image_to_string(
        image,
        config="--psm 6"
    )

    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, bool]:
    """
    Extract text from a PDF.

    Returns:
        (text, used_ocr)
    """

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    extracted_pages = []

    for page in pdf:
        page_text = page.get_text("text")

        if page_text:
            extracted_pages.append(
                page_text
            )

    text = "\n".join(
        extracted_pages
    ).strip()

    # Normal text-based PDF
    if len(text.split()) >= 10:
        pdf.close()

        return text, False

    # --------------------------------------------------------
    # OCR FALLBACK FOR SCANNED PDF
    # --------------------------------------------------------

    ocr_pages = []

    for page in pdf:

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        page_text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        if page_text:
            ocr_pages.append(
                page_text
            )

    pdf.close()

    ocr_text = "\n".join(
        ocr_pages
    ).strip()

    return ocr_text, True


def extract_text_from_file(
    filename: str,
    file_bytes: bytes
) -> tuple[str, str, bool]:
    """
    Extract text from an uploaded image or PDF.

    Returns:
        text,
        source_type,
        used_ocr
    """

    extension = get_extension(filename)

    if extension in ALLOWED_IMAGE_EXTENSIONS:

        text = extract_text_from_image(
            file_bytes
        )

        return text, "Image", True

    if extension in ALLOWED_PDF_EXTENSIONS:

        text, used_ocr = extract_text_from_pdf(
            file_bytes
        )

        source_type = (
            "Scanned PDF"
            if used_ocr
            else "PDF"
        )

        return text, source_type, used_ocr

    raise ValueError(
        "Unsupported file type."
    )


def normalize_extracted_text(text: str) -> str:
    """
    Clean obvious OCR artifacts while preserving
    meaningful article text.
    """

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()