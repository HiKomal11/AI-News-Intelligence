from pathlib import Path
from uuid import uuid4
import traceback
import pickle

import joblib
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from PIL import Image, ImageDraw, ImageFont

<<<<<<< HEAD

=======
>>>>>>> 9d23a86 (Optimize TensorFlow LSTM memory usage)


# ============================================================
# INTERNAL IMPORTS
# ============================================================

from src.preprocessing.text_cleaner import clean_text

from src.upload_utils import (
    allowed_file,
    extract_text_from_file,
    normalize_extracted_text
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

UPLOAD_DIR = BASE_DIR / "uploads"

GENERATED_DIR = BASE_DIR / "results" / "generated"


UPLOAD_DIR.mkdir(
    exist_ok=True
)

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE SETTINGS
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading TF-IDF vectorizer...")

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("Loading Logistic Regression model...")

model = joblib.load(
    MODEL_DIR / "logistic_regression.pkl"
)

print("TF-IDF model loaded successfully.")


# ============================================================
# LSTM TFLITE MODEL
# ============================================================

lstm_interpreter = None
lstm_input_details = None
lstm_output_details = None
lstm_tokenizer = None
def load_lstm_model():

    global lstm_interpreter
    global lstm_input_details
    global lstm_output_details
    global lstm_tokenizer

    if (
        lstm_interpreter is not None
        and lstm_tokenizer is not None
    ):
        return

    try:

        print("======================================")
        print("Loading LSTM TFLite model...")
        print("======================================")

        from ai_edge_litert.interpreter import Interpreter

        model_path = (
            MODEL_DIR /
            "lstm_news_classifier.tflite"
        )

        tokenizer_path = (
            MODEL_DIR /
            "lstm_tokenizer.pkl"
        )

        lstm_interpreter = Interpreter(
            model_path=str(model_path)
        )

        lstm_interpreter.allocate_tensors()

        lstm_input_details = (
            lstm_interpreter.get_input_details()
        )

        lstm_output_details = (
            lstm_interpreter.get_output_details()
        )

        print("TFLite model loaded successfully.")

        print("Loading LSTM tokenizer...")

        lstm_tokenizer = joblib.load(
            tokenizer_path
        )

        print("LSTM tokenizer loaded successfully.")

        print("Input details:")
        print(lstm_input_details)

        print("Output details:")
        print(lstm_output_details)

        print("LSTM initialization completed.")

    except Exception as e:

        print("======================================")
        print("LSTM TFLITE ERROR")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))
        print("======================================")

        lstm_interpreter = None
        lstm_input_details = None
        lstm_output_details = None
        lstm_tokenizer = None

        raise RuntimeError(
            f"Failed to load LSTM TFLite model: {str(e)}"
        )
# ============================================================
# LSTM SETTINGS
# ============================================================

LSTM_MAX_SEQUENCE_LENGTH = 200


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(
    text: str,
    selected_model: str
) -> dict:

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(
            "No usable text was found after preprocessing."
        )

    # ========================================================
    # TF-IDF + LOGISTIC REGRESSION
    # ========================================================

    if selected_model == "tfidf":

        text_vector = vectorizer.transform(
            [cleaned_text]
        )

        prediction = int(
            model.predict(text_vector)[0]
        )

        probabilities = model.predict_proba(
            text_vector
        )[0]

        confidence = float(
            max(probabilities) * 100
        )

        model_name = (
            "TF-IDF + Logistic Regression"
        )

    # ========================================================
    # LSTM TFLITE
    # ========================================================

    elif selected_model == "lstm":

        # Load TFLite model + tokenizer
        load_lstm_model()

        # ----------------------------------------------------
        # TEXT → TOKEN SEQUENCE
        # ----------------------------------------------------

        sequence = lstm_tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        sequence = sequence[0][
            :LSTM_MAX_SEQUENCE_LENGTH
        ]

        # ----------------------------------------------------
        # PADDING
        # IMPORTANT: TFLite model expects INT32
        # ----------------------------------------------------
        input_dtype = lstm_input_details[0]["dtype"]

        padded_sequence = np.zeros(
            (
                1,
                LSTM_MAX_SEQUENCE_LENGTH
            ),
            dtype=input_dtype
        )

        if sequence:

            padded_sequence[
                0,
                :len(sequence)
            ] = sequence

        # ----------------------------------------------------
        # GET TFLITE INPUT / OUTPUT INDEX
        # ----------------------------------------------------

        input_index = (
            lstm_input_details[0]["index"]
        )

        output_index = (
            lstm_output_details[0]["index"]
        )

        # ----------------------------------------------------
        # RUN TFLITE MODEL
        # ----------------------------------------------------

        lstm_interpreter.set_tensor(
            input_index,
            padded_sequence
        )

        lstm_interpreter.invoke()

        # ----------------------------------------------------
        # GET PROBABILITY
        # ----------------------------------------------------

        probability = float(
            lstm_interpreter
            .get_tensor(output_index)[0][0]
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        #
        # 0 = Fake
        # 1 = Real
        # ----------------------------------------------------

        if probability >= 0.5:

            prediction = 1

            confidence = (
                probability * 100
            )

        else:

            prediction = 0

            confidence = (
                (1 - probability) * 100
            )

        model_name = (
            "LSTM Neural Network (TFLite)"
        )

    # ========================================================
    # INVALID MODEL
    # ========================================================

    else:

        raise ValueError(
            f"Unknown model selected: {selected_model}"
        )

    # ========================================================
    # LABEL
    # ========================================================

    if prediction == 0:

        label = "FAKE"
        result = "Fake News"

    else:

        label = "REAL"
        result = "Real News"

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "prediction": label,

        "result": result,

        "confidence": round(
            confidence,
            2
        ),

        "model": model_name,

        "word_count": len(
            text.split()
        ),

        "character_count": len(
            text
        ),

        "extracted_text": text
    }

# ============================================================
# RESULT PDF
# ============================================================

def generate_result_pdf(
    result: dict,
    source_name: str,
    source_type: str
) -> str:

    result_id = uuid4().hex[:12]

    filename = (
        f"news_analysis_{result_id}.pdf"
    )

    output_path = (
        GENERATED_DIR / filename
    )

    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4
    )

    width, height = A4

    y = height - 60

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        24
    )

    pdf.drawString(
        50,
        y,
        "AI News Intelligence"
    )

    y -= 35

    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        y,
        "News Analysis Report"
    )

    y -= 45

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        y,
        result["result"]
    )

    y -= 35

    pdf.setFont(
        "Helvetica",
        13
    )

    details = [
        (
            "Confidence",
            f'{result["confidence"]}%'
        ),
        (
            "Model",
            result["model"]
        ),
        (
            "Source",
            source_type
        ),
        (
            "File",
            source_name
        ),
        (
            "Words",
            str(result["word_count"])
        ),
        (
            "Characters",
            str(result["character_count"])
        )
    ]

    for label, value in details:

        pdf.setFont(
            "Helvetica-Bold",
            11
        )

        pdf.drawString(
            50,
            y,
            f"{label}:"
        )

        pdf.setFont(
            "Helvetica",
            11
        )

        pdf.drawString(
            150,
            y,
            str(value)
        )

        y -= 23

    y -= 20

    # --------------------------------------------------------
    # EXTRACTED TEXT
    # --------------------------------------------------------

    pdf.setFont(
        "Helvetica-Bold",
        14
    )

    pdf.drawString(
        50,
        y,
        "Extracted / Analyzed Text"
    )

    y -= 25

    pdf.setFont(
        "Helvetica",
        9
    )

    lines = simpleSplit(
        result["extracted_text"],
        "Helvetica",
        9,
        width - 100
    )

    for line in lines:

        if y < 50:

            pdf.showPage()

            y = height - 50

            pdf.setFont(
                "Helvetica",
                9
            )

        pdf.drawString(
            50,
            y,
            line
        )

        y -= 13

    pdf.save()

    return filename


# ============================================================
# RESULT PNG
# ============================================================

def generate_result_png(
    result: dict,
    source_name: str,
    source_type: str
) -> str:

    result_id = uuid4().hex[:12]

    filename = (
        f"news_analysis_{result_id}.png"
    )

    output_path = (
        GENERATED_DIR / filename
    )

    width = 1200
    height = 850

    image = Image.new(
        "RGB",
        (
            width,
            height
        ),
        "white"
    )

    draw = ImageDraw.Draw(
        image
    )

    # --------------------------------------------------------
    # FONTS
    # --------------------------------------------------------

    try:

        font_title = ImageFont.truetype(
            "arial.ttf",
            42
        )

        font_heading = ImageFont.truetype(
            "arialbd.ttf",
            30
        )

        font_normal = ImageFont.truetype(
            "arial.ttf",
            24
        )

        font_small = ImageFont.truetype(
            "arial.ttf",
            18
        )

    except OSError:

        font_title = ImageFont.load_default()
        font_heading = font_title
        font_normal = font_title
        font_small = font_title

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    draw.text(
        (60, 45),
        "AI News Intelligence",
        fill="black",
        font=font_title
    )

    draw.text(
        (60, 105),
        "News Analysis Report",
        fill="gray",
        font=font_normal
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result_text = (
        f'{result["result"]}  |  '
        f'{result["confidence"]}%'
    )

    draw.text(
        (60, 190),
        result_text,
        fill="black",
        font=font_heading
    )

    # --------------------------------------------------------
    # CONFIDENCE BAR
    # --------------------------------------------------------

    bar_x = 60
    bar_y = 260
    bar_width = 1080
    bar_height = 25

    draw.rectangle(
        (
            bar_x,
            bar_y,
            bar_x + bar_width,
            bar_y + bar_height
        ),
        fill="#dddddd"
    )

    filled_width = int(
        bar_width *
        result["confidence"] /
        100
    )

    draw.rectangle(
        (
            bar_x,
            bar_y,
            bar_x + filled_width,
            bar_y + bar_height
        ),
        fill="#222222"
    )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    details = [
        f'Model: {result["model"]}',
        f'Source: {source_type}',
        f'File: {source_name}',
        f'Words: {result["word_count"]}',
        f'Characters: {result["character_count"]}'
    ]

    y = 330

    for detail in details:

        draw.text(
            (60, y),
            detail,
            fill="black",
            font=font_normal
        )

        y += 42

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    draw.text(
        (60, 590),
        "This result is a machine-learning prediction,",
        fill="gray",
        font=font_small
    )

    draw.text(
        (60, 620),
        "not guaranteed factual verification.",
        fill="gray",
        font=font_small
    )

    draw.text(
        (60, 730),
        "AI News Intelligence | 2026",
        fill="gray",
        font=font_small
    )

    image.save(
        output_path,
        "PNG"
    )

    return filename


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# TEXT PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        text = (
            data.get(
                "text",
                ""
            )
            .strip()
        )

        selected_model = data.get(
            "model",
            "tfidf"
        )

        if not text:

            return jsonify({
                "error":
                "Please enter a news article."
            }), 400

        if len(text.split()) < 5:

            return jsonify({
                "error":
                "Please enter a longer news article."
            }), 400


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        result = predict_text(
            text,
            selected_model
        )


        # ----------------------------------------------------
        # SOURCE INFORMATION
        # ----------------------------------------------------

        source_type = "Typed Text"
        source_name = "Text entered by user"

        result["source_type"] = source_type


        # ----------------------------------------------------
        # GENERATE RESULT FILES
        # ----------------------------------------------------

        pdf_filename = generate_result_pdf(
            result,
            source_name,
            source_type
        )

        png_filename = generate_result_png(
            result,
            source_name,
            source_type
        )


        # ----------------------------------------------------
        # DOWNLOAD LINKS
        # ----------------------------------------------------

        result["pdf_url"] = (
            f"/download/{pdf_filename}"
        )

        result["png_url"] = (
            f"/download/{png_filename}"
        )


        return jsonify(
            result
        )


    except Exception as e:

        print("======================================")
        print("PREDICTION ERROR")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))
        traceback.print_exc()
        print("======================================")

        return jsonify({
            "error": str(e)
        }), 500
# ============================================================
# FILE UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    try:

        uploaded_file = request.files.get(
            "file"
        )

        selected_model = request.form.get(
            "model",
            "tfidf"
        )

        if not uploaded_file:

            return jsonify({
                "error":
                "Please select an image or PDF."
            }), 400

        filename = uploaded_file.filename or ""

        if not allowed_file(filename):

            return jsonify({
                "error":
                "Unsupported file type. "
                "Please upload PNG, JPG, JPEG, WEBP, or PDF."
            }), 400

        file_bytes = uploaded_file.read()

        if not file_bytes:

            return jsonify({
                "error":
                "The uploaded file is empty."
            }), 400

        extracted_text, source_type, used_ocr = (
            extract_text_from_file(
                filename,
                file_bytes
            )
        )

        extracted_text = normalize_extracted_text(
            extracted_text
        )

        if not extracted_text:

            return jsonify({
                "error":
                "No readable text could be extracted from the file."
            }), 400

        if len(extracted_text.split()) < 5:

            return jsonify({
                "error":
                "The file does not contain enough readable text."
            }), 400

        result = predict_text(
            extracted_text,
            selected_model
        )

        result["source_type"] = source_type
        result["used_ocr"] = used_ocr
        result["filename"] = filename

        pdf_filename = generate_result_pdf(
            result,
            filename,
            source_type
        )

        png_filename = generate_result_png(
            result,
            filename,
            source_type
        )

        result["pdf_url"] = (
            f"/download/{pdf_filename}"
        )

        result["png_url"] = (
            f"/download/{png_filename}"
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "Upload error:",
            e
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# DOWNLOAD GENERATED FILE
# ============================================================

@app.route(
    "/download/<filename>"
)
def download_result(filename):

    return send_from_directory(
        GENERATED_DIR,
        filename,
        as_attachment=True
    )


# ============================================================
# ERROR HANDLER
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error":
        "File is too large. Maximum size is 16 MB."
    }), 413


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n======================================")
    print("        AI NEWS INTELLIGENCE")
    print("======================================")
    print(f"Starting server on port {port}")
    print("Open: http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )