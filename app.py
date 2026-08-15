from pathlib import Path
from uuid import uuid4
import traceback

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
#===========================================================
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

# 16 MB maximum upload
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ============================================================
# TF-IDF MODEL
# ============================================================

print("======================================")
print("Loading TF-IDF vectorizer...")
print("======================================")

vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

print("TF-IDF vectorizer loaded successfully.")


print("======================================")
print("Loading Logistic Regression model...")
print("======================================")

model = joblib.load(
    MODEL_DIR / "logistic_regression.pkl"
)

print("Logistic Regression model loaded successfully.")


# ============================================================
# LSTM VARIABLES
# ============================================================

# IMPORTANT:
#
# LSTM is NOT loaded when Flask starts.
#
# It is loaded only when the user selects:
#
#     "lstm"
#
# This reduces Render startup memory usage.
# ============================================================

lstm_interpreter = None
lstm_input_details = None
lstm_output_details = None
lstm_tokenizer = None


# ============================================================
# LSTM SETTINGS
# ============================================================

# Your TFLite model input is:
#
# [1, 200]
#
LSTM_MAX_SEQUENCE_LENGTH = 200


# Your tokenizer has:
#
# num_words = 10000
#
# Therefore valid IDs are:
#
# 1 ... 9999
#
LSTM_VOCAB_SIZE = 10000


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "AI News Intelligence"
    }), 200


# ============================================================
# LOAD LSTM MODEL
# ============================================================

def load_lstm_model():

    global lstm_interpreter
    global lstm_input_details
    global lstm_output_details
    global lstm_tokenizer

    # --------------------------------------------------------
    # Already loaded
    # --------------------------------------------------------

    if (
        lstm_interpreter is not None
        and lstm_tokenizer is not None
    ):
        return

    try:

        print("\n======================================")
        print("Loading LSTM TFLite model...")
        print("======================================")

        # ----------------------------------------------------
        # LiteRT
        # ----------------------------------------------------

        from ai_edge_litert.interpreter import Interpreter

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model_path = (
            MODEL_DIR /
            "lstm_news_classifier.tflite"
        )

        # ----------------------------------------------------
        # TOKENIZER
        #
        # IMPORTANT:
        #
        # We use lstm_tokenizer.pkl
        #
        # NOT:
        #
        # lstm_word_index.pkl
        # ----------------------------------------------------

        tokenizer_path = (
            MODEL_DIR /
            "lstm_tokenizer.pkl"
        )

        print(
            "TFLite model:",
            model_path
        )

        print(
            "Tokenizer:",
            tokenizer_path
        )

        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if not model_path.exists():

            raise FileNotFoundError(
                f"TFLite model not found: {model_path}"
            )

        # ----------------------------------------------------
        # CHECK TOKENIZER
        # ----------------------------------------------------

        if not tokenizer_path.exists():

            raise FileNotFoundError(
                f"LSTM tokenizer not found: {tokenizer_path}"
            )

        # ----------------------------------------------------
        # LOAD TFLITE
        # ----------------------------------------------------

        lstm_interpreter = Interpreter(
            model_path=str(model_path),
            num_threads=1
        )

        lstm_interpreter.allocate_tensors()

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        lstm_input_details = (
            lstm_interpreter.get_input_details()
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        lstm_output_details = (
            lstm_interpreter.get_output_details()
        )

        print(
            "TFLite model loaded successfully."
        )

        # ----------------------------------------------------
        # VERIFY INPUT
        # ----------------------------------------------------

        if not lstm_input_details:

            raise RuntimeError(
                "TFLite model has no input tensor."
            )

        input_shape = (
            lstm_input_details[0]["shape"]
        )

        input_dtype = (
            lstm_input_details[0]["dtype"]
        )

        print(
            "LSTM input shape:",
            input_shape
        )

        print(
            "LSTM input dtype:",
            input_dtype
        )

        # ----------------------------------------------------
        # VERIFY OUTPUT
        # ----------------------------------------------------

        if not lstm_output_details:

            raise RuntimeError(
                "TFLite model has no output tensor."
            )

        print(
            "LSTM output shape:",
            lstm_output_details[0]["shape"]
        )

        print(
            "LSTM output dtype:",
            lstm_output_details[0]["dtype"]
        )

        # ----------------------------------------------------
        # LOAD TOKENIZER
        # ----------------------------------------------------

        print(
            "Loading LSTM tokenizer..."
        )

        lstm_tokenizer = joblib.load(
            tokenizer_path
        )

        print(
            "LSTM tokenizer loaded successfully."
        )

        # ----------------------------------------------------
        # TOKENIZER INFORMATION
        # ----------------------------------------------------

        tokenizer_num_words = getattr(
            lstm_tokenizer,
            "num_words",
            None
        )

        tokenizer_word_index = getattr(
            lstm_tokenizer,
            "word_index",
            {}
        )

        print(
            "Tokenizer num_words:",
            tokenizer_num_words
        )

        print(
            "Tokenizer vocabulary:",
            len(tokenizer_word_index)
        )

        if tokenizer_word_index:

            print(
                "Tokenizer maximum index:",
                max(
                    tokenizer_word_index.values()
                )
            )

        # ----------------------------------------------------
        # FINAL
        # ----------------------------------------------------

        print(
            "LSTM vocabulary limit:",
            LSTM_VOCAB_SIZE
        )

        print(
            "LSTM sequence length:",
            LSTM_MAX_SEQUENCE_LENGTH
        )

        print(
            "LSTM initialization completed."
        )

        print(
            "======================================\n"
        )

    except Exception as e:

        print(
            "\n======================================"
        )

        print(
            "LSTM TFLITE ERROR"
        )

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print(
            "======================================\n"
        )

        # Reset
        lstm_interpreter = None
        lstm_input_details = None
        lstm_output_details = None
        lstm_tokenizer = None

        raise RuntimeError(
            f"Failed to load LSTM model: {str(e)}"
        )


# ============================================================
# LSTM TOKENIZATION
# ============================================================

def prepare_lstm_input(text):

    if lstm_tokenizer is None:

        raise RuntimeError(
            "LSTM tokenizer has not been loaded."
        )

    # --------------------------------------------------------
    # Convert text to token IDs
    # --------------------------------------------------------

    sequences = (
        lstm_tokenizer.texts_to_sequences(
            [text]
        )
    )

    if not sequences:

        sequence = []

    else:

        sequence = sequences[0]

    # --------------------------------------------------------
    # SAFETY FILTER
    #
    # This is important because:
    #
    # tokenizer vocabulary = 91800
    #
    # model vocabulary = 10000
    #
    # Any ID >= 10000 can cause:
    #
    # GATHER index out of bounds
    # --------------------------------------------------------

    safe_sequence = []

    for token_id in sequence:

        token_id = int(token_id)

        if (
            token_id > 0
            and token_id < LSTM_VOCAB_SIZE
        ):

            safe_sequence.append(
                token_id
            )

    # --------------------------------------------------------
    # Limit to 200 tokens
    # --------------------------------------------------------

    safe_sequence = safe_sequence[
        :LSTM_MAX_SEQUENCE_LENGTH
    ]

    # --------------------------------------------------------
    # Get model dtype
    # --------------------------------------------------------

    input_dtype = (
        lstm_input_details[0]["dtype"]
    )

    # --------------------------------------------------------
    # Create input
    # --------------------------------------------------------

    padded_sequence = np.zeros(
        (
            1,
            LSTM_MAX_SEQUENCE_LENGTH
        ),
        dtype=input_dtype
    )

    # --------------------------------------------------------
    # Insert tokens
    # --------------------------------------------------------

    if safe_sequence:

        padded_sequence[
            0,
            :len(safe_sequence)
        ] = safe_sequence

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    print(
        "Original token count:",
        len(sequence)
    )

    print(
        "Valid token count:",
        len(safe_sequence)
    )

    if safe_sequence:

        print(
            "Minimum token ID:",
            min(safe_sequence)
        )

        print(
            "Maximum token ID:",
            max(safe_sequence)
        )

    else:

        print(
            "No valid vocabulary tokens found."
        )

    print(
        "Input shape:",
        padded_sequence.shape
    )

    print(
        "Input dtype:",
        padded_sequence.dtype
    )

    return padded_sequence


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(
    text: str,
    selected_model: str
) -> dict:

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    cleaned_text = clean_text(text)

    if not cleaned_text:

        raise ValueError(
            "No usable text was found after preprocessing."
        )

    # Normalize model name
    selected_model = (
        str(selected_model)
        .strip()
        .lower()
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    if selected_model == "tfidf":

        print(
            "Running TF-IDF prediction..."
        )

        # ----------------------------------------------------
        # Vectorize
        # ----------------------------------------------------

        text_vector = (
            vectorizer.transform(
                [cleaned_text]
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            model.predict(
                text_vector
            )[0]
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                text_vector
            )[0]
        )

        confidence = float(
            max(probabilities) * 100
        )

        model_name = (
            "TF-IDF + Logistic Regression"
        )

        print(
            "TF-IDF prediction completed."
        )

    # ========================================================
    # LSTM
    # ========================================================

    elif selected_model == "lstm":

        print(
            "\n======================================"
        )

        print(
            "Running LSTM TFLite prediction..."
        )

        print(
            "======================================"
        )

        # ----------------------------------------------------
        # Load only now
        # ----------------------------------------------------

        load_lstm_model()

        # ----------------------------------------------------
        # Prepare input
        # ----------------------------------------------------

        padded_sequence = (
            prepare_lstm_input(
                cleaned_text
            )
        )

        # ----------------------------------------------------
        # Input index
        # ----------------------------------------------------

        input_index = (
            lstm_input_details[0]["index"]
        )

        # ----------------------------------------------------
        # Output index
        # ----------------------------------------------------

        output_index = (
            lstm_output_details[0]["index"]
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        expected_shape = (
            tuple(
                lstm_input_details[0]["shape"]
            )
        )

        actual_shape = (
            tuple(
                padded_sequence.shape
            )
        )

        print(
            "Expected input shape:",
            expected_shape
        )

        print(
            "Actual input shape:",
            actual_shape
        )

        if actual_shape != expected_shape:

            raise RuntimeError(
                f"LSTM input shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}"
            )

        # ----------------------------------------------------
        # Set input
        # ----------------------------------------------------

        lstm_interpreter.set_tensor(
            input_index,
            padded_sequence
        )

        print(
            "Input tensor set successfully."
        )

        # ----------------------------------------------------
        # Invoke
        # ----------------------------------------------------

        print(
            "Invoking LSTM TFLite model..."
        )

        lstm_interpreter.invoke()

        print(
            "TFLite model invoked successfully."
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output = (
            lstm_interpreter.get_tensor(
                output_index
            )
        )

        print(
            "Raw output:",
            output
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if output is None:

            raise RuntimeError(
                "LSTM returned no output."
            )

        output = np.asarray(
            output
        )

        if output.size == 0:

            raise RuntimeError(
                "LSTM returned empty output."
            )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = float(
            output.reshape(-1)[0]
        )

        # ----------------------------------------------------
        # Safety clamp
        # ----------------------------------------------------

        probability = max(
            0.0,
            min(
                1.0,
                probability
            )
        )

        print(
            "LSTM probability:",
            probability
        )

        # ----------------------------------------------------
        # Classification
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

        print(
            "LSTM prediction completed."
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
    # RESULT
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
    # Header
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
    # Result
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

    # --------------------------------------------------------
    # Details
    # --------------------------------------------------------

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
    # Text
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
    # Fonts
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
    # Header
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
    # Result
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
    # Confidence bar
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
    # Details
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
    # Disclaimer
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

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "error":
                "Invalid JSON request."
            }), 400

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text = (
            data.get(
                "text",
                ""
            )
            .strip()
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        selected_model = data.get(
            "model",
            "tfidf"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

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

        print(
            "\n======================================"
        )

        print(
            "POST /predict"
        )

        print(
            "Selected model:",
            selected_model
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predict_text(
            text,
            selected_model
        )

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        source_type = "Typed Text"

        source_name = "Text entered by user"

        result["source_type"] = source_type

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        pdf_filename = generate_result_pdf(
            result,
            source_name,
            source_type
        )

        # ----------------------------------------------------
        # PNG
        # ----------------------------------------------------

        png_filename = generate_result_png(
            result,
            source_name,
            source_type
        )

        # ----------------------------------------------------
        # URLs
        # ----------------------------------------------------

        result["pdf_url"] = (
            f"/download/{pdf_filename}"
        )

        result["png_url"] = (
            f"/download/{png_filename}"
        )

        print(
            "Prediction successful."
        )

        print(
            "======================================\n"
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "\n======================================"
        )

        print(
            "PREDICTION ERROR"
        )

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print(
            "======================================\n"
        )

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

        # ----------------------------------------------------
        # File check
        # ----------------------------------------------------

        if not uploaded_file:

            return jsonify({
                "error":
                "Please select an image or PDF."
            }), 400

        filename = (
            uploaded_file.filename or ""
        )

        # ----------------------------------------------------
        # Extension
        # ----------------------------------------------------

        if not allowed_file(filename):

            return jsonify({
                "error":
                "Unsupported file type. "
                "Please upload PNG, JPG, JPEG, WEBP, or PDF."
            }), 400

        # ----------------------------------------------------
        # Read
        # ----------------------------------------------------

        file_bytes = uploaded_file.read()

        if not file_bytes:

            return jsonify({
                "error":
                "The uploaded file is empty."
            }), 400

        print(
            "\n======================================"
        )

        print(
            "POST /upload"
        )

        print(
            "Filename:",
            filename
        )

        print(
            "Selected model:",
            selected_model
        )

        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        (
            extracted_text,
            source_type,
            used_ocr
        ) = extract_text_from_file(
            filename,
            file_bytes
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        extracted_text = (
            normalize_extracted_text(
                extracted_text
            )
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predict_text(
            extracted_text,
            selected_model
        )

        # ----------------------------------------------------
        # Source
        # ----------------------------------------------------

        result["source_type"] = source_type

        result["used_ocr"] = used_ocr

        result["filename"] = filename

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

        pdf_filename = generate_result_pdf(
            result,
            filename,
            source_type
        )

        # ----------------------------------------------------
        # PNG
        # ----------------------------------------------------

        png_filename = generate_result_png(
            result,
            filename,
            source_type
        )

        # ----------------------------------------------------
        # URLs
        # ----------------------------------------------------

        result["pdf_url"] = (
            f"/download/{pdf_filename}"
        )

        result["png_url"] = (
            f"/download/{png_filename}"
        )

        print(
            "Upload prediction successful."
        )

        print(
            "======================================\n"
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            "\n======================================"
        )

        print(
            "UPLOAD ERROR"
        )

        print(
            "Error type:",
            type(e).__name__
        )

        print(
            "Error:",
            str(e)
        )

        traceback.print_exc()

        print(
            "======================================\n"
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# DOWNLOAD
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
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "error":
        "File is too large. Maximum size is 16 MB."
    }), 413


# ============================================================
# GENERAL ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({
        "error":
        "Internal server error. Check the Render logs."
    }), 500


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":

    print("\n======================================")
    print("        AI NEWS INTELLIGENCE")
    print("======================================")
    print("Starting local development server...")
    print("Open: http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )