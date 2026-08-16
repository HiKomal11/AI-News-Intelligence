from pathlib import Path
from uuid import uuid4
import os
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
    parents=True,
    exist_ok=True
)

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILE SETTINGS
# ============================================================

# Maximum uploaded file size: 16 MB

app.config["MAX_CONTENT_LENGTH"] = (
    16 * 1024 * 1024
)


# ============================================================
# ENVIRONMENT DETECTION
# ============================================================
#
# LOCAL:
#     LSTM available
#
# RENDER:
#     LSTM disabled
#
# Render automatically provides the RENDER environment
# variable.
#
# We also support:
#
#     AI_NEWS_ENV=render
#
# or:
#
#     AI_NEWS_ENABLE_LSTM=0
#
# ============================================================

def is_render_environment():

    # --------------------------------------------------------
    # Render's built-in environment variable
    # --------------------------------------------------------

    render_variable = (
        os.getenv(
            "RENDER",
            ""
        )
        .strip()
        .lower()
    )

    if render_variable in {
        "1",
        "true",
        "yes",
        "on"
    }:

        return True


    # --------------------------------------------------------
    # Optional custom environment variable
    # --------------------------------------------------------

    ai_news_env = (
        os.getenv(
            "AI_NEWS_ENV",
            ""
        )
        .strip()
        .lower()
    )

    if ai_news_env == "render":

        return True


    # --------------------------------------------------------
    # Otherwise assume local
    # --------------------------------------------------------

    return False


# ============================================================
# ENVIRONMENT NAME
# ============================================================

IS_RENDER = is_render_environment()

ENVIRONMENT_NAME = (
    "render"
    if IS_RENDER
    else "local"
)


# ============================================================
# LSTM AVAILABILITY
# ============================================================
#
# Default behavior:
#
# LOCAL  -> LSTM enabled
# RENDER -> LSTM disabled
#
# Optional override:
#
# AI_NEWS_ENABLE_LSTM=1
# AI_NEWS_ENABLE_LSTM=0
#
# IMPORTANT:
#
# Render is always blocked unless you intentionally change
# this code.
#
# This prevents accidental LSTM activation on Render.
# ============================================================

lstm_override = (
    os.getenv(
        "AI_NEWS_ENABLE_LSTM",
        ""
    )
    .strip()
    .lower()
)


# ============================================================
# LSTM AVAILABILITY
# ============================================================
#
# TESTING CONFIGURATION:
#
# Local  -> TF-IDF + LSTM
# Render -> TF-IDF + LSTM
#
# AI_NEWS_ENABLE_LSTM can explicitly disable LSTM.
# ============================================================

if lstm_override in {
    "0",
    "false",
    "no",
    "off"
}:

    ENABLE_LSTM = False

else:

    ENABLE_LSTM = True

# ============================================================
# APPLICATION STARTUP INFORMATION
# ============================================================

print()
print("======================================")
print("       AI NEWS INTELLIGENCE")
print("======================================")

print(
    "Environment:",
    ENVIRONMENT_NAME
)

print(
    "Render detected:",
    IS_RENDER
)

print(
    "LSTM enabled:",
    ENABLE_LSTM
)


if ENABLE_LSTM:

    print(
        "Available models: TF-IDF + LSTM"
    )

else:

    print(
        "Available models: TF-IDF only"
    )

print("======================================")
print()


# ============================================================
# LOAD TF-IDF MODEL
# ============================================================

print("======================================")
print("Loading TF-IDF vectorizer...")
print("======================================")


TFIDF_VECTOR_PATH = (
    MODEL_DIR /
    "tfidf_vectorizer.pkl"
)


if not TFIDF_VECTOR_PATH.exists():

    raise FileNotFoundError(
        "TF-IDF vectorizer not found: "
        f"{TFIDF_VECTOR_PATH}"
    )


vectorizer = joblib.load(
    TFIDF_VECTOR_PATH
)


print(
    "TF-IDF vectorizer loaded successfully."
)


# ============================================================
# LOAD LOGISTIC REGRESSION
# ============================================================

print("======================================")
print("Loading Logistic Regression model...")
print("======================================")


LOGISTIC_MODEL_PATH = (
    MODEL_DIR /
    "logistic_regression.pkl"
)


if not LOGISTIC_MODEL_PATH.exists():

    raise FileNotFoundError(
        "Logistic Regression model not found: "
        f"{LOGISTIC_MODEL_PATH}"
    )


model = joblib.load(
    LOGISTIC_MODEL_PATH
)


print(
    "Logistic Regression model loaded successfully."
)


# ============================================================
# LSTM VARIABLES
# ============================================================
#
# LSTM is NOT loaded when Flask starts.
#
# It is loaded only when:
#
#     ENABLE_LSTM == True
#
# AND
#
#     user selects "lstm"
#
# On Render:
#
#     ENABLE_LSTM == False
#
# Therefore the TFLite model and tokenizer are never loaded.
#
# ============================================================

lstm_interpreter = None

lstm_input_details = None

lstm_output_details = None

lstm_tokenizer = None


# ============================================================
# LSTM SETTINGS
# ============================================================

LSTM_MAX_SEQUENCE_LENGTH = 200

LSTM_VOCAB_SIZE = 10000


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "ok",

        "service":
            "AI News Intelligence",

        "environment":
            ENVIRONMENT_NAME,

        "render":
            IS_RENDER,

        "tfidf":
            True,

        "lstm":
            ENABLE_LSTM,

        "available_models":
            (
                ["tfidf", "lstm"]
                if ENABLE_LSTM
                else
                ["tfidf"]
            )

    }), 200


# ============================================================
# AVAILABLE MODELS
# ============================================================

@app.route(
    "/models",
    methods=["GET"]
)
def available_models():

    models = {

        "tfidf": {

            "name":
                "TF-IDF + Logistic Regression",

            "available":
                True

        },

        "lstm": {

            "name":
                "LSTM Neural Network (TFLite)",

            "available":
                ENABLE_LSTM

        }

    }


    return jsonify({

        "environment":
            ENVIRONMENT_NAME,

        "models":
            models

    }), 200


# ============================================================
# LOAD LSTM MODEL
# ============================================================

def load_lstm_model():

    global lstm_interpreter
    global lstm_input_details
    global lstm_output_details
    global lstm_tokenizer


    # ========================================================
    # HARD BLOCK WHEN LSTM IS DISABLED
    # ========================================================

    if not ENABLE_LSTM:

        raise RuntimeError(
            "LSTM is not available on this deployment. "
            "Please select TF-IDF."
        )


    # ========================================================
    # ALREADY LOADED
    # ========================================================

    if (
        lstm_interpreter is not None
        and
        lstm_tokenizer is not None
        and
        lstm_input_details is not None
        and
        lstm_output_details is not None
    ):

        return


    try:

        print()
        print("======================================")
        print("Loading LSTM TFLite model...")
        print("======================================")


        # ====================================================
        # LiteRT
        # ====================================================

        from ai_edge_litert.interpreter import Interpreter


        # ====================================================
        # MODEL PATH
        # ====================================================

        model_path = (
            MODEL_DIR /
            "lstm_news_classifier.tflite"
        )


        # ====================================================
        # TOKENIZER PATH
        # ====================================================

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


        # ====================================================
        # CHECK MODEL
        # ====================================================

        if not model_path.exists():

            raise FileNotFoundError(
                "TFLite model not found: "
                f"{model_path}"
            )


        # ====================================================
        # CHECK TOKENIZER
        # ====================================================

        if not tokenizer_path.exists():

            raise FileNotFoundError(
                "LSTM tokenizer not found: "
                f"{tokenizer_path}"
            )


        # ====================================================
        # LOAD INTERPRETER
        # ====================================================

        lstm_interpreter = Interpreter(
            model_path=str(model_path),
            num_threads=1
        )


        # ====================================================
        # ALLOCATE TENSORS
        # ====================================================

        lstm_interpreter.allocate_tensors()


        # ====================================================
        # INPUT DETAILS
        # ====================================================

        lstm_input_details = (
            lstm_interpreter.get_input_details()
        )


        if not lstm_input_details:

            raise RuntimeError(
                "LSTM model has no input tensor."
            )


        # ====================================================
        # OUTPUT DETAILS
        # ====================================================

        lstm_output_details = (
            lstm_interpreter.get_output_details()
        )


        if not lstm_output_details:

            raise RuntimeError(
                "LSTM model has no output tensor."
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        print(
            "LSTM input shape:",
            lstm_input_details[0]["shape"]
        )

        print(
            "LSTM input dtype:",
            lstm_input_details[0]["dtype"]
        )

        print(
            "LSTM output shape:",
            lstm_output_details[0]["shape"]
        )

        print(
            "LSTM output dtype:",
            lstm_output_details[0]["dtype"]
        )


        # ====================================================
        # LOAD TOKENIZER
        # ====================================================

        print(
            "Loading LSTM tokenizer..."
        )


        lstm_tokenizer = joblib.load(
            tokenizer_path
        )


        print(
            "LSTM tokenizer loaded successfully."
        )


        # ====================================================
        # TOKENIZER INFORMATION
        # ====================================================

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
            "======================================"
        )

        print()


    except Exception as e:

        print()
        print("======================================")
        print("LSTM TFLITE ERROR")
        print("======================================")


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
            "======================================"
        )

        print()


        # ====================================================
        # RESET
        # ====================================================

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


    if lstm_input_details is None:

        raise RuntimeError(
            "LSTM input details are unavailable."
        )


    # ========================================================
    # TEXT -> TOKEN IDS
    # ========================================================

    sequences = (
        lstm_tokenizer.texts_to_sequences(
            [text]
        )
    )


    if not sequences:

        sequence = []

    else:

        sequence = sequences[0]


    # ========================================================
    # SAFETY FILTER
    #
    # Valid IDs:
    #
    # 1 <= token ID < 10000
    #
    # ========================================================

    safe_sequence = []


    for token_id in sequence:

        try:

            token_id = int(
                token_id
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        if (
            token_id > 0
            and
            token_id < LSTM_VOCAB_SIZE
        ):

            safe_sequence.append(
                token_id
            )


    # ========================================================
    # LIMIT SEQUENCE LENGTH
    # ========================================================

    safe_sequence = (
        safe_sequence[
            :LSTM_MAX_SEQUENCE_LENGTH
        ]
    )


    # ========================================================
    # MODEL INPUT DTYPE
    # ========================================================

    input_dtype = (
        lstm_input_details[0]["dtype"]
    )


    # ========================================================
    # CREATE PADDED INPUT
    # ========================================================

    padded_sequence = np.zeros(
        (
            1,
            LSTM_MAX_SEQUENCE_LENGTH
        ),
        dtype=input_dtype
    )


    # ========================================================
    # INSERT TOKENS
    # ========================================================

    if safe_sequence:

        padded_sequence[
            0,
            :len(safe_sequence)
        ] = safe_sequence


    # ========================================================
    # DEBUG
    # ========================================================

    print(
        "Original token count:",
        len(sequence)
    )

    print(
        "Valid token count:",
        len(safe_sequence)
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

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    cleaned_text = clean_text(
        text
    )


    if not cleaned_text:

        raise ValueError(
            "No usable text was found after preprocessing."
        )


    # ========================================================
    # NORMALIZE MODEL
    # ========================================================

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


        # ====================================================
        # VECTORIZE
        # ====================================================

        text_vector = (
            vectorizer.transform(
                [cleaned_text]
            )
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = int(
            model.predict(
                text_vector
            )[0]
        )


        # ====================================================
        # PROBABILITY
        # ====================================================

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    text_vector
                )[0]
            )


            confidence = float(
                max(probabilities) * 100
            )

        else:

            confidence = 0.0


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

        # ----------------------------------------------------
        # Render / disabled protection
        # ----------------------------------------------------

        if not ENABLE_LSTM:

            raise RuntimeError(
                "LSTM is not available on the deployed "
                "server. Please select TF-IDF."
            )


        print()
        print("======================================")
        print("Running LSTM TFLite prediction...")
        print("======================================")


        # ====================================================
        # LOAD LSTM
        # ====================================================

        load_lstm_model()


        # ====================================================
        # PREPARE INPUT
        # ====================================================

        padded_sequence = (
            prepare_lstm_input(
                cleaned_text
            )
        )


        # ====================================================
        # INPUT INDEX
        # ====================================================

        input_index = (
            lstm_input_details[0]["index"]
        )


        # ====================================================
        # OUTPUT INDEX
        # ====================================================

        output_index = (
            lstm_output_details[0]["index"]
        )


        # ====================================================
        # SHAPE VALIDATION
        # ====================================================

        expected_shape = tuple(
            lstm_input_details[0]["shape"]
        )


        actual_shape = tuple(
            padded_sequence.shape
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
                "LSTM input shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}"
            )


        # ====================================================
        # SET INPUT
        # ====================================================

        lstm_interpreter.set_tensor(
            input_index,
            padded_sequence
        )


        # ====================================================
        # RUN MODEL
        # ====================================================

        lstm_interpreter.invoke()


        # ====================================================
        # GET OUTPUT
        # ====================================================

        output = (
            lstm_interpreter.get_tensor(
                output_index
            )
        )


        print(
            "Raw LSTM output:",
            output
        )


        # ====================================================
        # VALIDATE OUTPUT
        # ====================================================

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


        # ====================================================
        # PROBABILITY
        # ====================================================

        probability = float(
            output.reshape(-1)[0]
        )


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


        # ====================================================
        # CLASSIFICATION
        #
        # 0 = Fake
        # 1 = Real
        # ====================================================

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
    # RETURN
    # ========================================================

    return {

        "prediction":
            label,

        "result":
            result,

        "confidence":
            round(
                confidence,
                2
            ),

        "model":
            model_name,

        "word_count":
            len(text.split()),

        "character_count":
            len(text),

        "extracted_text":
            text
    }


# ============================================================
# RESULT PDF
# ============================================================

def generate_result_pdf(
    result: dict,
    source_name: str,
    source_type: str
) -> str:

    result_id = (
        uuid4().hex[:12]
    )


    filename = (
        f"news_analysis_{result_id}.pdf"
    )


    output_path = (
        GENERATED_DIR /
        filename
    )


    pdf = canvas.Canvas(
        str(output_path),
        pagesize=A4
    )


    width, height = A4


    y = height - 60


    # ========================================================
    # HEADER
    # ========================================================

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


    # ========================================================
    # RESULT
    # ========================================================

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


    # ========================================================
    # DETAILS
    # ========================================================

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


    # ========================================================
    # ANALYZED TEXT
    # ========================================================

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


    analyzed_text = str(
        result.get(
            "extracted_text",
            ""
        )
    )


    lines = simpleSplit(
        analyzed_text,
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

    result_id = (
        uuid4().hex[:12]
    )


    filename = (
        f"news_analysis_{result_id}.png"
    )


    output_path = (
        GENERATED_DIR /
        filename
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


    # ========================================================
    # FONTS
    # ========================================================

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


    # ========================================================
    # HEADER
    # ========================================================

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


    # ========================================================
    # RESULT
    # ========================================================

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


    # ========================================================
    # CONFIDENCE BAR
    # ========================================================

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


    confidence_value = float(
        result.get(
            "confidence",
            0
        )
    )


    confidence_value = max(
        0,
        min(
            100,
            confidence_value
        )
    )


    filled_width = int(
        bar_width *
        confidence_value /
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


    # ========================================================
    # DETAILS
    # ========================================================

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


    # ========================================================
    # DISCLAIMER
    # ========================================================

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
        "index.html",
        enable_lstm=ENABLE_LSTM,
        environment=ENVIRONMENT_NAME
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


        # ====================================================
        # TEXT
        # ====================================================

        text = (
            data.get(
                "text",
                ""
            )
            .strip()
        )


        # ====================================================
        # MODEL
        # ====================================================

        selected_model = data.get(
            "model",
            "tfidf"
        )


        selected_model = (
            str(selected_model)
            .strip()
            .lower()
        )


        # ====================================================
        # VALID MODEL
        # ====================================================

        if selected_model not in {
            "tfidf",
            "lstm"
        }:

            return jsonify({

                "error":
                    "Invalid model selected. "
                    "Choose TF-IDF or LSTM."

            }), 400


        # ====================================================
        # RENDER LSTM BLOCK
        # ====================================================

        if (
            selected_model == "lstm"
            and
            not ENABLE_LSTM
        ):

            return jsonify({

                "error":
                    "LSTM is not available on the "
                    "Render deployment. Please select TF-IDF."

            }), 400


        # ====================================================
        # TEXT VALIDATION
        # ====================================================

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


        print()
        print("======================================")
        print("POST /predict")

        print(
            "Environment:",
            ENVIRONMENT_NAME
        )

        print(
            "Selected model:",
            selected_model
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        result = predict_text(
            text,
            selected_model
        )


        # ====================================================
        # SOURCE
        # ====================================================

        source_type = "Typed Text"

        source_name = (
            "Text entered by user"
        )


        result["source_type"] = (
            source_type
        )


        # ====================================================
        # PDF
        # ====================================================

        pdf_filename = (
            generate_result_pdf(
                result,
                source_name,
                source_type
            )
        )


        # ====================================================
        # PNG
        # ====================================================

        png_filename = (
            generate_result_png(
                result,
                source_name,
                source_type
            )
        )


        # ====================================================
        # DOWNLOAD URLS
        # ====================================================

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
            "======================================"
        )


        return jsonify(
            result
        ), 200


    except Exception as e:

        print()
        print("======================================")
        print("PREDICTION ERROR")
        print("======================================")


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
            "======================================"
        )


        return jsonify({

            "error":
                str(e)

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

        # ====================================================
        # FILE
        # ====================================================

        uploaded_file = (
            request.files.get(
                "file"
            )
        )


        # ====================================================
        # MODEL
        # ====================================================

        selected_model = (
            request.form.get(
                "model",
                "tfidf"
            )
        )


        selected_model = (
            str(selected_model)
            .strip()
            .lower()
        )


        # ====================================================
        # VALID MODEL
        # ====================================================

        if selected_model not in {
            "tfidf",
            "lstm"
        }:

            return jsonify({

                "error":
                    "Invalid model selected. "
                    "Choose TF-IDF or LSTM."

            }), 400


        # ====================================================
        # RENDER LSTM BLOCK
        # ====================================================

        if (
            selected_model == "lstm"
            and
            not ENABLE_LSTM
        ):

            return jsonify({

                "error":
                    "LSTM is not available on the "
                    "Render deployment. Please select TF-IDF."

            }), 400


        # ====================================================
        # FILE VALIDATION
        # ====================================================

        if not uploaded_file:

            return jsonify({

                "error":
                    "Please select an image or PDF."

            }), 400


        filename = (
            uploaded_file.filename or ""
        )


        # ====================================================
        # EXTENSION
        # ====================================================

        if not allowed_file(filename):

            return jsonify({

                "error":
                    "Unsupported file type. "
                    "Please upload PNG, JPG, JPEG, "
                    "WEBP, or PDF."

            }), 400


        # ====================================================
        # READ FILE
        # ====================================================

        file_bytes = (
            uploaded_file.read()
        )


        if not file_bytes:

            return jsonify({

                "error":
                    "The uploaded file is empty."

            }), 400


        print()
        print("======================================")
        print("POST /upload")

        print(
            "Environment:",
            ENVIRONMENT_NAME
        )

        print(
            "Filename:",
            filename
        )

        print(
            "Selected model:",
            selected_model
        )


        # ====================================================
        # EXTRACT TEXT
        # ====================================================

        (
            extracted_text,
            source_type,
            used_ocr
        ) = extract_text_from_file(
            filename,
            file_bytes
        )


        # ====================================================
        # NORMALIZE TEXT
        # ====================================================

        extracted_text = (
            normalize_extracted_text(
                extracted_text
            )
        )


        # ====================================================
        # VALIDATE EXTRACTED TEXT
        # ====================================================

        if not extracted_text:

            return jsonify({

                "error":
                    "No readable text could be extracted "
                    "from the file."

            }), 400


        if len(extracted_text.split()) < 5:

            return jsonify({

                "error":
                    "The file does not contain enough "
                    "readable text."

            }), 400


        # ====================================================
        # PREDICTION
        # ====================================================

        result = predict_text(
            extracted_text,
            selected_model
        )


        # ====================================================
        # SOURCE INFORMATION
        # ====================================================

        result["source_type"] = (
            source_type
        )


        result["used_ocr"] = (
            used_ocr
        )


        result["filename"] = (
            filename
        )


        # ====================================================
        # PDF
        # ====================================================

        pdf_filename = (
            generate_result_pdf(
                result,
                filename,
                source_type
            )
        )


        # ====================================================
        # PNG
        # ====================================================

        png_filename = (
            generate_result_png(
                result,
                filename,
                source_type
            )
        )


        # ====================================================
        # DOWNLOAD URLS
        # ====================================================

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
            "======================================"
        )


        return jsonify(
            result
        ), 200


    except Exception as e:

        print()
        print("======================================")
        print("UPLOAD ERROR")
        print("======================================")


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
            "======================================"
        )


        return jsonify({

            "error":
                str(e)

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
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "error":
            "File is too large. "
            "Maximum size is 16 MB."

    }), 413


# ============================================================
# GENERAL SERVER ERROR
# ============================================================

@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "error":
            "Internal server error. "
            "Check the server logs."

    }), 500

# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("======================================")
    print("       AI NEWS INTELLIGENCE")
    print("======================================")

    print(
        "Starting local development server..."
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print(
        "Environment:",
        ENVIRONMENT_NAME
    )

    print(
        "Render detected:",
        IS_RENDER
    )

    print(
        "LSTM enabled:",
        ENABLE_LSTM
    )

    if ENABLE_LSTM:

        print(
            "Models: TF-IDF + LSTM"
        )

    else:

        print(
            "Models: TF-IDF only"
        )

    print(
        "======================================"
    )

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )