import tensorflow as tf

MODEL_PATH = "models/lstm_news_classifier.keras"
OUTPUT_PATH = "models/lstm_news_classifier.tflite"

print("Loading Keras model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("======================================")
print("LSTM TFLite conversion successful!")
print("Saved:", OUTPUT_PATH)
print("Size:", len(tflite_model) / 1024 / 1024, "MB")
print("======================================")