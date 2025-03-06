import csv
import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertTokenizer
from tqdm import tqdm
from numba import cuda
from tensorflow.keras import mixed_precision

# ✅ Enable mixed precision training (Reduces GPU memory usage)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print("✅ Mixed Precision Enabled!")

# ✅ Enable GPU memory growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print("❌ GPU Memory Growth Error:", e)
else:
    print("❌ No GPU detected by TensorFlow!")

# ✅ Clear unused GPU memory before training
gc.collect()
cuda.close()
tf.keras.backend.clear_session()
print("✅ GPU memory cleared!")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Function to Read CSV and Convert It Into a TensorFlow Dataset
def convert_csv_to_dataset(file_path):
    tmp_dataset = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            category = 1 if row[0].strip().lower() == "spam" else 0
            message = row[1].strip()
            tmp_dataset.append((message, category))
    texts, labels = zip(*tmp_dataset)
    labels = np.array(labels, dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices((list(texts), labels))

# ✅ Function to Tokenize Text Using BERT Tokenizer
def tokenize_map_function(text, label):
    text_str = text.numpy().decode("utf-8")
    tokens = tokenizer(text_str, padding="max_length", truncation=True, max_length=256, return_tensors="np")
    input_ids = tf.convert_to_tensor(tokens["input_ids"][0], dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(tokens["attention_mask"][0], dtype=tf.int32)
    label = tf.convert_to_tensor(tf.cast(label, tf.int32))
    return input_ids, attention_mask, label

# ✅ Function to Apply Tokenization to Dataset
def preprocess_dataset(dataset):
    dataset = dataset.map(
        lambda text, label: tf.py_function(
            tokenize_map_function, [text, label], [tf.int32, tf.int32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda input_ids, attention_mask, label: (
            tf.ensure_shape(input_ids, [256]),
            tf.ensure_shape(attention_mask, [256]),
            tf.ensure_shape(label, []),
        )
    )
    return dataset.batch(1)  # ✅ Reduced batch size from 2 to 1

# ✅ Load and Process the Dataset
dataset = convert_csv_to_dataset("./mail_data.csv")
dataset = preprocess_dataset(dataset)

# ✅ Display Training Progress
for _ in tqdm(dataset, desc="Training Progress", unit="batch"):
    pass  # Just a placeholder to display progress

# ✅ Load Pre-Trained BERT Model
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# ✅ Build Spam Classification Model Using BERT
input_ids = keras.layers.Input(shape=(256,), dtype=tf.int32, name="input_ids")
attention_mask = keras.layers.Input(shape=(256,), dtype=tf.int32, name="attention_mask")

bert_output = bert_model(input_ids, attention_mask=attention_mask)["pooler_output"]
dense = keras.layers.Dense(16, activation="relu", dtype="float32")(bert_output)  # Force dtype to float32
output = keras.layers.Dense(1, activation="sigmoid", dtype="float32")(dense)

# ✅ Compile the Model
model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)
optimizer = keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Track start time
start_time = time.time()

# ✅ Train the Model
history = model.fit(
    dataset.map(lambda x, y, z: ((x, y), z)),
    epochs=5,
    verbose=1
)

# ✅ Show Total Training Time
end_time = time.time()
total_time = end_time - start_time
print(f"\n⏳ Training completed in {total_time:.2f} seconds (~{total_time / 60:.2f} minutes)")

# ✅ Save the Trained Model
model.save("bert_spam_model.keras")
print("✅ BERT spam model trained and saved as 'bert_spam_model.keras'")
