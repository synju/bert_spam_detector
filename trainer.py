import csv
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertTokenizer
from tqdm import tqdm

# Load a pre-trained BERT tokenizer (this converts text into numerical tokens)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 🔹 Function to Read CSV and Convert It Into a TensorFlow Dataset
def convert_csv_to_dataset(file_path):
	"""
	Reads a CSV file containing spam/ham messages and converts it into a TensorFlow dataset.

	CSV format:
	spam, "Win a free iPhone now!"
	ham, "Hello, how are you?"

	Returns:
	- A tf.data.Dataset with (text, label) pairs
	"""
	tmp_dataset = []
	with open(file_path, mode='r', encoding='utf-8') as file:
		reader = csv.reader(file)
		next(reader)  # Skip header

		for row in reader:
			category = 1 if row[0].strip().lower() == "spam" else 0
			message = row[1].strip()
			tmp_dataset.append((message, category))

	# Unpack data into separate lists
	texts, labels = zip(*tmp_dataset)
	labels = np.array(labels)

	return tf.data.Dataset.from_tensor_slices((list(texts), labels))


# 🔹 Function to Tokenize Each Message Using BERT Tokenizer
def tokenize_map_function(text, label):
	"""
	Converts raw text into BERT tokenized format.

	Steps:
	1. Convert TensorFlow tensor to a Python string (needed for tokenizer)
	2. Tokenize the text using BERT's tokenizer (with padding & truncation)
	3. Convert the tokenized output into TensorFlow tensors

	Returns:
	- input_ids: Tokenized text (integer array)
	- attention_mask: Array indicating which parts of the text are actual words vs padding
	- label: The spam/ham label
	"""
	text_str = text.numpy().decode("utf-8")  # Convert tensor to string
	tokens = tokenizer(text_str, padding="max_length", truncation=True, return_tensors="np")

	# Convert to TensorFlow tensors
	input_ids = tf.convert_to_tensor(tokens["input_ids"][0], dtype=tf.int32)
	attention_mask = tf.convert_to_tensor(tokens["attention_mask"][0], dtype=tf.int32)
	label = tf.convert_to_tensor(label, dtype=tf.int32)

	return input_ids, attention_mask, label


# 🔹 Function to Apply Tokenization to the Entire Dataset
def preprocess_dataset(dataset):
	"""
	Converts raw text dataset into tokenized format compatible with BERT.

	Steps:
	1. Use `tf.py_function` to apply `tokenize_map_function` to each text message
	2. Ensure the returned tensors have correct shapes (512 tokens max)
	3. Batch the dataset for training (2 samples per batch)

	Returns:
	- A TensorFlow dataset ready for training
	"""
	dataset = dataset.map(
		lambda text, label: tf.py_function(
			tokenize_map_function, [text, label], [tf.int32, tf.int32, tf.int32]
		),
		num_parallel_calls=tf.data.AUTOTUNE,
	)

	dataset = dataset.map(
		lambda input_ids, attention_mask, label: (
			tf.ensure_shape(input_ids, [512]),
			tf.ensure_shape(attention_mask, [512]),
			tf.ensure_shape(label, []),
		)
	)

	return dataset.batch(2)  # Group data into batches of 2 samples


# 🔹 Load and Process the Dataset
dataset = convert_csv_to_dataset("./mail_data.csv")
dataset = preprocess_dataset(dataset)

# ✅ Display Training Progress Without Breaking Dataset Structure
for _ in tqdm(dataset, desc="Training Progress", unit="batch"):
	pass  # Just a placeholder to display progress

# 🔹 Load a Pre-Trained BERT Model
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# 🔹 Build a Spam Classification Model Using BERT
"""
Model Architecture:
1. Takes in input IDs and attention masks
2. Passes them through BERT (outputs a feature vector)
3. Uses a dense (fully connected) layer with ReLU activation
4. Final output layer predicts spam (1) or ham (0) using sigmoid activation
"""
input_ids = keras.layers.Input(shape=(512,), dtype=tf.int32, name="input_ids")
attention_mask = keras.layers.Input(shape=(512,), dtype=tf.int32, name="attention_mask")

bert_output = bert_model(input_ids, attention_mask=attention_mask)["pooler_output"]
dense = keras.layers.Dense(16, activation="relu")(bert_output)
output = keras.layers.Dense(1, activation="sigmoid")(dense)

# 🔹 Compile the Model
model = keras.Model(inputs=[input_ids, attention_mask], outputs=output)
optimizer = keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Track start time
start_time = time.time()

# 🔹 Train the Model
"""
1. Restructures the dataset to match model's input format
2. Uses `map()` to pair (input_ids, attention_mask) with labels
3. Runs for 5 training cycles (epochs)
"""
history = model.fit(
	dataset.map(lambda x, y, z: ((x, y), z)),  # Ensure correct input structure
	epochs=5,
	verbose=1
)

# 🔹 Show Total Training Time
end_time = time.time()
total_time = end_time - start_time
print(f"\n⏳ Training completed in {total_time:.2f} seconds (~{total_time / 60:.2f} minutes)")

# 🔹 Save the Trained Model
model.save("bert_spam_model.keras")
print("✅ BERT spam model trained and saved as 'bert_spam_model.keras'")
