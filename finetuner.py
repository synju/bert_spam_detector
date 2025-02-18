import csv
import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertTokenizer

# Load tokenizer (must match the one used during training)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def load_new_data(file_path):
	"""
	Reads a new CSV file containing only new spam/ham data.
	Returns a TensorFlow dataset for training.
	"""
	tmp_dataset = []
	with open(file_path, mode='r', encoding='utf-8') as file:
		reader = csv.reader(file)
		next(reader)  # Skip header

		for row in reader:
			category = 1 if row[0].strip().lower() == "spam" else 0
			message = row[1].strip()
			tmp_dataset.append((message, category))

	if not tmp_dataset:
		print("⚠ No new data found in the CSV file. Exiting.")
		exit()

	texts, labels = zip(*tmp_dataset)
	labels = np.array(labels)

	dataset = tf.data.Dataset.from_tensor_slices((list(texts), labels))

	# Tokenization function
	def tokenize_map_function(text, label):
		text_str = text.numpy().decode("utf-8")  # Convert tensor to string
		tokens = tokenizer(text_str, padding="max_length", truncation=True, return_tensors="np")

		input_ids = tf.convert_to_tensor(tokens["input_ids"][0], dtype=tf.int32)
		attention_mask = tf.convert_to_tensor(tokens["attention_mask"][0], dtype=tf.int32)
		label = tf.convert_to_tensor(label, dtype=tf.int32)

		return input_ids, attention_mask, label

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

	return dataset.batch(2)


# 🔹 Load New Dataset (provided CSV file)
NEW_DATA_CSV = "new_data.csv"  # Replace with your actual new dataset file (excludes old training data)
dataset = load_new_data(NEW_DATA_CSV)

# 🔹 Load Existing Model for Fine-Tuning
try:
	model = keras.models.load_model("bert_spam_model.keras", compile=False)
	print("✅ Loaded existing model for fine-tuning.")
except:
	print("❌ No existing model found! Fine-tuning requires a pre-trained model.")
	exit()

# 🔹 Compile Model Again Before Training
optimizer = keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# 🔹 Start Fine-Tuning
start_time = time.time()

history = model.fit(
	dataset.map(lambda x, y, z: ((x, y), z)),  # Ensure correct input structure
	epochs=3,  # Fine-tune for a few epochs
	verbose=1
)

# 🔹 Show Fine-Tuning Time
end_time = time.time()
total_time = end_time - start_time
print(f"\n⏳ Fine-tuning completed in {total_time:.2f} seconds (~{total_time / 60:.2f} minutes)")

# 🔹 Save Fine-Tuned Model with Timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fine_tuned_model_name = f"finetuned_model_{timestamp}.keras"
model.save(fine_tuned_model_name
")
print(f"✅ Fine-tuned model saved as '{fine_tuned_model_name}'")
