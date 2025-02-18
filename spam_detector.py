import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# ✅ Register the TFBertModel for proper loading
custom_objects = {"TFBertModel": TFBertModel}

# ✅ Load the trained model ONCE and keep it in memory
print("🚀 Loading model... (This may take a few seconds)")
model = tf.keras.models.load_model("bert_spam_model.keras", custom_objects=custom_objects, compile=False)
print("✅ Model loaded successfully!")

# ✅ Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def is_spam(text: str) -> bool:
	"""
	Checks if a given text message is spam.

	Args:
	- text (str): The input message.

	Returns:
	- bool: True if spam, False if not spam.
	"""
	# Tokenize input text (convert to BERT-compatible format)
	tokens = tokenizer(text, padding="max_length", truncation=True, return_tensors="tf")

	# ✅ Use `tf.function` to speed up inference
	@tf.function
	def predict_fn(input_ids, attention_mask):
		return model([input_ids, attention_mask], training=False)

	# Run prediction
	prediction = predict_fn(tokens["input_ids"], tokens["attention_mask"])

	# If prediction is closer to 1, it's spam; otherwise, it's ham
	return bool(prediction.numpy()[0][0] > 0.5)
