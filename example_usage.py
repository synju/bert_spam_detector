from spam_detector import is_spam

# Example usage
if is_spam("You've won a free trip to Paris! Click here to claim your prize."):
	print("🚨 Spam detected!")
else:
	print("✅ This message is safe.")
