# 📨 BERT Spam Detector

This project uses **BERT** (Bidirectional Encoder Representations from Transformers) to detect spam messages. 
It includes scripts for training, fine-tuning, dataset management, and real-time spam detection.

## 📂 Project Structure

```
📂 spam_detector_project
 ├── trainer.py              # Train a spam classifier using BERT
 ├── spam_detector.py        # Load the trained model and classify messages as spam or not
 ├── example_usage.py        # Example script to test spam classification
 ├── dataset_updater.py      # Merge old and new datasets to maintain a complete training dataset
 ├── finetuner.py            # Fine-tune the existing model with new data
 ├── bert_spam_model.keras   # Trained BERT model for spam detection (Generated with trainer.py)
 ├── mail_data.csv           # Dataset containing spam/ham messages
 ├── new_data.csv            # New dataset used for fine-tuning
 ├── updated_mail_data.csv   # Merged dataset after running dataset_updater.py
 ├── README.md               # Documentation
```

## 🚀 Usage Instructions

### **1️⃣ Training a New Model**
To train a **new** spam detection model using `mail_data.csv`:
```bash
python trainer.py
```
- Loads `mail_data.csv`
- Trains a new BERT model
- Saves it as `bert_spam_model.keras`

### **2️⃣ Running Spam Detection**
To check if a message is spam, use:
```bash
python example_usage.py
```
#### **Example (`example_usage.py`):**
```python
from spam_detector import is_spam

if is_spam("You've won a free trip to Paris! Click here to claim your prize."):
    print("🚨 Spam detected!")
else:
    print("✅ This message is safe.")
```

### **3️⃣ Fine-Tuning the Model with New Data**
To improve the model using **new data**:
1. Place new training data in `new_data.csv`
2. Run:
   ```bash
   python finetuner.py
   ```
3. A **fine-tuned model** will be saved as `finetuned_model_YYYYMMDD_HHMMSS.keras`

### **4️⃣ Merging Old & New Data**
To create a **combined dataset** from old and new data:
```bash
python dataset_updater.py
```
- Merges `mail_data.csv` and `new_data.csv`
- Saves the combined dataset as `updated_mail_data.csv`

## 📝 Script Breakdown

### **`trainer.py`** - Train the Model
- Reads `mail_data.csv`
- Tokenizes messages using BERT
- Trains a spam classifier
- Saves the trained model as `bert_spam_model.keras`

### **`spam_detector.py`** - Detect Spam
- Loads the trained model (`bert_spam_model.keras`)
- Uses BERT to classify new messages
- Function: `is_spam(text: str) -> bool`

### **`example_usage.py`** - Test Spam Classification
- Imports `is_spam()`
- Passes a sample message
- Prints whether it's spam or not

### **`finetuner.py`** - Fine-Tune the Model
- Loads `bert_spam_model.keras`
- Trains the model with `new_data.csv`
- Saves a **fine-tuned model** with a timestamp

### **`dataset_updater.py`** - Merge Datasets
- Reads `mail_data.csv` and `new_data.csv`
- Removes duplicate messages
- Saves `updated_mail_data.csv`

## 📊 Dataset Format

The CSV files should have **two columns**:
```
Label, Message
spam, "Win a free iPhone now!"
ham, "Hello, how are you?"
```
- **`mail_data.csv`** → Original dataset
- **`new_data.csv`** → New spam/ham data
- **`updated_mail_data.csv`** → Merged dataset

## 🔧 Requirements

📌 Install dependencies using:
```bash
pip install tensorflow transformers tqdm numpy
```

## 🏆 Future Improvements
- Optimize inference speed by running on GPU
- Collect more real-world spam messages
- Convert this into a web API

## 🔧 Supplementary Scripts
- **`cuda_test.py`** → Tests if Cuda GPU utilization is enabled on the system or not.

## 🤖 Author
Built with ❤️ using **TensorFlow & BERT**
https://github.com/synju
