import csv
import os

# 🔹 File paths
OLD_DATA_CSV = "mail_data.csv"  # Old dataset
NEW_DATA_CSV = "new_data.csv"  # Newly collected dataset
UPDATED_DATA_CSV = "updated_mail_data.csv"  # Merged dataset


def load_csv_data(file_path):
	"""Loads data from a CSV file and returns it as a set of tuples."""
	data = set()
	if os.path.exists(file_path):  # Ensure file exists before reading
		with open(file_path, mode='r', encoding='utf-8') as file:
			reader = csv.reader(file)
			next(reader)  # Skip header
			for row in reader:
				if len(row) == 2:  # Ensure valid row format
					data.add((row[0].strip().lower(), row[1].strip()))  # Normalize spam/ham label
	return data


# 🔹 Load data from both CSVs
old_data = load_csv_data(OLD_DATA_CSV)
new_data = load_csv_data(NEW_DATA_CSV)

# 🔹 Merge datasets (set ensures no duplicates)
merged_data = old_data.union(new_data)

if not merged_data:
	print("⚠ No data found to merge! Exiting.")
	exit()

# 🔹 Save merged dataset
with open(UPDATED_DATA_CSV, mode='w', encoding='utf-8', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["Label", "Message"])  # Write header
	writer.writerows(sorted(merged_data))  # Sort data for consistency

print(f"✅ Merged dataset saved as '{UPDATED_DATA_CSV}' with {len(merged_data)} unique messages.")
