# This code was run on Google colab since I wanted to spend my UoE GPU on fine-tuning only

# Load the csv file
from google.colab import files
import pandas as pd

def upload_csv():
    uploaded = files.upload()
    for file_name in uploaded.keys():
        print(f'Uploaded file: {file_name}') # Sanity check since there are 100 files I needed to work on
        return file_name

csv_file_name = upload_csv()







# Preprocess to get acceptable pairs of preverbalized - verbalized sentences
from transformers import T5Tokenizer
import pandas as pd

# Load the tokenizer for FLAN-T5-small
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Read the uploaded CSV file into a DataFrame
csv_file_name = "file_10_long.csv"  # replace with your actual file name
df = pd.read_csv(csv_file_name)

# Ensure the columns are indexed correctly (assuming columns are 0-indexed)
column_2 = df.columns[1]
column_3 = df.columns[2]

# Initialize a list to store the indices of rows to keep
rows_to_keep = []

# Iterate over each row and tokenize the text in column 2 and column 3
for index, row in df.iterrows():
    tokens_column_2 = tokenizer(row[column_2], return_tensors="pt")["input_ids"]
    tokens_column_3 = tokenizer(row[column_3], return_tensors="pt")["input_ids"]

    # Check if the number of tokens in both columns exceeds 512
    if tokens_column_2.size(1) + tokens_column_3.size(1) <= 512:
        rows_to_keep.append(index)

# Create a new DataFrame with only the rows to keep
df_filtered = df.loc[rows_to_keep]

# Output the filtered DataFrame to a new CSV file
filtered_csv_file_name = "no_long_token_10.csv"
df_filtered.to_csv(filtered_csv_file_name, index=False)

print(f"Filtered CSV file has been saved as {filtered_csv_file_name}")

# Download file:
files.download(filtered_csv_file_name)
