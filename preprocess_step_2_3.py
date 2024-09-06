import pandas as pd

def preprocess_data(df):
    """
    Processes a DataFrame containing text normalization data, extracting sentences, normalizations, 
    and their corresponding class labels. Handles end-of-sentence markers ('<eos>') to split sentences 
    and appends them to lists for further processing.

    Args:
        df (pd.DataFrame): The input DataFrame with at least 3 columns where:
                          - Column 0 is the class label
                          - Column 1 is the word
                          - Column 2 is the normalized version of the word
                          - Column 3 (optional) is for special tokens

    Returns:
        tuple: A tuple containing three lists:
               - sentences (list of str): List of sentences formed by joining words.
               - normalizations (list of str): List of normalized sentences.
               - classes (list of str): List of class sequences corresponding to each sentence.
    """
    sentences = []
    normalizations = []
    classes = []

    sentence = []
    normalization = []
    class_list = []

    for idx, row in df.iterrows():
        if len(row) < 3:
            continue  # Skip rows with less than 3 columns

        class_label = str(row.iloc[0])  # Column 0 is for 'class'
        word = str(row.iloc[1])         # Column 1 is for 'word'
        norm = str(row.iloc[2])         # Column 2 is for 'normalization'
        special_token = str(row.iloc[3]) if len(row) > 3 else None  # Column 3 is for 'special token'

        if class_label == '<eos>':  # End of sentence
            if sentence:
                # Join sentence parts into single strings and append to lists
                sentences.append(" ".join(sentence))
                normalizations.append(" ".join(normalization))
                classes.append(" ".join(class_list))
            sentence = []
            normalization = []
            class_list = []
        else:
            sentence.append(word)
            normalization.append(norm)
            class_list.append(class_label)  # Append the class label

    # Handle the last sentence if it does not end with '<eos>'
    if sentence:
        sentences.append(" ".join(sentence))
        normalizations.append(" ".join(normalization))
        classes.append(" ".join(class_list))

    return sentences, normalizations, classes

def save_to_csv(sentences, normalizations, classes, file_name):
    """
    Saves the processed data (sentences, normalizations, and classes) into a CSV file 
    with columns 'classes', 'sentences', and 'normalizations'.

    Args:
        sentences (list of str): List of processed sentences.
        normalizations (list of str): List of normalized versions of the sentences.
        classes (list of str): List of class sequences corresponding to each sentence.
        file_name (str): The name of the output CSV file.

    Raises:
        ValueError: If the lengths of sentences, normalizations, and classes do not match.
    """
    if not (len(sentences) == len(normalizations) == len(classes)):
        raise ValueError("Mismatch in lengths of sentences, normalizations, and classes lists")

    data = {
        "classes": classes,
        "sentences": sentences,
        "normalizations": normalizations
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)

# File path
file_path = ""

# Read the dataset into a DataFrame with tab delimiter
try:
    df = pd.read_csv(file_path, delimiter='\t', on_bad_lines='skip', quoting=3)
    print("DataFrame loaded:")
    print(df.head())  # Print the first few rows to check if data is loaded correctly
except Exception as e:
    print(f"Error reading CSV file: {e}")
    df = pd.DataFrame()  # Define df as an empty DataFrame in case of error

# Proceed if DataFrame is successfully loaded
if not df.empty:
    # Preprocess the dataset
    sentences, normalizations, classes = preprocess_data(df)

    # Debugging output to verify content
    print("Sentences:", sentences[:10])  # Print first 10 entries for brevity
    print("Normalizations:", normalizations[:10])
    print("Classes:", classes[:10])

    # Save the processed data to a CSV file
    output_file = ""
    save_to_csv(sentences, normalizations, classes, output_file)
else:
    print("No data to process.")
