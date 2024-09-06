import csv

def process_csv(input_csv, output_csv):
    """
    Processes each csv file: <self> and "sil" are turned into the preverbalized word and punctuation, respectively.

    Args:
        input_csv (str): Path to the input CSV file.
        output_csv (str): Path to the output CSV file.
    """

    with open(input_csv, 'r', encoding='utf-8', newline='') as infile, \
            open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, delimiter='\t')

        for line in infile:
            line = line.strip().split('\t')

            # Handle lines with 2 columns
            if len(line) == 2:
                writer.writerow(line)
                continue

            # Handle lines starting with <eos>
            if line[0] == '<eos>':
                writer.writerow(line)
                continue

            # Handle "PUNCT" and "sil" cases
            if line[0] == "PUNCT" and line[1] == "sil":
                writer.writerow(line)
                continue

            # Replace <self> and "sil" in the third column
            line[2] = line[2].replace('<self>', line[1]).replace('sil', line[1])
            writer.writerow(line)

if __name__ == "__main__":
    input_csv = ""
    output_csv = ""
    process_csv(input_csv, output_csv)
