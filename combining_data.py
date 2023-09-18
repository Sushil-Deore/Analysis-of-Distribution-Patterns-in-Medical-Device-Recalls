import os
import csv

def read_csv_file(file_path):
    """
    Reads a CSV file and returns its content as a list of rows (excluding the header).
    """
    rows = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)

        # Read and skip the header row
        header = next(csv_reader) 

        for row in csv_reader:
            rows.append(row)
            
    return header, rows

def combine_csv_files(input_directory, output_file):
    """
    Combines multiple CSV files from a directory into a single CSV file.
    """
    all_rows = []
    header = None

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_directory, filename)
            file_header, file_rows = read_csv_file(file_path)
            
            if not header:
                header = file_header  # Set the header from the first file
            
            all_rows.extend(file_rows)  # Append rows from the current file

    # Write the combined data to the output CSV file
    with open(output_file, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        
        # Write the header row
        csv_writer.writerow(header)
        
        # Write all rows from the list
        csv_writer.writerows(all_rows)

    print(f"Combined data from {len(os.listdir(input_directory))} CSV files into {output_file}")

if __name__ == "__main__":
    csv_directory = "./output_files_raw/"
    output_file = "OpenFDA_combined_data.csv"
    combine_csv_files(csv_directory, output_file)