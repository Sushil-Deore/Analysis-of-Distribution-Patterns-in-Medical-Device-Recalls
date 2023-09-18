import requests
import csv
import os

# Function to create an output folder if it doesn't exist
def create_output_folder(folder_name):
    os.makedirs(folder_name, exist_ok=True)

# Function to create a CSV file with specified field names as headers
def create_csv_file(file_name, field_names):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(field_names)

# Function to fetch data from an API and write it to a CSV file
def fetch_data_and_write_to_csv(url, query_parameters, file_name, field_names):
    response = requests.get(url, params=query_parameters)
    data = response.json()
    records = data.get("results")

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Writing the header row
        writer.writerow(field_names)  
        
        for record in records:
            # Extract and format data as needed, then write to the CSV file
            row = [record.get(field, "") for field in field_names]
            writer.writerow(row)

def main():
    # Define the API endpoint and query parameters
    base_url = "https://api.fda.gov/device/recall.json"

    query_parameters = {'search': "root_cause_description.exact:'Device Design'", 
                        'limit': 1000}
    
    # Desired dataset size
    total_records = 100000  

    # Create a folder for your output files
    output_folder = "output_files_raw"
    create_output_folder(output_folder)

    # Create an empty CSV file with the specified fields as headers
    field_names = ["event_date_posted",
                   "event_date_terminated",
                   "recall_status",
                   "recalling_firm",
                   "address_1",
                   "address_2",
                   "city",
                   "state",
                   "postal_code",
                   "country",
                   "reason_for_recall",
                   "product_code",
                   "root_cause_description",
                   "product_description",
                   "product_quantity",
                   "distribution_pattern",
                   "action",
                   "openfda.device_class"]

    # Initialize file count
    file_count = 0

    # Make API requests and append data to multiple files
    offset = 0
    while offset < total_records:
        file_name = os.path.join(output_folder, f"dataset_{file_count}.csv")
        create_csv_file(file_name, field_names)
        
        fetch_data_and_write_to_csv(base_url, query_parameters, file_name, field_names)
        
        offset += query_parameters['limit']
        file_count += 1

if __name__ == "__main__":
    main()
