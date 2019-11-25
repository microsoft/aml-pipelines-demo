import os
import argparse
        
# Define arguments
parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--raw_data_dir', type=str, help='Directory where raw data is stored')
args = parser.parse_args()

# Get arguments from parser
raw_data_dir = args.raw_data_dir

print(os.listdir(raw_data_dir))
    