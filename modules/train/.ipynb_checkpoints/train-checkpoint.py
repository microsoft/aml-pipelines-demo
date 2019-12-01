 # Define arguments
parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--train_dir', type=str, help='Directory where training data is stored')
parser.add_argument('--valid_dir', type=str, help='Directory where validation data is stored')
parser.add_argument('--output_dir', type=str, help='Directory to output the model to')
args = parser.parse_args()

# Get arguments from parser
train_dir = args.train_dir
valid_dir = args.valid_dir
output_dir = args.output_dir
