import os
import requests
import argparse
from time_util import time_limit

# Define arguments
parser = argparse.ArgumentParser(description='Web scraping arg parser')
parser.add_argument('--output_dir', type=str, help='Directory to store output raw data')
parser.add_argument('--num_images', type=int, help='Number of images per class')
args = parser.parse_args()

# Get arguments from parser
output_dir = args.output_dir
num_images = args.num_images

# Set search headers and URL
headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'

# Define API endpoints
subscription_key = os.environ['COGNITIVE_SERVICES_API_KEY']
search_url = 'https://eastus.api.cognitive.microsoft.com/bing/v7.0/images/search'

# Define classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'ship']

# Make query for each class and download images
for name in classes:

    dir_name = os.path.join(output_dir, name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    counter = 0
    num_searches = int(num_images/150)+1

    for i in range(num_searches):
        
        response = requests.get(
            search_url, 
            headers = {
                'Ocp-Apim-Subscription-Key' : subscription_key
            }, 
            params = {
                'q': name, 
                'imageType': 'photo',
                'count': 150,
                'offset': i*150
            })
        response.raise_for_status()
        results = response.json()["value"]

        for image in results:
            if counter > num_images:
                break
            if image['encodingFormat'] == 'jpeg':
                print('Writing image {} for {}...'.format(counter, name))
                filename = '{}/{}.jpg'.format(dir_name, counter)
                try:
                    with time_limit(5):
                        with open(filename, 'wb') as file:
                            download = requests.get(image['contentUrl'], headers=headers)
                            file.write(download.content)
                        counter += 1
                except:
                    print('Skipping {} due to download error:'.format(filename))

