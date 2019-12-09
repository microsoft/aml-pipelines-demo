import urllib.request
import base64
import io
import requests 
import json
import argparse
from PIL import Image
from io import BytesIO
from azureml.core.webservice import Webservice
from azureml.core import Workspace

def imgToBase64(img):
    '''Convert pillow image to base64-encoded image'''
    imgio = BytesIO()
    img.save(imgio, 'JPEG')
    img_str = base64.b64encode(imgio.getvalue())
    return img_str.decode('utf-8')


# Define arguments
parser = argparse.ArgumentParser(description='Test script parser')
parser.add_argument('--image_url', type=str, help='URL of the image to score', default='https://compote.slate.com/images/222e0b84-f164-4fb1-90e7-d20bc27acd8c.jpg')
image_url = parser.parse_args().image_url

# get scoring url
aci_service_name = 'object-recognition-service'
workspace = Workspace.from_config()
aci_service = Webservice(workspace, name=aci_service_name)
scoring_url = aci_service.scoring_uri

# Download image and convert to base 64
with urllib.request.urlopen(image_url) as url:
    test_img = io.BytesIO(url.read())

base64Img = imgToBase64(Image.open(test_img))

# Get prediciton through endpoint
input_data = '{\"data\": \"'+ base64Img +'\"}'
headers = {'Content-Type':'application/json'}
response = requests.post(scoring_url, input_data, headers=headers)
print(json.loads(response.text))