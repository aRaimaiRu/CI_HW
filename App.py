import io
import os
import json
import random
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
from flask import Flask, jsonify, request ,url_for,render_template, redirect
from flask_cors import CORS,cross_origin
import copy
print(torch.__version__)


myPredictClass = ['buildings' ,'forest' ,'glacier' ,'mountain', 'sea', 'street']


app = Flask(__name__)
# imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
device = 'cpu'

model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)#6
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load("resnet50.pth",map_location=torch.device('cpu')).state_dict())
model_ft.eval()


def transform_image(image_bytes):
    my_transforms =  T.Compose([T.RandomResizedCrop(size = 150),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(), 
                              T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    # print(image_bytes)
    # print(image)
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model_ft.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    print("predicted_idx",predicted_idx);
    return myPredictClass[int(predicted_idx)]

@app.route('/hello')
@cross_origin()
def hello():
    return 'Hello'

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Check if no file was submitted to the HTML form
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            filename = file.filename
            
            file = request.files['file']
            img_bytes = file.read()
            output = get_prediction(image_bytes=img_bytes)
            image = Image.open(io.BytesIO(img_bytes))
            image.save(os.path.join('static', filename))
            path_to_image = url_for('static', filename = filename)
            result = {
                'output': output,
                'path_to_image': path_to_image,
                'size':150
            }
            return render_template('show.html', result=result,bg=url_for('static', filename = "BG.png" ))
    return render_template('index.html',bg=url_for('static', filename = "BG.png" ))


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    file = request.files['file']
    img_bytes = file.read()
    # print(img_bytes)
    class_id  = get_prediction(image_bytes=img_bytes)
    return jsonify({'Predict': class_id})



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0',port=port)