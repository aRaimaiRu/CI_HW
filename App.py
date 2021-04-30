import io
import os
import json
import random
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS,cross_origin
print(torch.__version__)


myPredictClass = ['buildings' ,'forest' ,'glacier' ,'mountain', 'sea', 'street']
Predictclass = ["Not Mosquito", "Mosquito"]
Randomspecies = ["Mosquito", "Anopheles"]

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

@app.route('/')
@cross_origin()
def hello():
    return 'Hello'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    
    file = request.files['file']
    img_bytes = file.read()
    # print(img_bytes)
    class_id  = get_prediction(image_bytes=img_bytes)
    if(class_id == "Mosquito"):
        resultRandom = random.choices(Randomspecies, weights=(30,70), k=1)
        # print(resultRandom[0])
        return jsonify({'Predict': resultRandom[0]})
    else:
        return jsonify({'Predict': class_id})
    # return jsonify({'Predict': class_id})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0',port=port)