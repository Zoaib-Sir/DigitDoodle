from flask import Flask, request, jsonify, render_template, session
import base64
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )
        
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

# Load model
model = EnhancedNet()
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

def enhanced_preprocess(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    
    # Threshold and invert
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Remove noise
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find and center main contour
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(cnt)
        roi = processed[y:y+h, x:x+w]
        
        # Maintain aspect ratio
        if h > w:
            new_h = 20
            new_w = int(w * (20 / h))
        else:
            new_w = 20
            new_h = int(h * (20 / w))
        
        resized = cv2.resize(roi, (new_w, new_h))
        
        # Center in 28x28 canvas
        canvas = np.zeros((28, 28), dtype=np.uint8)
        dx = (28 - new_w) // 2
        dy = (28 - new_h) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized
    else:
        canvas = processed
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(Image.fromarray(canvas)).unsqueeze(0)

def predict_digit(image_bytes):
    tensor = enhanced_preprocess(image_bytes)
    with torch.no_grad():
        output = model(tensor)
    return output.argmax().item()

def generate_problem():
    operators = ['+', '-', '*']
    op = random.choice(operators)
    
    if op == '+':
        a = random.randint(0, 99)
        b = random.randint(0, 99 - a)
        answer = a + b
    elif op == '-':
        a = random.randint(0, 99)
        b = random.randint(0, a)
        answer = a - b
    else:
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        answer = a * b
    
    return f"{a} {op} {b}", answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/new_problem')
def new_problem():
    problem, answer = generate_problem()
    session['correct_answer'] = answer
    return jsonify({
        'problem': problem,
        'answer': answer
    })

@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()
        correct_answer = session.get('correct_answer', 'Unknown')
        
        # Process images directly from base64
        drawing1_bytes = base64.b64decode(data['drawing1'].split(',')[1])
        drawing2_bytes = base64.b64decode(data['drawing2'].split(',')[1])
        
        # Make predictions
        prediction1 = predict_digit(drawing1_bytes)
        prediction2 = predict_digit(drawing2_bytes)
        user_answer = int(f"{prediction1}{prediction2}")
        
        # Verify answer
        correct = user_answer == correct_answer
        
        return jsonify({
            'correct': correct,
            'correctAnswer': correct_answer,
            'userAnswer': user_answer
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'correctAnswer': session.get('correct_answer', 'Unknown'),
            'userAnswer': "?",
            'correct': False
        }), 500

if __name__ == '__main__':
    app.run()