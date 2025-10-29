from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms

# Import your model and threshold function
from tumor_model import CNN, threshold  # Replace with the actual file name of your model

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Load your pre-trained model
device = torch.device('cpu')  # Use 'cuda' if available
model = CNN()
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

# Define a constant threshold
threshold_value = 0.5  # Adjust based on your task and model characteristics

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] + '/' + filename
        file.save(filepath)

        # Preprocess the image for model input
        input_tensor = preprocess_image(filepath)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Apply the threshold to make a binary prediction
        prediction = 1 if output.item() >= threshold_value else 0

        # Render the prediction result
        return render_template('result.html', prediction=prediction)

    return redirect(request.url)

def preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor

if __name__ == '__main__':
    app.run(debug=True)
