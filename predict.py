import torch
import argparse
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from model import IRClassifier

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for grayscale
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_path, angle, time_value):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load and preprocess the image
    img = load_image(image_path).to(device)
    
    # Prepare other inputs - convert to Long tensor for embedding layers
    angle_tensor = torch.tensor([angle], dtype=torch.long).to(device)
    time_tensor = torch.tensor([time_value], dtype=torch.long).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img, angle_tensor, time_tensor)
        probability = output.item()
        prediction = 1 if probability >= 0.5 else 0
        
    return prediction, probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict drunk or not using trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to IR image")
    parser.add_argument("--angle", type=int, required=True, help="Camera angle (0=eyes, 1=right, 2=front, 3=unknown)")
    parser.add_argument("--time", type=int, required=True, help="Time value (0=sober, 1=20mins, 2=40mins, 3=60mins)")
    args = parser.parse_args()
    
    # Load the model
    model = IRClassifier()
    model.load_state_dict(torch.load("model.pt"))
    
    # Make prediction
    prediction, probability = predict(model, args.image, args.angle, args.time)
    
    if prediction == 1:
        print(f"Prediction: DRUNK (probability: {probability:.2f})")
    else:
        print(f"Prediction: NOT DRUNK (probability: {probability:.2f})") 