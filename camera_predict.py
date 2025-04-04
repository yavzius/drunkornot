import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import IRClassifier
import argparse
import time

def apply_ir_effect(frame, effect_strength=0.8):
    """Apply various transformations to make the image look more like IR"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(gray)
    
    # Apply slight Gaussian blur to mimic IR's lower resolution
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Blend original grayscale with processed version based on effect strength
    result = cv2.addWeighted(gray, 1 - effect_strength, blurred, effect_strength, 0)
    
    # Convert back to BGR for display
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def preprocess_for_ir_model(frame):
    """Convert a regular webcam image to format similar to IR images"""
    # Convert to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply transformations similar to those in predict.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Adjusted for grayscale
    ])
    
    # Return tensor ready for model
    return transform(frame_pil).unsqueeze(0)

def capture_and_predict():
    parser = argparse.ArgumentParser(description="Capture webcam image and predict drunk or not")
    parser.add_argument("--angle", type=int, default=2,
                        help="Camera angle (0=eyes, 1=right, 2=front, 3=unknown)")
    parser.add_argument("--time", type=int, default=1, 
                        help="Time value (0=sober, 1=20mins, 2=40mins, 3=60mins)")
    parser.add_argument("--save", action="store_true", help="Save the captured image")
    parser.add_argument("--effect", type=float, default=0.8, 
                        help="IR effect strength (0.0-1.0)")
    args = parser.parse_args()
    
    # Load the model
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IRClassifier()
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded, using device: {device}")
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set better resolution if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Create windows
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("IR Simulation", cv2.WINDOW_NORMAL)
    
    # Settings
    effect_strength = args.effect
    angle_value = args.angle
    time_value = args.time
    
    # Define angle and time labels
    angle_labels = {0: "Eyes", 1: "Right", 2: "Front", 3: "Unknown"}
    time_labels = {0: "Sober", 1: "20mins", 2: "40mins", 3: "60mins"}
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Apply IR effect for display
            ir_frame = apply_ir_effect(frame, effect_strength)
            
            # Display settings info
            info_text = f"Angle: {angle_labels[angle_value]} | Time: {time_labels[time_value]} | Effect: {effect_strength:.1f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Quit | E: Effect+/-", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "A: Angle | T: Time", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frames
            cv2.imshow("Original", frame)
            cv2.imshow("IR Simulation", ir_frame)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Change settings
            if key == ord('e'):
                # Increase effect strength
                effect_strength = min(1.0, effect_strength + 0.1)
            elif key == ord('E'):
                # Decrease effect strength
                effect_strength = max(0.0, effect_strength - 0.1)
            elif key == ord('a'):
                # Cycle angle
                angle_value = (angle_value + 1) % 4
            elif key == ord('t'):
                # Cycle time
                time_value = (time_value + 1) % 4
            
            # Process image and predict
            elif key == ord(' '):
                # Process image for model
                print("Processing image...")
                img_tensor = preprocess_for_ir_model(ir_frame)  # Use the IR processed image
                img_tensor = img_tensor.to(device)
                
                # Convert angle and time to tensor
                angle_tensor = torch.tensor([angle_value], dtype=torch.long).to(device)
                time_tensor = torch.tensor([time_value], dtype=torch.long).to(device)
                
                # Make prediction
                print("Predicting...")
                with torch.no_grad():
                    output = model(img_tensor, angle_tensor, time_tensor)
                    probability = output.item()
                    prediction = 1 if probability >= 0.5 else 0
                
                # Display result
                result_text = f"DRUNK (p={probability:.2f})" if prediction == 1 else f"SOBER (p={probability:.2f})"
                print(f"Prediction: {result_text}")
                
                # Show result on image
                result_frame = ir_frame.copy()
                cv2.putText(result_frame, result_text, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("IR Simulation", result_frame)
                
                # Save image if requested
                if args.save:
                    timestamp = int(time.time())
                    orig_filename = f"capture_orig_{timestamp}.jpg"
                    ir_filename = f"capture_ir_{timestamp}.jpg"
                    cv2.imwrite(orig_filename, frame)
                    cv2.imwrite(ir_filename, ir_frame)
                    print(f"Images saved as {orig_filename} and {ir_filename}")
                
                # Wait a moment to show result
                cv2.waitKey(2000)
            
            # Quit
            elif key == ord('q'):
                break
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")

if __name__ == "__main__":
    capture_and_predict() 