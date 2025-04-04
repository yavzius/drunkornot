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
    
    # Apply contrast stretching to better simulate IR's higher contrast
    min_val = np.min(gray)
    max_val = np.max(gray)
    stretched = np.uint8(255 * ((gray - min_val) / (max_val - min_val)))
    
    # Apply a slight blur to simulate IR's lower resolution
    blurred = cv2.GaussianBlur(stretched, (3, 3), 0)
    
    # Add slight noise to simulate IR camera characteristics
    noise = np.random.normal(0, 2, blurred.shape).astype(np.uint8)
    with_noise = cv2.add(blurred, noise)
    
    # Blend original grayscale with processed version based on effect strength
    result = cv2.addWeighted(gray, 1 - effect_strength, with_noise, effect_strength, 0)
    
    # Convert back to BGR for display
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def preprocess_for_ir_model(frame):
    """Convert a regular webcam image to format similar to IR images"""
    # Convert to PIL Image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply transformations matching those in dataset.py - no normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),
        # Removed normalization to match dataset.py
    ])
    
    # Return tensor ready for model
    return transform(frame_pil).unsqueeze(0)

def detect_faces(frame):
    """Detect faces in the frame using OpenCV's face detector"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Make detection very strict with higher minNeighbors and larger minSize
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # If we found faces, filter out potential false positives
    if len(faces) > 0:
        # Sort faces by area (largest first)
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        
        # Keep only faces that meet size criteria relative to frame
        frame_height, frame_width = frame.shape[:2]
        min_face_area_ratio = 0.01  # Face should be at least 1% of frame area
        
        filtered_faces = []
        for (x, y, w, h) in faces:
            face_area_ratio = (w * h) / (frame_width * frame_height)
            if face_area_ratio >= min_face_area_ratio:
                filtered_faces.append((x, y, w, h))
        
        # If we found multiple faces, only keep the largest one or ones that are 
        # at least 70% the size of the largest face
        if len(filtered_faces) > 1:
            largest_area = filtered_faces[0][2] * filtered_faces[0][3]
            filtered_faces = [face for face in filtered_faces if (face[2] * face[3]) >= 0.7 * largest_area]
            
        return np.array(filtered_faces)
    
    return faces

def capture_and_predict():
    parser = argparse.ArgumentParser(description="Capture webcam image and predict drunk or not")
    parser.add_argument("--angle", type=int, default=2,
                        help="Camera angle (0=eyes, 1=right, 2=front, 3=unknown)")
    parser.add_argument("--time", type=int, default=0, 
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
    
    # For tracking face predictions
    face_predictions = []
    last_prediction_time = 0
    prediction_interval = 1.0  # seconds between predictions
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break
            
            # Apply IR effect for the full frame display
            ir_frame = apply_ir_effect(frame, effect_strength)
            
            # Display settings info
            info_text = f"Angle: {angle_labels[angle_value]} | Time: {time_labels[time_value]} | Effect: {effect_strength:.1f}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "SPACE: Predict | Q: Quit | E: Effect+/-", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "A: Angle | T: Time", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect faces in the original frame
            faces = detect_faces(frame)
            
            # Create display frames
            display_frame = frame.copy()
            display_ir_frame = ir_frame.copy()
            
            # Draw face boxes and predictions
            for i, (x, y, w, h) in enumerate(faces):
                # Draw rectangle around the face
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(display_ir_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # If we have predictions, display them
                if i < len(face_predictions):
                    prediction, probability = face_predictions[i]
                    result_text = f"DRUNK ({probability:.2f})" if prediction == 1 else f"SOBER ({probability:.2f})"
                    
                    # Display prediction text
                    cv2.putText(display_frame, result_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(display_ir_frame, result_text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display frames
            cv2.imshow("Original", display_frame)
            cv2.imshow("IR Simulation", display_ir_frame)
            
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
            current_time = time.time()
            should_predict = key == ord(' ') or (current_time - last_prediction_time > prediction_interval)
            
            if should_predict and len(faces) > 0:
                print("Processing faces for prediction...")
                face_predictions = []
                last_prediction_time = current_time
                
                # First apply IR effect to the whole frame
                ir_processed_frame = apply_ir_effect(frame, effect_strength)
                
                for (x, y, w, h) in faces:
                    # Extract face region from the IR processed frame
                    face_region = ir_processed_frame[y:y+h, x:x+w]
                    
                    # Ensure the face region is not empty
                    if face_region.size == 0:
                        print(f"Warning: Empty face region detected at ({x},{y},{w},{h})")
                        continue
                    
                    # Process face for model
                    try:
                        img_tensor = preprocess_for_ir_model(face_region)
                        img_tensor = img_tensor.to(device)
                        
                        # Convert angle and time to tensor
                        angle_tensor = torch.tensor([angle_value], dtype=torch.long).to(device)
                        time_tensor = torch.tensor([time_value], dtype=torch.long).to(device)
                        
                        # Make prediction
                        with torch.no_grad():
                            output = model(img_tensor, angle_tensor, time_tensor)
                            probability = output.item()
                            # Adjust threshold to be more conservative about drunk predictions
                            prediction = 1 if probability >= 0.65 else 0
                        
                        print(f"Face prediction: {'DRUNK' if prediction == 1 else 'SOBER'} ({probability:.2f})")
                        face_predictions.append((prediction, probability))
                    except Exception as e:
                        print(f"Error predicting face: {e}")
                        continue
                
                # Save image if requested and space was pressed
                if args.save and key == ord(' '):
                    timestamp = int(time.time())
                    orig_filename = f"capture_orig_{timestamp}.jpg"
                    ir_filename = f"capture_ir_{timestamp}.jpg"
                    cv2.imwrite(orig_filename, display_frame)
                    cv2.imwrite(ir_filename, display_ir_frame)
                    print(f"Images saved as {orig_filename} and {ir_filename}")
            
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