import torch
import os
import json
import argparse
from model import IRClassifier
from predict import predict

def get_angle_index(angle_str):
    angle_map = {'eyes': 0, 'right': 1, 'front': 2, 'unknown': 3}
    return angle_map.get(angle_str, 3)  # Default to unknown if not found

def get_time_index(time_str):
    time_map = {'sober': 0, '20mins': 1, '40mins': 2, '60mins': 3}
    return time_map.get(time_str, 3)  # Default to unknown if not found

def batch_predict(model, images_dir, metadata_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = {}
    
    # If metadata file is provided, use it
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for item in metadata:
            image_path = os.path.join(images_dir, os.path.basename(item['image_path']))
            if os.path.exists(image_path):
                angle_idx = get_angle_index(item['angle'])
                time_idx = get_time_index(item['time'])
                prediction, probability = predict(model, image_path, angle_idx, time_idx)
                results[os.path.basename(item['image_path'])] = {
                    "prediction": "DRUNK" if prediction == 1 else "NOT DRUNK",
                    "probability": round(probability, 2)
                }
    else:
        # Process all images in the directory
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for filename in os.listdir(images_dir):
            if any(filename.lower().endswith(ext) for ext in valid_extensions):
                image_path = os.path.join(images_dir, filename)
                
                # Try to extract angle and time from filename
                angle_idx = 3  # Default: unknown
                time_idx = 3   # Default: unknown
                
                # Parse filename for clues - format example: 21_ilias_3_f_M_27_110.jpg
                # where 3 = time period (3 = 40mins), f = front angle
                parts = filename.split('_')
                if len(parts) >= 4:
                    # Check if angle marker exists
                    if parts[3] == 'e':
                        angle_idx = 0  # eyes
                    elif parts[3] == 'r':
                        angle_idx = 1  # right 
                    elif parts[3] == 'f':
                        angle_idx = 2  # front
                    
                    # Check time period from folder or filename
                    if '20mins' in images_dir:
                        time_idx = 1
                    elif '40mins' in images_dir:
                        time_idx = 2
                    elif '60mins' in images_dir:
                        time_idx = 3
                    elif 'sober' in images_dir:
                        time_idx = 0
                
                prediction, probability = predict(model, image_path, angle_idx, time_idx)
                results[filename] = {
                    "prediction": "DRUNK" if prediction == 1 else "NOT DRUNK",
                    "probability": round(probability, 2),
                    "angle": angle_idx,
                    "time": time_idx
                }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict using trained model")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory containing IR images")
    parser.add_argument("--metadata", type=str, help="Optional metadata JSON file with angle and time info")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file for predictions")
    args = parser.parse_args()
    
    # Load the model
    model = IRClassifier()
    model.load_state_dict(torch.load("model.pt"))
    
    # Make predictions
    results = batch_predict(model, args.images_dir, args.metadata)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Processed {len(results)} images. Results saved to {args.output}") 