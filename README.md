# Drunk Detection System with Webcam Support

This project uses a trained model to detect intoxication based on infrared (IR) facial images. The system can work with both IR images and regular webcam images that are processed to mimic IR characteristics.

## Setup

1. Make sure you have the required packages installed:
   ```
   pip3 install torch torchvision opencv-python pillow numpy
   ```

2. The model has been trained on infrared face images at different angles (front, eye-level, right profile) and different time points after alcohol consumption (sober, 20 mins, 40 mins, 60 mins).

## Usage

### Predicting with IR Images

To run prediction on existing IR images:
```
python3 predict.py --image PATH_TO_IMAGE --angle ANGLE_VALUE --time TIME_VALUE
```

Where:
- `ANGLE_VALUE` is 0 for eyes, 1 for right profile, 2 for front, 3 for unknown
- `TIME_VALUE` is 0 for sober, 1 for 20mins, 2 for 40mins, 3 for 60mins

Example:
```
python3 predict.py --image drunk_sober_data/20mins/01_petros_2_e_M_20_71_033.jpg --angle 0 --time 1
```

### Webcam-based Prediction

To use a regular webcam for drunk detection:
```
python3 camera_predict.py [--angle ANGLE] [--time TIME] [--effect EFFECT_STRENGTH] [--save]
```

The script will:
1. Open your webcam
2. Apply transformations to simulate IR imaging
3. Process the image for prediction
4. Display the result (DRUNK or SOBER with probability)

#### Interactive Controls:
- **SPACE**: Capture and predict
- **A**: Cycle through camera angles
- **T**: Cycle through time values
- **E**: Increase IR effect strength
- **SHIFT+E**: Decrease IR effect strength
- **Q**: Quit

#### Command-line Options:
- `--angle`: Set initial camera angle (default: 2/front)
- `--time`: Set initial time value (default: 1/20mins)
- `--effect`: Set initial IR effect strength from 0.0-1.0 (default: 0.8)
- `--save`: Save captured images

## Model Details

The model was trained on a dataset of infrared facial images of subjects in both sober and intoxicated states. It uses a ResNet18-based architecture with additional embedding for camera angle and time information.

For best results with webcam images:
1. Ensure good lighting conditions
2. Position your face similarly to the training data angles
3. Adjust the IR effect strength for optimal results 