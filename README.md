# Iron Man CV2

A computer vision project using OpenCV and MediaPipe for hand tracking and gesture recognition with Iron Man-themed visual effects.

## Overview

This project implements real-time hand detection, tracking, and gesture recognition using MediaPipe and OpenCV. It features sci-fi themed visual overlays inspired by Iron Man's suit interface, providing an immersive augmented reality experience.

## Features

- **Dual Hand Detection**: Supports tracking of both left and right hands simultaneously
- **Hand Landmarks**: Extracts 21 hand landmarks for each detected hand
- **Gesture Recognition**: Identifies various hand gestures for interaction
- **Sci-Fi Visual Effects**: Tech-themed skeleton overlays and visual enhancements
- **Real-time Processing**: Optimized for 30 FPS video processing
- **Confidence Thresholds**: Configurable detection and tracking confidence levels

## Requirements

- Python 3.7+
- OpenCV (cv2)
- MediaPipe
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ironman_cv2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python mediapipe numpy
```

## Usage

Run the main script:
```bash
python main.py
```

The application will:
1. Access your webcam
2. Display real-time hand tracking with visual overlays
3. Recognize and respond to hand gestures

## Project Structure

- `main.py` - Main application entry point with HandTracker class and gesture recognition
- `requirements.txt` - Python package dependencies

## Configuration

Edit the configuration section in `main.py` to customize:
- `WIDTH, HEIGHT` - Video resolution (default: 1280x720)
- `FPS_TARGET` - Target frames per second (default: 30)
- Detection confidence thresholds
- Visual styling and colors

## How It Works

### Hand Detection
The project uses MediaPipe's pre-trained hand detection model to identify hands in the video stream and extract 21 landmark points representing the hand skeleton.

### Gesture Recognition
Hand landmarks are analyzed to recognize common gestures like:
- Fist
- Open hand
- Pointing
- Thumbs up/down
- And more custom gestures

### Visual Rendering
Detected hands are rendered with:
- Skeletal overlay showing joint connections
- Tech-inspired color schemes
- Real-time frame rate display

## Controls

- Press `ESC` or `q` to exit the application

## Performance

- Optimized for real-time processing at 30 FPS
- Supports dual-hand tracking without significant performance degradation
- Configurable confidence thresholds for accuracy vs. responsiveness trade-off

## Future Enhancements

- Additional gesture recognition patterns
- Hand pose estimation
- Gesture-based UI controls
- Multi-hand interaction logic
- Recording and playback capabilities

## License

[Specify your license here]

## Author

[Your name/organization]

## Acknowledgments

- MediaPipe for hand detection models
- OpenCV for computer vision utilities
