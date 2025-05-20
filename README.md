# Smart Focus: A Deep Learning-Based Student Engagement Tracking System

***Developed by:J.K.LIWIN JOSE***


A comprehensive classroom monitoring system that combines facial recognition and behavior analysis to automate attendance tracking and measure student engagement in real-time.

## Features

- **Automated Attendance Tracking**: Recognizes students' faces and records their presence in CSV format
- **Behavior Analysis**: Detects and classifies student behaviors as attentive or non-attentive
- **Engagement Metrics**: Calculates attendance ratios based on behavioral patterns
- **Real-time Monitoring**: Provides visual feedback with bounding boxes and labels
- **Attendance Reports**: Generates detailed attendance summaries with engagement statistics



## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- NumPy
- Pandas
- Ultralytics (YOLOv11)
- face_recognition

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/classroom-monitoring.git
   cd classroom-monitoring
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained YOLOv8 model for behavior detection:
   - Ensure `trainedmodel4.pt` is placed in the project root directory

## Directory Structure

```
classroom-monitoring/
├── main.py                 # Main script
├── trainedmodel4.pt        # Pre-trained YOLO model
├── Training_images/        # Directory containing face training images
│   ├── student1.jpg
│   ├── student2.jpg
│   └── ...
├── Attendance.csv          # Generated attendance records
├── attendance_sum.csv      # Generated attendance summary
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Usage

1. Prepare your training images:
   - Add clear face images of each student to the `Training_images/` directory
   - Name each image file with the student's name (e.g., `liwin.jpg`)

2. Configure camera settings (if necessary):
   - Modify the `CAMERA_INDEX` value in the script to match your camera setup

3. Run the monitoring system:
   ```bash
   python main.py
   ```

4. View the output:
   - Real-time monitoring will be displayed on screen
   - Press 'q' to exit the application
   - Attendance data will be saved to `Attendance.csv`
   - Attendance summary with engagement metrics will be saved to `attendance_sum.csv`

## Behavior Classification

The system classifies student behaviors into two categories:

### Attentive Behaviors
- Upright posture
- Raising head
- Hand raising
- Reading
- Writing

### Non-Attentive Behaviors
- Using phone
- Bending over
- Bowing head
- Sleeping
- Turning head

## Customization

- **Behavior Categories**: Modify the `attentive_behaviors` and `non_attentive_behaviors` lists in the code
- **Attendance Threshold**: Adjust the attendance ratio threshold (currently 50%) to change attendance criteria
- **Camera Resolution**: Modify the camera resolution settings for different hardware setups

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [face_recognition](https://github.com/ageitgey/face_recognition) library
- OpenCV community

---

*Note: This project is intended for educational purposes only. Please ensure you have appropriate consent before implementing facial recognition systems in real classroom environments.*
