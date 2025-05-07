# Smart Focus: A Deep Learning-Based Student Engagement Tracking System

***Developed by:J.K.LIWIN JOSE***
## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

## Overview

Smart Focus is an advanced classroom monitoring system that leverages deep learning to track student engagement and attendance in real-time. The system uses computer vision to identify students through facial recognition and analyze their behavior patterns to determine their level of attentiveness during class sessions.

## Features

- **Automated Attendance Tracking**: Identifies students and records their presence automatically
- **Behavior Analysis**: Classifies student behaviors into attentive and non-attentive categories
- **Real-time Monitoring**: Processes video feed to provide immediate insights
- **Engagement Metrics**: Calculates engagement ratios and attendance status
- **Data Export**: Generates CSV reports with detailed attendance and engagement statistics

## Technology Stack

- **Python**: Core programming language
- **OpenCV**: For camera interface and image processing
- **YOLOv8**: Custom-trained model for behavior detection
- **Face Recognition**: For student identification
- **PyTorch**: Deep learning framework supporting YOLOv8
- **Pandas**: For data manipulation and reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-focus.git
cd smart-focus
