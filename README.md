# Seamless Focal Stack

## Description
This program aims to enhance the photographic experience by addressing a key challenge in traditional photography: the inability to refocus after a photo is taken. Traditional cameras require pre-capture focusing, but our program leverages focal-stack refocusing, a technique where multiple images at different focal lengths are captured and stored. Users can then choose the sharpest image for any given point. The project's primary goal is to eliminate radial image distortions and blurring caused by lens movement during focal-stack shooting, significantly improving the post-capture refocusing experience

## Installation
Ensure you have Python, OpenCV, NumPy, and tqdm installed. Clone the repository to your local machine.
```bash
pip install -r requirements.txt
```
## Setup
Before running the program, navigate to the Team04 directory using the command:`cd path/to/Team04`

## Usage
Run the script in the following ways:

- Default, just run: `python refocus.py`
- To show an all in focus image: `python refocus.py -i "input_folder" -all`
- To show the refocus window with aligned focal stack: `python refocus.py -i "input_folder" --aligned`
- To show the refocus window with the original focal stack: `python refocus.py -i "input_folder" --not-aligned`
- If the input directory (`-i "input_folder"`) is not specified, the default input folder 'test' will be used.
## Features
- Aligns images to correct shifts during shooting.
- Computes sharpness gradients using Laplacian convolution.
- Interactive focal point selection.

