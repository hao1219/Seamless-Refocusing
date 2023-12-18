# Seamless Focal Stack

## Description
This program aims to enhance the photographic experience by addressing a key challenge in traditional photography: the inability to refocus after a photo is taken. Traditional cameras require pre-capture focusing, but our program leverages focal-stack refocusing, a technique where multiple images at different focal lengths are captured and stored. Users can then choose the sharpest image for any given point. The project's primary goal is to eliminate radial image distortions and blurring caused by lens movement during focal-stack shooting, significantly improving the post-capture refocusing experience

## Installation
Ensure you have Python, OpenCV, NumPy, and tqdm installed. Clone the repository to your local machine.
```bash
pip install -r requirements.txt
```
## Setup
Before running the program, navigate to the Team04 directory using the command:`cd path/to/seamless_refocus`

## Usage
- Basic usage,show refocus window with default test aligned focal stack: `python refocus.py`
- To show an all in focus image: `python refocus.py -i "input_folder" -all`
- To show the refocus window with aligned focal stack: `python refocus.py -i "input_folder" --aligned`
- To show the refocus window with the original focal stack: `python refocus.py -i "input_folder" --not-aligned`
- To use with custom parameters: `python main.py -i <input_folder> -dx <dx_factor> -dy <dy_factor> -bt <black_threshold>`
- If the input directory (`-i "input_folder"`) is not specified, the default input folder in focal_stacks 'test' will be used.

## Parameters
- `-i`, `--input`: Input directory (default 'test')
- `-dx`, `--dx_factor`: dx factor for sharpness calculation (default 200)
- `-dy`, `--dy_factor`: dy factor for sharpness calculation (default 200)
- `-bt`, `--black_threshold`: Threshold for black border removal (default 30)
- `-all`, `--all_focus`: Call all_focus function to generate an all-focus image
- `--aligned`: Use aligned images (default)
- `--not-aligned`: Use non-aligned images

## Features
- Aligns images to correct shifts during shooting.
- Computes sharpness gradients using Laplacian convolution.
- Interactive focal point selection.
## Demo
Here's a quick look at the Seamless Focal Stack in action:
![Seamless refocusing]([URL_TO_YOUR_GIF](https://github.com/hao1219/Seamless-Refocusing/blob/main/demo/aligned.gif))


## License
This project and algorithm was inspired and adapted from the following sources:

- [momonala's focal stacking](https://github.com/momonala/focus-stack/tree/master)
- [Image Alignment (Feature Based) using OpenCV (C++/Python)](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
- [Image Alignment (Feature Based) using OpenCV (C++/Python)](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
- [Image Alignment (ECC) in OpenCV (C++/Python)](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
  
Images provided in seamless_refocus/focal_stacks:
- [leaves](https://github.com/hosseinjavidnia/Depth-Focal-Stack/tree/master/Data)
- [pcb and depthmap](https://github.com/PetteriAimonen/focus-stack/tree/master/examples)

