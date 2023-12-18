import cv2
import numpy as np
import os
from typing import List
from tqdm import tqdm
import image_processing as ip
import time
import argparse
from all_focus import all_focus

class FocusStacker:
    def __init__(self, folder, aligned_folder,dx_factor=200, dy_factor=200,black_threshold = 30):
        self.folder = folder
        self.dx_factor = dx_factor
        self.dy_factor = dy_factor
        self.black_threshold = black_threshold
        self.aligned_folder = os.path.join('focal_stacks',aligned_folder)
        
        
        
        self.images = []
        self.filenames = []
        self.aligned_images = []
        self.filenames_alligned = []
        
        #loaf original images in focal stack
        self.images, self.filenames = ip.load_images_from_folder(self.folder)
        #align original images (without cropping)
        self.aligned_images,self.filenames_alligned = ip.align_images(self.filenames,self.aligned_folder,self.images)
        #crop the border due to homography
        self.aligned_images = ip.remove_border(self.black_threshold,self.aligned_images)
        #write aligned_images into aligned dir
        ip.write_images_into_folder(self.aligned_images,self.aligned_folder,self.filenames_alligned)
        self.M, self.N = self.aligned_images[0].shape[:2]
        #aligned imgs gradient and sharpness
        self.gradients_aligned,self.sharpness_measure_aligned = ip.calculate_gradients_and_sharpness(self.aligned_images,self.M,self.N,self.dx_factor,self.dy_factor)
        
        #original imgs gradient and sharpness
        self.M, self.N = self.images[0].shape[:2]
        self.gradients_aligned_non_aligned,self.sharpness_measure_non_aligned = ip.calculate_gradients_and_sharpness(self.images,self.M,self.N,self.dx_factor,self.dy_factor)
 

    def _refocus(self, x, y, isAligned):
        measure = self.sharpness_measure_aligned if isAligned else self.sharpness_measure_non_aligned
        sharpness_values = [sm[y, x] for sm in measure]
        sharpest_image_index = sharpness_values.index(max(sharpness_values))
        return sharpest_image_index

    def show_image_with_certain_focal_point(self, window_name, isAligned):
        if isAligned:
            img_file = self.aligned_images
            filenames = self.filenames_alligned
        else:
            img_file = self.images
            filenames = self.filenames

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"pixel({x},{y})")
                sharpest_image_index = self._refocus(x, y ,isAligned)
                filename = filenames[sharpest_image_index]
                display_image = img_file[sharpest_image_index].copy()
                cv2.putText(display_image, filename, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(window_name, display_image)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, self.images[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_all_focus(self):
        all_foc_img = all_focus(self.aligned_images,self.sharpness_measure_aligned)
        cv2.imshow("ALL_FOCUS", all_foc_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Focus Stacker')
    parser.add_argument('-i', '--input', default='test', help='Input directory')
    parser.add_argument('-dx', '--dx_factor', type=int, default=200, help='dx factor for sharpness calculation')
    parser.add_argument('-dy', '--dy_factor', type=int, default=200, help='dy factor for sharpness calculation')
    parser.add_argument('-bt', '--black_threshold', type=int, default=30, help='Threshold for black border removal')
    parser.add_argument('-all', '--all_focus', action='store_true', help='create all focus image')
    parser.add_argument('--aligned', dest='aligned', action='store_true')
    parser.add_argument('--not-aligned', dest='aligned', action='store_false')
    parser.set_defaults(aligned=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    focus_stacker = FocusStacker(os.path.join('focal_stacks', args.input), 
                                 args.input + '_aligned',
                                 dx_factor=args.dx_factor,
                                 dy_factor=args.dy_factor,
                                 black_threshold=args.black_threshold)
    if args.all_focus:
        focus_stacker.show_all_focus()
    elif args.aligned is not None:
        focus_stacker.show_image_with_certain_focal_point("Refocus Window", isAligned=args.aligned)
    # Rest of the main function

if __name__ == '__main__':
    main()