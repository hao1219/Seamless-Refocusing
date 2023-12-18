import cv2
import numpy as np
import os
from typing import List
from tqdm import tqdm
import time
import argparse

class FocusStacker:
    def __init__(self, folder, aligned_folder,dx_factor=100, dy_factor=100):
        self.folder = folder
        self.dx_factor = dx_factor
        self.dy_factor = dy_factor
        self.aligned_folder = aligned_folder
        self.display_resize = 1
        self.all_focus_kernel    = [1,1]
        self.laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        self.filenames = []
        self.filenames_alligned = []
        self.images, self.filenames = self._load_images_from_folder(self.folder)

        self._align_images(self.images)
        # Load the aligned images for further processing
        self.aligned_images , self.filenames_alligned = self._load_images_from_folder(self.aligned_folder)
        self._remove_border()
        #load again
        self.aligned_images , self.filenames_alligned = self._load_images_from_folder(self.aligned_folder)
        self.M, self.N = self.aligned_images[0].shape[:2]
        self.gradients,self._sharpness_measure = self._calculate_gradients_and_sharpness(isAligned = True)
        
        #original
        self.M, self.N = self.images[0].shape[:2]
        self.gradients_non_aligned,self._sharpness_measure_non_aligned = self._calculate_gradients_and_sharpness(isAligned = False)
   

    def _load_images_from_folder(self, folder):
        images = []
        filenames = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
                #print(filename)
                #print(img.shape)
                
        return images,filenames
        
    def _calculate_gradients_and_sharpness(self, isAligned):
        dx, dy = self.M // self.dx_factor, self.N // self.dy_factor
        gradients = []
        sharpness_measure = []

        images_to_process = self.aligned_images if isAligned else self.images

        for img in images_to_process:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_gradient = cv2.filter2D(gray, -1, self.laplacian_kernel)
            #print(laplacian_gradient.shape)
            gradients.append(laplacian_gradient)
            local_sharpness = cv2.boxFilter(laplacian_gradient, -1, (2*dx+1, 2*dy+1))
            sharpness_measure.append(local_sharpness)
            #print(local_sharpness.shape)
            

        #print(sharpness_measure[0].shape)
        return gradients, sharpness_measure

    def refocus(self, x, y, isAligned):
        measure = self._sharpness_measure if isAligned else self._sharpness_measure_non_aligned
        sharpness_values = [sm[y, x] for sm in measure]
        sharpest_image_index = sharpness_values.index(max(sharpness_values))
        return sharpest_image_index

    def _align_images(self, images: List[np.ndarray]):
        def _find_homography(
            _img1_key_points: np.ndarray, _image_2_kp: np.ndarray, _matches: List
        ):
            image_1_points = np.zeros((len(_matches), 1, 2), dtype=np.float32)
            image_2_points = np.zeros((len(_matches), 1, 2), dtype=np.float32)

            for j in range(0, len(_matches)):
                image_1_points[j] = _img1_key_points[_matches[j].queryIdx].pt
                image_2_points[j] = _image_2_kp[_matches[j].trainIdx].pt

            homography, mask = cv2.findHomography(
                image_1_points, image_2_points, cv2.RANSAC, ransacReprojThreshold=2.0
            )
            return homography
        if not os.path.exists(self.aligned_folder):
            os.makedirs(self.aligned_folder)
        filenames = os.listdir(self.folder)
        aligned_imgs = []
        detector = cv2.SIFT_create()

        base_image = images[0]
        base_filename = filenames[0].split('.')[0] + "_aligned.png"
        base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_keypoints, base_descriptors = detector.detectAndCompute(base_image_gray, None)
        cv2.imwrite(os.path.join(self.aligned_folder, base_filename), base_image)

        for i in tqdm(range(1, len(images)), desc="Aligning images"):
            image = images[i]
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(image_gray, None)

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors, base_descriptors, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

            if len(good_matches) > 100:
                src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([base_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    
                aligned_img = cv2.warpPerspective(image, homography_matrix, (image.shape[1], image.shape[0]))

        
                base_height, base_width = base_image.shape[:2]
                cropped_img = aligned_img[0:base_height, 0:base_width]

                aligned_filename = filenames[i].split('.')[0] + "_aligned.png"
                cv2.imwrite(os.path.join(self.aligned_folder, aligned_filename), aligned_img)

    def _remove_border(self):
        def find_black_borders(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = 10  # Define black threshold

            rows, cols = gray.shape
            left, right, top, bottom = 0, cols, 0, rows

            for i in range(cols):
                if np.sum(gray[:, i]) > threshold * rows:
                    left = i
                    break

            for i in range(cols - 1, -1, -1):
                if np.sum(gray[:, i]) > threshold * rows:
                    right = i
                    break

            for i in range(rows):
                if np.sum(gray[i, :]) > threshold * cols:
                    top = i
                    break

            for i in range(rows - 1, -1, -1):
                if np.sum(gray[i, :]) > threshold * cols:
                    bottom = i
                    break

            return np.array([left, right, top, bottom])

        images = self.aligned_images
        borderd_imgs = []
        size_info_array = np.zeros((len(images), 4))
        for idx,image in enumerate(images):
            size_info_array[idx] = find_black_borders(image)
        size_info_array = size_info_array.astype(int)
        left    = np.max(size_info_array[:,0])
        right   = np.min(size_info_array[:,1])
        top     = np.max(size_info_array[:,2])
        bottom  = np.min(size_info_array[:,3])    
        for idx,image in enumerate(images):
            img_path = os.path.join(self.aligned_folder, self.filenames_alligned[idx])
            image = image[top:bottom,left:right]
            borderd_imgs.append(image)
            cv2.imwrite(img_path,image)

        return borderd_imgs
    def _refocusALLKernel(self, coord):

        h_low, h_roof, w_low, w_roof = coord
        measure             = self._sharpness_measure.copy()
        sharpness_values    = np.sum(measure[:, h_low:h_roof, w_low:w_roof], axis=(1, 2))
        sharpest_image_index = np.argmax(sharpness_values)

        return sharpest_image_index

    def _refocusALL(self, ALL_FOC_IMG, img_list):

        measure                 = self._sharpness_measure.copy()
        height, width           = measure[0].shape
        max_gradient_indices    = np.argmax(measure, axis=0)
        # print(max_gradient_indices.shape, max_gradient_indices[0])

        for h in range(height):
            for w in range(width):
                idx                 = max_gradient_indices[h][w]
                ALL_FOC_IMG[h][w]   = img_list[idx][h][w]
        ALL_FOC_IMG =ALL_FOC_IMG.astype('uint8')
    
        return ALL_FOC_IMG
    def _resize_image_for_display(self, image):
        target_width = 800
        target_height = 600
        resized_image = cv2.resize(image, (target_width, target_height))

        return resized_image
    def all_focus(self, isAligned=True):
        all_focal_runtime = time.time()
        height_seg_len  = self.all_focus_kernel[0]
        width_seg_len   = self.all_focus_kernel[1]
        images_list     = self.aligned_images.copy() if isAligned else self.images.copy()
        height, width   = images_list[0].shape[:2]
        all_foc_img     = np.empty(images_list[0].shape)

        if self.all_focus_kernel == [1,1]:
            all_foc_img = self._refocusALL(all_foc_img,images_list)

        else:
            height_seg      = np.ceil(height/height_seg_len).astype('int')
            width_seg       = np.ceil(width/width_seg_len).astype('int')
            for h_seg in range(height_seg):
                h_roof = min(height-1, (h_seg+1)*height_seg_len)
                for w_seg in range(width_seg):
                    print(h_seg, w_seg)
                    w_roof = min(width-1, (w_seg+1)*width_seg_len)
                    h_low = h_seg * height_seg_len
                    w_low = w_seg * width_seg_len
                    idx = self._refocusALLKernel([h_low, h_roof, w_low, w_roof])
                    focus_final = images_list[idx][h_low:h_roof, w_low:w_roof,:]
                    all_foc_img[h_low:h_roof, w_low:w_roof, :] = focus_final
                    focus_zero  = all_foc_img[h_low:h_roof, w_low:w_roof, :] 
                    all_foc_img = all_foc_img.astype('uint8')
        print(f'All focus image RunTime: {time.time() - all_focal_runtime:.4f}\nshape: {all_foc_img.shape}')
        if self.display_resize:
            all_foc_img = self._resize_image_for_display(all_foc_img) 
        cv2.imshow("ALL_FOCUS", all_foc_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                sharpest_image_index = self.refocus(x, y ,isAligned)
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Focus Stacker')
    parser.add_argument('-i', '--input', default='test', help='Input directory')
    parser.add_argument('-all', '--all_focus', action='store_true', help='Call all_focus function')
    parser.add_argument('--aligned', dest='aligned', action='store_true')
    parser.add_argument('--not-aligned', dest='aligned', action='store_false')
    parser.set_defaults(aligned=True)
    return parser.parse_args()

def main():
    args = parse_arguments()
    focus_stacker = FocusStacker(os.path.join('focal_stacks',args.input), args.input + '_aligned')
    if args.all_focus:
        focus_stacker.all_focus()
    elif args.aligned is not None:
        focus_stacker.show_image_with_certain_focal_point("Refocus Window", isAligned=args.aligned)

if __name__ == '__main__':
    main()