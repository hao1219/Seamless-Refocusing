import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import List

def load_images_from_folder(folder):
    images, filenames = [], []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def write_images_into_folder(images,folder,filenames):
    if not os.path.exists(folder):
            os.makedirs(folder)
    for img＿idx,filename in enumerate(filenames):
        cv2.imwrite(os.path.join(folder, filename),images[img_idx])

    
def calculate_gradients_and_sharpness(images, M, N, dx_factor, dy_factor):
    laplacian_kernel = np.array([[0, -1, 0], 
                                 [-1, 4, -1], 
                                 [0, -1, 0]], dtype=np.float32)
    dx, dy = M // dx_factor, N // dy_factor
    gradients, sharpness_measure = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_gradient = cv2.filter2D(gray, -1, laplacian_kernel)
        gradients.append(laplacian_gradient)
        local_sharpness = cv2.boxFilter(laplacian_gradient, -1, (2*dx+1, 2*dy+1))
        sharpness_measure.append(local_sharpness)
    return gradients, sharpness_measure

def remove_border(threshold,images):
        def find_black_borders(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        borderd_imgs = []
        size_info_array = np.zeros((len(images), 4))
        for idx,image in enumerate(images):
            size_info_array[idx] = find_black_borders(image)
        size_info_array = size_info_array.astype(int)
        left    = np.max(size_info_array[:,0])
        right   = np.min(size_info_array[:,1])
        top     = np.max(size_info_array[:,2])
        bottom  = np.min(size_info_array[:,3])    
        for img in images:
            img = img[top:bottom,left:right]
            borderd_imgs.append(img)

        return borderd_imgs

def align_images(filenames,aligned_folder,images):
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
        if not os.path.exists(aligned_folder):
            os.makedirs(aligned_folder)
        aligned_imgs = []
        aligned_filenames = []

        detector = cv2.SIFT_create()
        
        base_image = images[0]
        base_filename = filenames[0].split('.')[0] + "_aligned.png"
        aligned_imgs.append(base_image)
        aligned_filenames.append(base_filename)
        base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        base_keypoints, base_descriptors = detector.detectAndCompute(base_image_gray, None)
        cv2.imwrite(os.path.join(aligned_folder, base_filename), base_image)

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

        
                #base_height, base_width = base_image.shape[:2]
                #cropped_img = aligned_img[0:base_height, 0:base_width]

                aligned_filename = filenames[i].split('.')[0] + "_aligned.png"
        

                aligned_imgs.append(aligned_img)
                aligned_filenames.append(aligned_filename)
                
        return aligned_imgs , aligned_filenames
         