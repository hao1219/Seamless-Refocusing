import numpy as np
import time
import cv2

def refocusALL(all_foc_img, img_list,sharpness_measure):

        height, width           = sharpness_measure[0].shape
        max_gradient_indices    = np.argmax(sharpness_measure, axis=0)

        for h in range(height):
            for w in range(width):
                idx                 = max_gradient_indices[h][w]
                all_foc_img[h][w]   = img_list[idx][h][w]
        all_foc_img =all_foc_img.astype('uint8')
    
        return all_foc_img
def refocusALLKernel(coord ,sharpness_measure):

        h_low, h_roof, w_low, w_roof = coord
        sharpness_values    = np.sum(sharpness_measure[:, h_low:h_roof, w_low:w_roof], axis=(1, 2))
        sharpest_image_index = np.argmax(sharpness_values)

        return sharpest_image_index

def resize_image_for_display(image):
        target_width = 800
        target_height = 600
        resized_image = cv2.resize(image, (target_width, target_height))
        return resized_image


def all_focus(images,sharpness_measure):
        display_resize = 1
        all_focus_kernel    = [1,1]
        height_seg_len  = all_focus_kernel[0]
        width_seg_len   = all_focus_kernel[1]
        height, width   = images[0].shape[:2]
        all_foc_img     = np.empty(images[0].shape)

        if all_focus_kernel == [1,1]:
            all_foc_img = refocusALL(all_foc_img,images,sharpness_measure)

        else:
            height_seg      = np.ceil(height/height_seg_len).astype('int')
            width_seg       = np.ceil(width/width_seg_len).astype('int')
            for h_seg in range(height_seg):
                h_roof = min(height-1, (h_seg+1)*height_seg_len)
                for w_seg in range(width_seg):

                    w_roof = min(width-1, (w_seg+1)*width_seg_len)
                    h_low = h_seg * height_seg_len
                    w_low = w_seg * width_seg_len
                    idx = refocusALLKernel([h_low, h_roof, w_low, w_roof],sharpness_measure)
                    focus_final = images[idx][h_low:h_roof, w_low:w_roof,:]
                    all_foc_img[h_low:h_roof, w_low:w_roof, :] = focus_final
                    focus_zero  = all_foc_img[h_low:h_roof, w_low:w_roof, :] 
                    all_foc_img = all_foc_img.astype('uint8')

        if display_resize:
            all_foc_img = resize_image_for_display(all_foc_img) 

        return(all_foc_img)
        