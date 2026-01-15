import cv2
import matplotlib.pyplot as plt
import numpy as np

def center_crop(img, crop_width, crop_height):
    h, w = img.shape[:2]

    center_y, center_x = h // 2, w // 2

    start_x = center_x - crop_width // 2
    start_y = center_y - crop_height // 2
    end_x = start_x + crop_width
    end_y = start_y + crop_height

    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(w, end_x)
    end_y = min(h, end_y)

    cropped_img = img[start_y:end_y, start_x:end_x]
    
    return cropped_img.copy() 

def img_1():
    img = cv2.imread("imgs/TUMO_0.JPG")

    vhs_overlay = cv2.imread("imgs/VHS.avif")

    bCh = np.dot(img, [1, 0, 0]).astype(np.float32)
    gCh = np.dot(img, [0, 1, 0]).astype(np.float32)
    rCh = np.dot(img, [0, 0, 1]).astype(np.float32)

    rIntensity = 0.7
    gIntensity = 0.7
    bIntensity = 0.8

    noiseIntensity = 0.3

    rChF = rCh * rIntensity + (np.random.random(rCh.shape) - 0.5) * 150
    gChF = gCh * rIntensity + (np.random.random(gCh.shape) - 0.5) * 150
    bChF = bCh * bIntensity + (np.random.random(rCh.shape) - 0.5) * 150

    rChF = np.roll(rChF, axis=1, shift=25)

    reconstructed = cv2.add(np.stack(
        (
            np.clip(bChF, 0, 255).astype(np.uint8),
            np.clip(gChF, 0, 255).astype(np.uint8),
            np.clip(rChF, 0, 255).astype(np.uint8),
        ),
        axis=2
    ), (cv2.resize(vhs_overlay, (bCh.shape[1], bCh.shape[0]), interpolation=cv2.INTER_LINEAR)).astype(np.uint8))

    cv2.imwrite("out/VHS1.png", reconstructed)

    cv2.imshow("VHS Effect", reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_1()