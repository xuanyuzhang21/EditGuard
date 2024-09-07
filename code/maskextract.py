import os
from PIL import Image
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.2, type=float, help='Path to option YMAL file.')
    args = parser.parse_args()

    input_folder = 'results/test_age-set'
    output_folder = 'results/mask'

    for filename in os.listdir(input_folder):
        if filename.endswith('_0_0_LRGT.png'):
            digits = filename.split('_')[0]
            if digits.isdigit():
                digits = int(digits)
                
                if digits >= 0 and digits <= 1000:

                    input_path_LRGT = os.path.join(input_folder, filename)
                    input_path_SR_h = os.path.join(input_folder, filename).replace('LRGT', 'SR_h')

                    image_LRGT = Image.open(input_path_LRGT).convert("RGB")
                    image_SR_h = Image.open(input_path_SR_h).convert("RGB")

                    w, h = image_SR_h.size
                    image_LRGT = image_LRGT.resize((w, h))
                    
                    array_LRGT = np.array(image_LRGT) / 255.
                    array_SR_h = np.array(image_SR_h) / 255.
                    
                    residual = np.abs(array_LRGT - array_SR_h)

                    threshold = args.threshold
                    mask = np.where(residual > threshold, 1, 0)
                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    output_path = os.path.join(output_folder, str(digits+1).zfill(4)+'.png')

                    mask = np.sum(mask, axis=2)

                    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_image.save(output_path)