import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def evaluate_fusion(original_image, fused_image):
    # Convert the images to grayscale if they are RGB
    if original_image.ndim == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if fused_image.ndim == 3:
        fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2GRAY)

    # Evaluate SSIM, PSNR, and MSE
    ssim_index, _ = ssim(original_image, fused_image, full=True)
    psnr_value = psnr(original_image, fused_image)
    mse_value = mse(original_image, fused_image)

    return ssim_index, psnr_value, mse_value

def main():
    # Load your original and fused images
    original_image = cv2.imread("dataset/Patient Data/p1/ct.jpg")
    fused_image = cv2.imread("dataset/Patient Data/p1/fusion.jpg")
    # Check if the images have the same number of dimensions
    if original_image.ndim != fused_image.ndim:
        raise ValueError("Number of dimensions of the images must be the same.")

    # Evaluate fusion
    ssim_index, psnr_value, mse_value = evaluate_fusion(original_image, fused_image)

    # Print results
    print(f"Structural Similarity Index (SSIM): {ssim_index}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value} dB")
    print(f"Mean Squared Error (MSE): {mse_value}")

if __name__ == "__main__":
    main()
