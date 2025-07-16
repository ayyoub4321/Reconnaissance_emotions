import numpy as np
import cv2

def gabor_filter(roi):
    if roi is None or roi.size == 0:
        return 0
    """Applique un filtre de Gabor à la région d'intérêt (ROI)."""
    # Paramètres du filtre Gabor
    ksize = 31  # Taille du noyau
    sigma = 4.0  # Écart type de la gaussienne
    theta = np.pi / 4  # Angle de rotation du noyau
    lambd = 10.0  # Longueur d'onde
    gamma = 0.5  # Ratio d'aspect

    # Création du noyau Gabor
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0)
    gabor_output = cv2.filter2D(roi, cv2.CV_32F, gabor_kernel)
    
    return np.mean(gabor_output)
def sobel_filter(roi):
    if roi is None or roi.size == 0:
        return 0
    """Applique un filtre Sobel à la région d'intérêt (ROI)."""
    # Appliquer Sobel en X et Y
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

    # Combiner les résultats en calculant la magnitude du gradient
    sobel_output = cv2.magnitude(sobel_x, sobel_y)
    
    return np.mean(sobel_output)
def laplacian_filter(roi):
    if roi is None or roi.size == 0:
        return 0
    """Applique un filtre Laplacien à la région d'intérêt (ROI)."""
    laplacian_output = cv2.Laplacian(roi, cv2.CV_64F)
    
    return np.mean(laplacian_output)
from skimage.feature import hog
from skimage import exposure

def hog_filter(roi):
    if roi is None or roi.size == 0:
        return 0
    """Applique un filtre HOG à la région d'intérêt (ROI)."""
    # Conversion de l'image en niveaux de gris
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Extraire les descripteurs HOG
    fd, hog_image = hog(roi_gray, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)

    # Améliorer la visibilité de l'image HOG
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return np.mean(hog_image_rescaled)
# ajouter 
from skimage.feature import local_binary_pattern
def lbp_filter(roi):
    if roi is None or roi.size == 0:
        return 0
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(roi_gray, P=8, R=1, method="uniform")
    return np.mean(lbp)
