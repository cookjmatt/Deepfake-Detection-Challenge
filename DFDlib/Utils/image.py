import cv2

# Resize image to a maximum dimension while maintaining aspect ratio
def resize_to_max(img, max_dim):
    h, w = img.shape[:2]
    max_img = max(h, w)
    if max_img <= max_dim: return img
    ratio = max_dim / max_img
    dim = (int(w*ratio), int(h*ratio))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)