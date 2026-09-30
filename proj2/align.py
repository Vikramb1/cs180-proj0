import cv2
import numpy as np


def similarity_matrix(src, dst):
    (x1, y1), (x2, y2) = src
    (u1, v1), (u2, v2) = dst
    z = complex(u2 - u1, v2 - v1) / complex(x2 - x1, y2 - y1)
    a, b = z.real, z.imag
    return np.array([[a, -b, u1 - (a * x1 - b * y1)], [b, a, v1 - (b * x1 + a * y1)]])


def align(image, src, dst, size, border=cv2.BORDER_REFLECT_101):
    m = similarity_matrix(src, dst)
    out = cv2.warpAffine(image.astype(np.float32), m, size, flags=cv2.INTER_CUBIC, borderMode=border)
    return out.astype(np.float64).clip(0, 1)
