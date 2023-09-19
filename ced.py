import numpy as np
from PIL import Image
import scipy
import os

def gaussian_kernel(size, sigma=1): 
  # size = 2k+1
  k = size // 2
  x, y = np.mgrid[-k:k+1, -k:k+1]
  coef = 1 / (2.0 * np.pi * sigma ** 2)
  kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2) ) * coef
  return kernel

filename = 'simple_cat'
filetype = 'jpg'

if not os.path.isdir(filename):
  os.makedirs(filename)
image = Image.open(f'{filename}.{filetype}').convert('L') # converts to greyscale, alternatively could do equation: Y' = 0.2989 R + 0.5870 G + 0.1140 B?
image = np.asarray(image).astype(np.float32) # Doing convolutions, keep type as float until no longer needed

# def convolute2D(image, kernel):
#   result = np.ones_like(image)
#   K = kernel.shape[0] // 2
#   N, M = image.shape
#   for i in range(0, N):
#     for j in range(0, M):
#       for r in range(-K, K+1):
#         for c in range(-K, K+1):
#           if 0 <= i+r and i+r < N and 0 <= j+c and j+c < M:
#             result[i][j] += 1.0 * kernel[K+r][K+c] * image[i+r][j+c]
#   return result
  
# Noise reduce w/ Gaussian kernel
# Relying on derivatives, highly sensitive to noise!
# result = convolute2D(np.asarray(image), gaussian_kernel(5, 1.4))
blurred_image = scipy.ndimage.convolve(image, gaussian_kernel(5, 0.64), mode='constant', cval=0.0) # 0.64 is the magic number (apparently)
Image.fromarray(np.round(blurred_image).astype(np.uint8)).save(f"{filename}/blurred_{filename}_G_5x5.png")

# Take gradients and angles in x and y direction
def sobel_filter(image):
  Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
  Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

  Ix = scipy.ndimage.convolve(image, Kx, mode='constant', cval=0.0)
  Iy = scipy.ndimage.convolve(image, Ky, mode='constant', cval=0.0)

  result = np.hypot(Ix, Iy)
  result = result / result.max() * 255.0
  theta = np.arctan2(Iy, Ix) * 180 / np.pi
  theta[theta < 0] += 180
  return result, theta

gradient_intensity_image, theta = sobel_filter(blurred_image)
gradient_intensity_image = np.round(gradient_intensity_image).astype(np.uint8) # no more convolutions, save as uint8
Image.fromarray(gradient_intensity_image).save(f"{filename}/gi_{filename}_G_5x5.png")

def non_maximum_supression(image, angle):
  N, M = image.shape
  result = np.zeros_like(image)
  # thresholds = np.array([[157.5, 0, -1, 0, 1], [112.5, -1, -1, 1, 1], [67.5, -1, 0, 1, 0], [22.5, -1, 1, 1, -1], [0, 0, 1, 0, -1]], np.int32)
  for i in range(1, N-1):
    for j in range(1, M-1):
        q = 255
        r = 255
        
        #angle 0
        if 0 <= angle[i,j] < 22.5 or 157.5 <= angle[i,j] <= 180:
            q = image[i, j+1]
            r = image[i, j-1]
        #angle 45
        elif 22.5 <= angle[i,j] < 67.5:
            q = image[i+1, j-1]
            r = image[i-1, j+1]
        #angle 90
        elif 67.5 <= angle[i,j] < 112.5:
            q = image[i+1, j]
            r = image[i-1, j]
        #angle 135
        elif 112.5 <= angle[i,j] < 157.5:
            q = image[i-1, j-1]
            r = image[i+1, j+1]

        if image[i,j] >= q and image[i,j] >= r:
            result[i,j] = image[i,j]
      # q, r = 0, 0
      # for alpha, r1, c1, r2, c2 in thresholds:
      #   if theta[i][j] > alpha:
      #     q = image[i+r1][j+c1]
      #     r = image[i+r2][j+c2]
      #     break
      #   if image[i][j] > q and image[i][j] > r:
      #     result[i][j] = image[i][j]

  return result

nms_image = non_maximum_supression(gradient_intensity_image, theta)
Image.fromarray(nms_image).save(f"{filename}/nms_{filename}_G_5x5.png")

def double_threshold(image, ltr=0.005, htr=0.1):
  high_threshold = htr * image.max()
  low_threshold = ltr * image.max()

  result = np.zeros_like(image)
  result[image >= high_threshold] = 255
  result[(image >= low_threshold) & (image < high_threshold)] = 255 // 2
  result[image < low_threshold] = 0

  return result


dt_image = double_threshold(nms_image)
Image.fromarray(dt_image).save(f"{filename}/dt_{filename}_G_5x5.png")

def hysteresis(image, weak = 255 // 2, strong = 255):
  N, M = image.shape
  result = np.zeros_like(image)
  move = np.array([[1, 1, -1, -1, 1, -1, 0, 0], [1, -1, 1, -1, 0, 0, 1, -1]], np.int32)
  for i in range(N):
    for j in range(M):
        if image[i][j] == weak:
          count = 0
          for r, c in zip(move[0], move[1]):
            if 0 <= i + r and i + r < N and 0 <= j + c and j + c < M and image[i+r][j+c] == strong:
              count += 1
          if count > 0:
            result[i][j] = strong
        else:
          result[i][j] = image[i][j]
  return result

hys_image = hysteresis(dt_image)
Image.fromarray(hys_image).save(f"{filename}/hys_{filename}_G_5x5.png")