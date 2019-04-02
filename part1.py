import cv2
import numpy as np
from matplotlib import pyplot as plt

blur1 = cv2.imread('part1/blur1.png', 0)
blur2 = cv2.imread('part1/blur2.png', 0)
blur3 = cv2.imread('part1/blur3.png', 0)
oriimg = cv2.imread('part1/original.jpg', 0)

# fft to convert the image to freq domain
fblur1 = np.fft.fft2(blur1)
fblur2 = np.fft.fft2(blur2)
fblur3 = np.fft.fft2(blur3)
fori = np.fft.fft2(oriimg)

def magspec(fim):
    # shift the center
    fshiftblur =np.fft.fftshift(fim)
    magnitude_spectrum = 20*np.log(np.abs(fshiftblur))
    return fshiftblur, magnitude_spectrum


fshiftblur1, magnitude_spectrum1 = magspec(fblur1)
fshiftblur2, magnitude_spectrum2 = magspec(fblur2)
fshiftblur3, magnitude_spectrum3 = magspec(fblur3)
fshiftori, magnitude_spectrumori = magspec(fori)


# ---------------------------------------------------------------

def decon(fshiftblur, fshiftor):
    # blur image / original image = kernel
    # after that, blur /kernel and use inverse fft = original image
    kernel = fshiftblur / fshiftor
    result = fshiftblur / kernel
    magnitude_spectrumkernel = 20 * np.log(np.abs(kernel))
    magnitude_spectrumresult = 20 * np.log(np.abs(result))
    return kernel, result, magnitude_spectrumkernel, magnitude_spectrumresult


kernel1, result1, magnitude_spectrumkernel1, magnitude_spectrumresult1 = decon(fshiftblur1, fshiftori)
kernel2, result2, magnitude_spectrumkernel2, magnitude_spectrumresult2 = decon(fshiftblur2, fshiftori)
kernel3, result3, magnitude_spectrumkernel3, magnitude_spectrumresult3 = decon(fshiftblur3, fshiftori)

def ifft(image):
    # inverse fft to get the image back
    fshift = image
    # shift back (shifted the center before)
    f_ishift = np.fft.ifftshift(fshift)
    d_shift = np.array(np.dstack([f_ishift.real, f_ishift.imag]))
    img_back = cv2.idft(d_shift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    return img_back


im_backresult1 = ifft(result1)
im_backresult2 = ifft(result2)
im_backresult3 = ifft(result3)

# images
plt.figure('images', figsize=(16,16))
plt.subplot(221), plt.imshow(blur1, cmap='gray')
plt.title('blur1'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(blur2, cmap='gray')
plt.title('blur2'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(blur3, cmap='gray')
plt.title('blur3'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(oriimg, cmap='gray')
plt.title('original image'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part1/images.png')
plt.show()

# kernels
plt.figure('kernels', figsize=(16,16))
plt.subplot(221), plt.imshow(magnitude_spectrumkernel1, cmap='gray')
plt.title('kernel1 magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrumkernel2, cmap='gray')
plt.title('kernel2 magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(magnitude_spectrumkernel3, cmap='gray')
plt.title('kernel3 magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(magnitude_spectrumori, cmap='gray')
plt.title('original img. mag. spectrum'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part1/kernels.png')
plt.show()

# deconvolution results

plt.figure('results', figsize=(16,16))
plt.subplot(221), plt.imshow(im_backresult1, cmap='gray')
plt.title('deconvolution result1'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(im_backresult2, cmap = 'gray')
plt.title('deconvolution result2'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(im_backresult3, cmap = 'gray')
plt.title('deconvolution result3'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(oriimg, cmap = 'gray')
plt.title('original image'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part1/deconresults.png')
plt.show()
