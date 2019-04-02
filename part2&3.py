import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fp

blur1 = cv2.imread('part2/blur1.png', 0)
blur2 = cv2.imread('part2/blur2.png', 0)
blur3 = cv2.imread('part2/blur3.png', 0)

def ft(blur):
    # fft to convert the image to freq domain
    f = np.fft.fft2(blur)
    # shift the center
    fshiftt = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshiftt))
    return f, fshiftt, magnitude_spectrum


f1, fshift1, magnitude_spectrum1 = ft(blur1)
f2, fshift2, magnitude_spectrum2 = ft(blur2)
f3, fshift3, magnitude_spectrum3 = ft(blur3)

# -------------------------------------------PART2-----------------------------------------------------------------------------
# function for zero padding
def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

# generating the kernels according to angles
def kernel(size,angle):
    if angle == 0:
        blur_motion_kernel = np.zeros((size, size))
        blur_motion_kernel[int((size-1)/2), :] = np.ones(size)
        blur_motion_kernel = blur_motion_kernel / size
        blur_motion_kernel = np.lib.pad(blur_motion_kernel, (((blur1.shape[0] - size) // 2, (blur1.shape[0] - size) // 2 + 1),((blur1.shape[1] - size) // 2, (blur1.shape[1] - size) // 2 + 1)), padwithzeros)
        kernelF = np.fft.fft2(blur_motion_kernel)
        kernelmag = np.abs(kernelF)
        return blur_motion_kernel, kernelmag, kernelF

    elif angle == 90:

        blur_motion_kernel1 = np.zeros((size, size))
        blur_motion_kernel1[:, int((size - 1) / 2)] = np.ones(size)
        blur_motion_kernel1 = blur_motion_kernel1 / size
        blur_motion_kernel1 = np.lib.pad(blur_motion_kernel1, (((blur2.shape[0] - size) // 2, (blur2.shape[0] - size) // 2+1), ((blur2.shape[1] - size) // 2, (blur2.shape[1] - size) // 2 + 1)), padwithzeros)
        kernelF1 = np.fft.fft2(blur_motion_kernel1)
        kernelmag1 = np.abs(kernelF1)
        return blur_motion_kernel1, kernelmag1, kernelF1

    elif angle == 135:

        blur_motion_kernel = np.eye(size)
        blur_motion_kernel = blur_motion_kernel / size
        blur_motion_kernel = np.lib.pad(blur_motion_kernel, (((blur1.shape[0] - size) // 2, (blur1.shape[0] - size) // 2+1), ((blur1.shape[1] - size) // 2, (blur1.shape[1] - size) // 2 + 1)), padwithzeros)
        kernelF2 = np.fft.fft2(blur_motion_kernel)
        kernelmag2 = np.abs(kernelF2)
        return blur_motion_kernel, kernelmag2, kernelF2


kernel_motion_blur1, kernelmag1, kernelF1 = kernel(21, 0)
kernel_motion_blur2, kernelmag2, kernelF2 = kernel(21, 135)
kernel_motion_blur3, kernelmag3, kernelF3 = kernel(15, 90)

# -------------------------------------PART-3A------------------------------------------------------------------------

# noisy image = blur / kernel without threshold

def noisy(blur, kernel_motion_blur):
    f = fp.fft2(blur)
    freq_kernel = fp.fft2(fp.ifftshift(kernel_motion_blur))
    blur = f / (10**-10 + freq_kernel)
    im_blur = fp.ifft2(blur).real
    im_blur = im_blur / np.max(im_blur)

    return im_blur


noisy1 = noisy(blur1, kernel_motion_blur1)
noisy2 = noisy(blur2, kernel_motion_blur2)
noisy3 = noisy(blur3, kernel_motion_blur3)

# --------------------------------PART-3B--------------------------------------------------------------

# kernel thresholding and restoration operation

def ft(blur, kernel_motion_blur, thresvalue):
    freq = np.fft.fft2(blur)
    freq_kernel = np.fft.fft2(np.fft.ifftshift(kernel_motion_blur))
    # numpy where function for regularization
    kernell = np.where(abs(freq_kernel) > thresvalue, freq_kernel, 1)

    return freq, freq_kernel, kernell


freq1, freq_kernel1, kernel1 = ft(blur1, kernel_motion_blur1, 0.04)
freq2, freq_kernel2, kernel2 = ft(blur2, kernel_motion_blur2, 0.01)
freq3, freq_kernel3, kernel3 = ft(blur3, kernel_motion_blur3, 0.01)

def restore(freq, kernel):
    # blur / kernel and inverse fft = restored image
    fshift = freq / (10**-6 + kernel)
    # inverse fft to get image back
    f_ishift = np.fft.ifftshift(fshift)
    d_shift = np.array(np.dstack([f_ishift.real, f_ishift.imag]))
    restored = cv2.idft(d_shift)
    restored = cv2.magnitude(restored[:, :, 0], restored[:, :, 1])
    return restored


restored1 = restore(freq1, kernel1)
restored2 = restore(freq2, kernel2)
restored3 = restore(freq3, kernel3)


# ---------------------PART 2 RESULTS-------------------------------------

plt.figure('part 2 kernels', figsize=(9,18))
plt.subplot(311), plt.imshow(kernelmag1, cmap='gray')
plt.title('angle 0 kernel magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(312), plt.imshow(kernelmag2, cmap='gray')
plt.title('angle 135 kernel magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(313), plt.imshow(kernelmag3, cmap='gray')
plt.title('angle 90 kernel magnitude spectrum'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part2/kernels.png')
plt.show()

# ---------------------PART 3A RESULTS-------------------------------------
plt.figure('part3a', figsize=(12,18))
plt.subplot(321), plt.imshow(noisy1, cmap='gray', interpolation='nearest')
plt.title('blur/kernel'), plt.xticks([]), plt.yticks([])
plt.subplot(322), plt.imshow(blur1, cmap='gray', interpolation='nearest')
plt.title('blur1'), plt.xticks([]), plt.yticks([])
plt.subplot(323), plt.imshow(noisy2, cmap='gray', interpolation='nearest')
plt.title('blur/kernel'), plt.xticks([]), plt.yticks([])
plt.subplot(324), plt.imshow(blur2, cmap='gray', interpolation='nearest')
plt.title('blur2'), plt.xticks([]), plt.yticks([])
plt.subplot(325), plt.imshow(noisy3, cmap='gray', interpolation='nearest')
plt.title('blur/kernel'), plt.xticks([]), plt.yticks([])
plt.subplot(326), plt.imshow(blur3, cmap='gray', interpolation='nearest')
plt.title('blur3'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part3a/noisies.png')
plt.show()

# ---------------------PART 3B RESULTS-------------------------------------
plt.figure('restored image 1', figsize=(16, 8))
plt.subplot(122), plt.imshow(blur1, cmap='gray')
plt.title('blur1'), plt.xticks([]), plt.yticks([])
plt.subplot(121), plt.imshow(restored1, cmap='gray')
plt.title('restored image 1'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part3b/restored1.png')
plt.show()

plt.figure('restored image 2', figsize=(16, 8))
plt.subplot(122), plt.imshow(blur2, cmap='gray')
plt.title('blur2'), plt.xticks([]), plt.yticks([])
plt.subplot(121), plt.imshow(restored2, cmap='gray')
plt.title('restored image 2'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part3b/restored2.png')
plt.show()

plt.figure('restored image 3', figsize=(16, 8))
plt.subplot(121), plt.imshow(restored3, cmap='gray')
plt.title('restored image 3'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur3, cmap='gray')
plt.title('blur3'), plt.xticks([]), plt.yticks([])
plt.savefig('results/part3b/restored3.png')
plt.show()






