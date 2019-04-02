import numpy as np
from scipy.signal import convolve2d as conv2
from skimage import color, data
from matplotlib import pyplot as plt


# appliying the formula
def richardson_lucy(noisyimage, kernel, iterations):
    restoredimage = np.full((noisyimage.shape[0], noisyimage.shape[1]), 0.5)
    for i in range(iterations):
        restoredimage = restoredimage * conv2(noisyimage / conv2(restoredimage, kernel, 'same'), kernel, 'same')
    return restoredimage


astro = color.rgb2gray(data.astronaut())
psf = np.ones((5, 5)) / 25
astro = conv2(astro, psf, 'same')

# Add Noise to Image
astro_noisy = astro.copy()
astro_noisy += (np.random.poisson(lam=25, size=astro.shape) - 10) / 255.

# function call
deconvolved = richardson_lucy(astro_noisy, psf, 30)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()
for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(astro)
ax[0].set_title('Original Data')

ax[1].imshow(astro_noisy)
ax[1].set_title('Noisy data')

ax[2].imshow(deconvolved, vmin=astro_noisy.min(), vmax=astro_noisy.max())
ax[2].set_title('Restoration using\nRichardson-Lucy')

fig.subplots_adjust(wspace=0.02, hspace=0.2,top=0.9, bottom=0.05, left=0, right=1)
plt.savefig('results/part4/astro30.png')
plt.show()

