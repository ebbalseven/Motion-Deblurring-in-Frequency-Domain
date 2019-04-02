import cv2
import numpy as np


def bluredge(img, d=70):
    h, w = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motionkernel(angle, d, size=70):
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    size2 = size // 2
    A[:,2] = (size2, size2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kernel = cv2.warpAffine(kernel, A, (size, size), flags=cv2.INTER_CUBIC)
    return kernel

def defocuskernel(d, size=70):
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (size, size), d, 255, -1, cv2.LINE_AA, shift=1)
    kernel = np.float32(kernel) / 255.0
    return kernel

if __name__ == '__main__':

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['circle', 'angle=', 'd=', 'snr='])
    opts = dict(opts)

    def imread(imgno):
        blur = cv2.imread('part2/blur'+str(imgno)+'.png', 0)
        blur = np.float32(blur) / 255.0
        img = bluredge(blur)
        image = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

        return blur, image

    blur1, image1 = imread(1)
    blur2, image2 = imread(2)
    blur3, image3 = imread(3)

    win = 'deconvolution'
    defocus = '--circle' in opts

    def update(_, blur, image):
        ang = np.deg2rad(cv2.getTrackbarPos('angle', win))
        d = cv2.getTrackbarPos('d', win)
        noise = 10 ** (-0.1 * cv2.getTrackbarPos('SNR (db)', win))

        if defocus:
            psf = defocuskernel(d)
        else:
            psf = motionkernel(ang, d)
            cv2.imshow('psf', psf)

        psf /= psf.sum()
        psf_pad = np.zeros_like(blur)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
        PSF2 = (PSF ** 2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[..., np.newaxis]
        RES = cv2.mulSpectrums(image, iPSF, 0)
        res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        res = np.roll(res, -kh // 2, 0)
        res = np.roll(res, -kw // 2, 1)
        cv2.imshow(win, res)


    def trackbarshow(blur, img, angle, d, snr):
        cv2.namedWindow(win)
        cv2.namedWindow('psf', 0)
        cv2.createTrackbar('angle', win, int(opts.get('--angle', angle)), 180, update)
        cv2.createTrackbar('d', win, int(opts.get('--d', d)), 50, update)
        cv2.createTrackbar('SNR (db)', win, int(opts.get('--snr', snr)), 50, update)
        update(None, blur, img)
        cv2.waitKey()


    trackbarshow(blur1, image1, 0, 21, 26)
    trackbarshow(blur2, image2, 45, 30, 31)
    trackbarshow(blur3, image3, 90, 15, 27)
