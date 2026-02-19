import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from scipy.special import j1

N = 1024

def gray(filename):
    '''transforms a colorful image into grayscale'''
    im = img.imread(filename)
    im_gray = np.dot(im[:, :, :3], [0.299, 0.587, 0.114])
    return im_gray 

def downscale_im(im):
    '''rescales image to [0,1] range '''
    diff = im.max() - im.min()
    im_scaled = (im - im.min())/diff
    return im_scaled

def apply_blur(image, type, par):
    '''constructs the appropriate kernel and applies blur on a transformed image (frequency domain)'''
    #define the meshgrid
    X = np.linspace(-1*N//2, N//2-1, num=N)
    Y = np.linspace(-1*N//2, N//2-1, num=N)
    Wx, Wy = np.meshgrid(X, Y) #grid of integers k, s.t. the associated frequencies are kpi/N
    W = Wx**2 + Wy**2

    #compute the kernel
    if type == 'gauss':
        K_hat = np.exp(-1 * W / (2*par**2))

    elif type == 'lin_motion':
        K_hat = np.exp(-1j*par * np.pi*Wx/N) * np.sinc(par*Wx / N)

    #elif type == 'box':
    #    K_hat = np.sinc(par*Wx/N) * np.sinc(par*Wy/N)

    elif type == 'unfocus': 
        arg = par * np.sqrt(W) * 2 * np.pi / N
        arg = np.where(W==0, 1, arg)
        K_hat = 2 * j1(arg) / arg
        K_hat = np.where(W==0, 1, K_hat)
        #(taking lim j1(x)/x = 1 as x -> 0)


    K_hat = fft.ifftshift(K_hat)
    #compute g = K * f_hat 
    f_hat = fft.fft2(image)
    #apply the filter
    g = f_hat * K_hat 
    #reconstruct image with inverse transform
    g_rec = np.real(fft.ifft2(g))
    return g_rec, K_hat

def add_noise_snr(im, snr):
    '''adds gaussian noise based on the desired signal-to-noise ratio'''
    #find sigma
    sigma = 10**((-1)*snr/20) * np.std(im)
    #create noise matrix
    noise = np.random.normal(0, sigma, (N, N))
    #apply noise
    im_noised = im + noise
    return im_noised, noise

def naive_sol(K_hat, im, threshold=1e-5):
    '''naive solution with added safeguards to make computation possible'''
    g_hat = np.fft.fft2(im)  
    # Create a mask of safe frequencies
    valid_mask = np.abs(K_hat) > threshold
    f_hat_rec = np.zeros_like(g_hat)
    f_hat_rec[valid_mask] = g_hat[valid_mask] / K_hat[valid_mask]
    
    f_rec = np.real(np.fft.ifft2(f_hat_rec))
    return f_rec

def tikh_filter(penalty, mu, K_hat, im):
    '''computes the Tikhonov Filter with a given type of penalty'''
    g_hat = fft.fft2(im)
    if penalty == 'L2':
        P = np.ones((N, N))
    elif penalty == 'H1' or penalty == 'H2':
        X = np.linspace(-1*N//2, N//2-1, num=N)
        Y = np.linspace(-1*N//2, N//2-1, num=N)
        Wx, Wy = np.meshgrid(X, Y)
        P = Wx**2 + Wy**2

        if penalty == 'H2': P = P**2

    P_hat = fft.ifftshift(P)
    Rmu = np.conj(K_hat) / (np.abs(K_hat)**2 + mu*P_hat )
    fmu_hat = Rmu * g_hat
    fmu = np.real(fft.ifft2(fmu_hat))
    return fmu, Rmu

def spectral_window(Om, K_hat, im):
    '''reconstructs the image after masking out frequencies above a threshold'''
    #transform the image to frequency domain
    g_hat = fft.fft2(im)
    #prepare the frequency grid
    X = np.linspace(-1*N//2, N//2-1, num=N)
    Y = np.linspace(-1*N//2, N//2-1, num=N)
    Wx, Wy = np.meshgrid(X, Y)
    W = np.sqrt(Wx**2 + Wy**2) 
    #mask out very low values of K_hat
    valid_mask = np.abs(K_hat) > 1e-5
    #mask out very high frequencies
    R_Om = np.zeros_like(g_hat)
    W_Om = np.where(W < Om, 1, 0)
    W_Om = fft.ifftshift(W_Om)
    #apply the filter
    R_Om[valid_mask] = W_Om[valid_mask] / K_hat[valid_mask]
    f_Om_hat = R_Om * g_hat
    #reconstruct the image
    f_Om = np.real(fft.ifft2(f_Om_hat))
    return f_Om, R_Om

def error(R, K_hat, im_true, noise):
    f_hat = fft.fft2(im_true)
    noise_hat = fft.fft2(noise)

    g_hat = K_hat * f_hat 
    B = R*g_hat - f_hat 
    V = R*noise_hat
    E = R*(g_hat + noise_hat) - f_hat

    bias = np.sum(np.abs(B)**2) / N**4
    variance = np.sum(np.abs(V)**2) / N**4
    err = np.sum(np.abs(E)**2 / N**4)
    return bias, variance, err

def find_Om(K_hat, im_blurred_noised, im, noise):
    Om = np.linspace(0, 1.4*N, num=140)
    B=[]
    V=[]
    E=[]
    for om in Om:
        _, R = spectral_window(om, K_hat, im_blurred_noised)
        bias, variance, err = error(R, K_hat, im, noise)
        B.append(bias)
        V.append(variance)
        E.append(err)

    min_error = np.min(E)
    argmin = np.argmin(E)
    best_om= Om[argmin]

    '''
    plt.figure(figsize=(12,8))

    plt.plot(Om, B, label='Bias (Approximation Error)', color='steelblue')
    plt.plot(Om, V, label='Variance (Noise Error)', color='sandybrown')
    plt.plot(Om, E, label='Total Error', color='lightcoral', linestyle = '--', linewidth=2)

    #plt.xscale('log') 
    plt.legend() 

    plt.xlabel('Regularization Parameter (omega)')
    plt.ylabel('Error Value')
    plt.title('Bias-Variance Trade-off')
    plt.grid(True, which="both", ls="-", alpha=0.5) 

    plt.show()
    '''
    return best_om, min_error

def plot_error(penalty, K_hat, im_blurred_noised, im, noise):
    #if penalty == 'L2' : Mu = np.logspace(-4, -1, num=9)
    #if penalty == 'H1' : Mu = np.logspace(-9, -6, num=9)
    #elif penalty == 'H2': Mu = np.logspace(-13, -10, num=9)
    Mu= np.logspace(-15, 0, num=30)

    B=[]
    V=[]
    E=[]
    for mu in Mu:
        _, Rmu = tikh_filter(penalty, mu, K_hat, im_blurred_noised)
        bias, variance, err = error(Rmu, K_hat, im, noise)
        B.append(bias)
        V.append(variance)
        E.append(err)

    min_error = np.min(E)
    argmin = np.argmin(E)
    best_mu= Mu[argmin]
    '''
    plt.figure(figsize=(12,8))

    plt.semilogx(Mu, B, label='Bias (Approximation Error)', color='steelblue')
    plt.semilogx(Mu, V, label='Variance (Noise Error)', color='sandybrown')
    plt.semilogx(Mu, E, label='Total Error', color='lightcoral', linestyle = '--', linewidth=2)

    plt.axvline(x=best_mu, color='gray', linestyle = '--')

    #plt.xscale('log') 
    plt.legend(fontsize=14) 

    plt.xlabel('Regularization Parameter (mu)', fontsize=16)
    plt.ylabel('Error Value', fontsize=16)
    plt.title('Bias-Variance Trade-off', fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.5) 

    plt.show()
    '''
    return best_mu, min_error


def plot_images():
    '''saves each reconstructed image and it's error, creates a comparison image of all reconstructions'''
    errors = dict()
    
    im = downscale_im(gray('dog.jpeg'))
    plt.figure(figsize=(36,24))
    plt.imshow(im, cmap='gray')
    plt.savefig('reconstructed/original.png')
    
    im_blurred, K_hat = apply_blur(im, 'lin_motion', blur_pars['lin_motion'])
    im_blurred_noised, noise = add_noise_snr(im_blurred, 40)
    plt.figure(figsize=(36,24))
    plt.imshow(im_blurred_noised, cmap='gray')
    plt.savefig('reconstructed/blurred.png')
    '''
    im_rec_naive = naive_sol(K_hat, im_blurred_noised)
    plt.figure(figsize=(36,24))
    plt.imshow(im_rec_naive, cmap='gray')
    plt.savefig('reconstructed/naive.png')
    
    best_om, errors['spectral'] = find_Om(K_hat, im_blurred_noised, im, noise)
    im_rec_spect, R_om = spectral_window(best_om, K_hat, im_blurred_noised)
    plt.figure(figsize=(36,24))
    plt.imshow(im_rec_spect, cmap='gray')
    plt.savefig('reconstructed/spectral.png')
    
    best_muL2, errors['L2'] = plot_error('L2', K_hat, im_blurred_noised, im, noise)
    im_rec_tikhL2, Rmu = tikh_filter("L2", best_muL2, K_hat, im_blurred_noised)
    plt.figure(figsize=(36,24))
    plt.imshow(im_rec_tikhL2, cmap='gray')
    plt.savefig('reconstructed/tikhL2.png')
    '''
    best_muH1, errors['H1'] = plot_error('H1', K_hat, im_blurred_noised, im, noise)
    im_rec_tikhH1, Rmu = tikh_filter("H1", best_muH1, K_hat, im_blurred_noised)
    plt.figure(figsize=(36,24))
    plt.imshow(im_rec_tikhH1, cmap='gray')
    plt.savefig('reconstructed/tikhH1.png')

    print(best_muH1)
    '''
    best_muH2, errors['H2'] = plot_error('H2', K_hat, im_blurred_noised, im, noise)
    im_rec_tikhH2, Rmu = tikh_filter("H2", best_muH2, K_hat, im_blurred_noised)
    plt.figure(figsize=(36,24))
    plt.imshow(im_rec_tikhH2, cmap='gray')
    plt.savefig('reconstructed/tikhH2.png')
    
    best_rec = min(errors, key=errors.get)
    print('Best reconstruction: ', best_rec)
    print(best_om, best_muL2, best_muH1, best_muH2)

    #comparative plot of original and reconstructed images
    plt.figure(figsize=(36,24))
    plt.subplot(2, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title("original image", fontsize=48)

    plt.subplot(2, 3, 2)
    plt.imshow(im_blurred_noised, cmap='gray')
    plt.title("blurred image with noise", fontsize=48)

    plt.subplot(2, 3, 3)
    plt.imshow(im_rec_naive, cmap='gray')
    plt.title("the naive solution", fontsize=48)

    plt.subplot(2, 3, 4)
    plt.imshow(im_rec_spect, cmap='gray')
    plt.title("spectral window solution", fontsize=48)

    plt.subplot(2, 3, 5)
    plt.imshow(im_rec_tikhL2, cmap='gray')
    plt.title("tikhonov solution (L2)", fontsize=48)

    plt.subplot(2, 3, 6)
    plt.imshow(im_rec_tikhH1, cmap='gray')
    plt.title("tikhonov solution (H1)", fontsize=48)

    plt.savefig('reconstructed/compared.png')
    '''
blur_pars = {
    'gauss': 30, #lower value more blur
    'lin_motion': 40, #higher value more blur
    'unfocus': 8, #higher value more blur
}

plot_images()