import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft

N = 1024

def gray(filename):
    '''transforms a colorful image into grayscale'''
    im = img.imread(filename)
    im_gray = np.dot(im[:, :, :3], [0.2989, 0.5870, 0.1140])
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

    #compute the kernel
    if type == 'gauss':
        W = Wx**2 + Wy**2
        K_hat = np.exp(-1 * W / par**2)

    elif type == 'lin_motion':
        K_hat = np.exp(1j*par * np.pi*Wx/N) * np.sinc(par*Wx / N)

    elif type == 'box':
        K_hat = np.sinc(par*Wx/N) * np.sinc(par*Wy/N)

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

def naive_sol_safe(K_hat, im, threshold=1e-5):
    '''source: Gemini
    naive solution with added safeguards to make computation possible'''
    g_hat = np.fft.fft2(im)
    
    # Create a mask of safe frequencies
    # We only divide where the kernel magnitude is larger than the threshold
    valid_mask = np.abs(K_hat) > threshold
    
    # Initialize result with zeros (or a copy of g_hat to preserve background)
    f_hat_rec = np.zeros_like(g_hat)
    
    # Perform division ONLY on safe indices
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
    #transform the image to frequency domain
    g_hat = fft.fft2(im)
    #prepare the frequency grid
    X = np.linspace(-1*N//2, N//2-1, num=N)
    Y = np.linspace(-1*N//2, N//2-1, num=N)
    Wx, Wy = np.meshgrid(X, Y)
    W = np.sqrt(Wx**2 + Wy**2) 
    #mask out very high frequencies
    W_Om = np.where(W < Om, 1, 0)
    W_Om = fft.ifftshift(W_Om)
    #apply the filter
    R_om = W_Om / K_hat
    f_Om_hat = R_om * g_hat
    #reconstruct the image
    f_Om = np.real(fft.ifft2(f_Om_hat))
    return f_Om, R_om

def error(R, K_hat, im_true, noise):
    f_hat = fft.fft2(im_true)
    noise_hat = fft.fft2(noise)

    g_hat = K_hat * f_hat 
    B = R*g_hat - f_hat 
    V = R*noise_hat

    bias = np.sum(np.abs(B)**2) / N**2
    variance = np.sum(np.abs(V)**2) / N**2
    return bias, variance

def find_best(par):
    if par == 'mu': 
        PAR = np.logspace(-14, 0, num=80)
    elif par == 'om':
        PAR = np.linspace(N//5, N//2, num=20)

    B=[]
    V=[]
    E=[]

    for par in PAR:
        if par == 'mu': _, R = tikh_filter('H2', par, K_hat, im_blurred_noised)
        elif par == 'om': _, R = spectral_window(par, K_hat, im_blurred_noised)
        bias, variance = error(R, K_hat, im, noise)
        B.append(bias)
        V.append(variance)
        E.append(bias + variance)
    argmin = np.argmin(E)
    min= PAR[argmin]
    return min, PAR, B, V, E

def find_Om():
    Om = np.linspace(N//5, N//2, num=10)
    B=[]
    V=[]
    E=[]
    for om in Om:
        _, R = spectral_window(om, K_hat, im_blurred_noised)
        bias, variance = error(R, K_hat, im, noise)
        B.append(bias)
        V.append(variance)
        E.append(bias + variance)
    argmin = np.argmin(E)
    min= Om[argmin]
    return min

def plot_error_old():
    Mu = np.logspace(-9, -1, num=80)
    B=[]
    V=[]
    E=[]
    for mu in Mu:
        _, RmuL2 = tikh_filter('H1', mu, K_hat, im_blurred_noised)
        bias, variance = error(RmuL2, K_hat, im, noise)
        B.append(bias)
        V.append(variance)
        E.append(bias + variance)

    argmin = np.argmin(E)
    min= Mu[argmin]

    plt.figure(figsize=(12,8))

    plt.semilogx(Mu, B, label='Bias (Approximation Error)', color='steelblue')
    plt.semilogx(Mu, V, label='Variance (Noise Error)', color='sandybrown')
    plt.semilogx(Mu, E, label='Total Error', color='lightcoral', linestyle = '--', linewidth=2)

    #plt.xscale('log') 
    plt.legend() 

    plt.xlabel('Regularization Parameter (mu)')
    plt.ylabel('Error Value')
    plt.title('Bias-Variance Trade-off')
    plt.grid(True, which="both", ls="-", alpha=0.5) 

    plt.show()
    return min

def plot_error(PAR, B, V ,E):
    plt.figure(figsize=(12,8))

    plt.semilogx(PAR, B, label='Bias (Approximation Error)', color='steelblue')
    plt.semilogx(PAR, V, label='Variance (Noise Error)', color='sandybrown')
    plt.semilogx(PAR, E, label='Total Error', color='lightcoral', linestyle = '--', linewidth=2)

    #plt.xscale('log') 
    plt.legend() 

    plt.xlabel('Regularization Parameter')
    plt.ylabel('Error Value')
    plt.title('Bias-Variance Trade-off')
    plt.grid(True, which="both", ls="-", alpha=0.5) 

    plt.show()

def plot_images():

    plt.figure(figsize=(36,24))

    plt.subplot(2, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title("original image")

    plt.subplot(2, 3, 2)
    plt.imshow(im_blurred, cmap='gray')
    plt.title("blurred image")

    plt.subplot(2, 3, 3)
    plt.imshow(im_blurred_noised, cmap='gray')
    plt.title("blurred image with noise")

    plt.subplot(2, 3, 4)
    plt.imshow(im_rec_naive, cmap='gray')
    plt.title("the naive solution")

    plt.subplot(2, 3, 5)
    plt.imshow(im_rec_tikh, cmap='gray')
    #plt.title(f"window solution (Om = {best_om})")
    plt.title("the tikhonov solution (H1 penalty)")

    #plt.subplot(2, 3, 6)
    #plt.imshow(im_rec_spect, cmap='gray')
    #plt.title("the spectral window solution")

    plt.show()


im = downscale_im(gray('oupi.jpeg'))
im_blurred, K_hat = apply_blur(im, 'lin_motion', 100)
im_blurred_noised, noise = add_noise_snr(im_blurred, 30)
min_err, PAR, B, V, E = find_best('mu')
im_rec_naive = naive_sol_safe(K_hat, im_blurred_noised)
im_rec_tikh, RmuL2 = tikh_filter('H1', min_err, K_hat, im_blurred_noised)
#im_rec_spect, R_om = spectral_window(150, K_hat, im_blurred_noised)
#best_om = find_Om()
#im_rec_spect, _ = spectral_window(best_om, K_hat, im_blurred_noised)
plot_error(PAR, B, V, E)
plot_images()