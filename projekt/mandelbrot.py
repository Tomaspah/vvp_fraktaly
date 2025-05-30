import numpy as np
import matplotlib.pyplot as plt
import numba
from ipywidgets import interact, IntSlider, ToggleButtons, FloatSlider

@numba.njit(parallel=True)
def mandelbrot_set(x_min: float=-2.0, x_max: float=1.0, y_min: float=-1.5, y_max: float=1.5, n: int=500, k: int=100, x: float=0, y: float=0, zoom: float=1.0) -> np.ndarray:
    """
    Vypocet Mandelbrotovy mnoziny.

    Parametry:
        x_min, x_max (float): Rozsah realne osy (x).
        y_min, y_max (float): Rozsah imaginarni osy (y).
        n (int): Rozliseni obrazu (pocet bodu na kazde ose).
        k (int): Maximalni pocet iteraci pro kontrolu divergence.
        x, y (float): Posun obrazu v realne a imaginarni ose.
        zoom (float): Faktor priblizeni obrazu (1.0 = bez priblizeni).

    Vracena hodnota:
        ndarray: Matice s hodnotami iteraci do divergence pro kazdy bod v rovine.
    """
    
    # Nastaveni rozsahu a rozliseni
    scale = 1.0 / zoom
    x_min *= scale
    x_max *= scale
    y_min *= scale
    y_max *= scale
    re = np.linspace(x_min+x, x_max+x, n)
    im = np.linspace(y_min+y, y_max+y, n)
    
    divergence_matrix = np.full((n, n), k, dtype=np.int32)
    
    # Prochazeni vsech bodu v rovine
    for ix in numba.prange(n):
        for iy in range(n):
            c = re[ix] + 1j * im[iy]
            z = 0.0 + 0.0j
            
            # Iterace pro vypocet Mandelbrotovy posloupnosti
            for i in range(k):
                z = z * z + c
                
                # Kontrola divergence, pokud |z| > 2 (bez pouziti numpy, nebot to zpomaluje celou funcki)
                if (z.real * z.real + z.imag * z.imag) > 4.0:
                    divergence_matrix[ix, iy] = i
                    break
                    
    return divergence_matrix

def plot_mandelbrot(n: int=500, k: int=100, cmap: str='hot', x: float=0, y: float=0, zoom: float=1.0):
    """
    Vykresleni Mandelbrotovy mnoziny.

    Parametry:
        n (int): Rozliseni obrazu (pocet bodu na kazde ose).
        k (int): Maximalni pocet iteraci pro kontrolu divergence.
        cmap (str): Nazev barevne mapy pro vykresleni.
        x, y (float): Posun obrazu v realne a imaginarni ose.
        zoom (float): Faktor priblizeni obrazu (1.0 = bez priblizeni).
    """
    data = mandelbrot_set(n=n, k=k, x=x, y=y, zoom=zoom)
    plt.figure(figsize=(9, 8))
    plt.imshow(data.T, cmap=cmap, origin='lower', extent=[-2.0, 1.0, -1.5, 1.5])
    plt.colorbar(label='Počet iterací do divergence')
    plt.title(f'Mandelbrotova množina (k={k}, n={n})')
    plt.show()

def interactive_mandelbrot():
    """
    Vytvori interaktivni widget pro vykresleni Mandelbrotovy mnoziny.
    Umozni uzivateli menit parametry jako rozliseni, pocet iteraci, barevnou mapu, posun a priblizeni.
    """
    interact(plot_mandelbrot,
         n=IntSlider(min=50, max=2500, step=50, value=500, description='Rozlišení'),
         k=IntSlider(min=5, max=500, step=5, value=100, description='Iterace'),
         cmap=ToggleButtons(options=['hot', 'viridis', 'inferno', 'gray', 'turbo'], value='hot', description='Barevná mapa'),
         x=FloatSlider(min=-2.0, max=2.0, step=0.01, value=0.0, description='Posun v x'),
         y=FloatSlider(min=-2.0, max=2.0, step=0.01, value=0.0, description='Posun v y'),
         zoom=FloatSlider(min=1.0, max=50.0, step=0.5, value=1.0, description='Priblížení'));


def old_mandelbrot_set(x_min: float=-2, x_max: float=1, y_min: float=-1.5, y_max: float=1.5, n: int=1000, k: int=100) -> np.ndarray:
    """
    Vypocet Mandelbrotovy mnoziny bez pouziti Numba.

    Parametry:
        x_min, x_max (float): Rozsah realne osy (x).
        y_min, y_max (float): Rozsah imaginarni osy (y).
        n (int): Rozliseni obrazu (pocet bodu na kazde ose).
        k (int): Maximalni pocet iteraci pro kontrolu divergence.
    Vracena hodnota:
        ndarray: Matice s hodnotami iteraci do divergence pro kazdy bod v rovine.
    """
    re, im = np.mgrid[x_min:x_max:n*1j, y_min:y_max:n*1j]
    c = re + 1j * im

    z = np.zeros_like(c, dtype=np.complex128)
    divergence_matrix = np.full(c.shape, k)

    for i in range(k):
        mask = np.abs(z) < 2
        z[mask] = z[mask]**2 + c[mask]
        divergence_matrix[mask & (np.abs(z) > 2)] = i
        
    
    return divergence_matrix