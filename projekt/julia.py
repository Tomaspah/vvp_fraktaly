import numpy as np
import matplotlib.pyplot as plt
import numba
from ipywidgets import interact, IntSlider, ToggleButtons, FloatSlider

@numba.njit(parallel=True)
def julia_set(c_real: float=-0.7, c_imag: float=0.27, x_min: float=-2.0, x_max: float=2.0, y_min: float=-2.0, y_max: float=2.0, n: int=1000, k: int=200, x: float=0.0, y: float=0.0, zoom: float=1.0) -> np.ndarray:
    """
    Vypocet Juliovy mnoziny.

    Parametry:
        c_real, c_imag (float): Realna a imaginarni cast komplexniho cisla c.
        x_min, x_max (float): Rozsah realne osy (x).
        y_min, y_max (float): Rozsah imaginarni osy (y).
        n (int): Rozliseni obrazu (pocet bodu na kazde ose).
        k (int): Maximalni pocet iteraci pro kontrolu divergence.
        x, y (float): Posun obrazu v realne a imaginarni ose.
        zoom (float): Faktor priblizeni obrazu (1.0 = bez priblizeni).

    Vracena hodnota:
        ndarray: Matice s hodnotami iteraci do divergence pro kazdy bod v rovine.
    """
    
    scale = 1.0 / zoom
    x_min *= scale
    x_max *= scale
    y_min *= scale
    y_max *= scale
    re = np.linspace(x_min + x, x_max + x, n)
    im = np.linspace(y_min + y, y_max + y, n)
    
    divergence_matrix = np.full((n, n), k, dtype=np.int32)
    
    # Nastaveni komplexniho cisla c (narozdil od Mandelbrotovy mnoziny ma Juliova mnozina pevne danou hodnotu c)
    c = complex(c_real, c_imag)
    
    # Prochazeni vsech bodu v rovine
    for ix in numba.prange(n):
        for iy in range(n):
            z = re[ix] + 1j * im[iy]
            
            # Iterace pro vypocet Juliovy posloupnosti
            for i in range(k):
                z = z * z + c

                # Kontrola divergence, pokud |z| > 2 (bez pouziti numpy, nebot to zpomaluje celou funcki)
                if (z.real * z.real + z.imag * z.imag) > 4.0:
                    divergence_matrix[ix, iy] = i
                    break
                    
    return divergence_matrix

def plot_julia(c_real: float=-0.7, c_imag: float=0.27015, n: int=1000, k: int=200, cmap: str='hot', x: float=0, y: float=0, zoom: float=1.0):
    """Vykresleni Juliovy mnoziny.

    Parametry:
        c_real, c_imag (float): Realna a imaginarni cast komplexniho cisla c.
        n (int): Rozliseni obrazu (pocet bodu na kazde ose).
        k (int): Maximalni pocet iteraci pro kontrolu divergence.
        cmap (str): Nazev barevne mapy pro vykresleni.
        x, y (float): Posun obrazu v realne a imaginarni ose.
        zoom (float): Faktor priblizeni obrazu (1.0 = bez priblizeni).
    """
    data = julia_set(n=n, k=k, c_real=c_real, c_imag=c_imag, x=x, y=y, zoom=zoom)
    plt.figure(figsize=(9, 8))
    plt.imshow(data.T, cmap=cmap, origin='lower', extent=[-2.0, 2.0, -2.0, 2.0])
    plt.colorbar(label='Počet iterací do divergence')
    plt.title(f'Juliova množina (k={k}, n={n})')
    plt.show()

def interactive_julia():
    """
    Vytvoreni interaktivniho widgetu pro Juliovu mnozinu.
    Umozni uzivateli menit parametry jako hodnoty komplexniho cisla c, rozliseni, pocet iteraci, barevnou mapu, posun a priblizeni.
    """
    interact(plot_julia,
            n=IntSlider(min=50, max=2500, step=50, value=1000, description='Rozlišení'),
            k=IntSlider(min=5, max=500, step=5, value=200, description='Iterace'),
            c_real=FloatSlider(min=-2.5, max=2.5, step=0.01, value=-0.7, description='Realná část c'),
            c_imag=FloatSlider(min=-2.5, max=2.5, step=0.01, value=0.27, description='Imag. část c'),
            cmap=ToggleButtons(options=['hot', 'viridis', 'inferno', 'gray', 'turbo'], value='hot', description='Barevná mapa'),
            x=FloatSlider(min=-2.0, max=2.0, step=0.01, value=0.0, description='Posun v x'),
            y=FloatSlider(min=-2.0, max=2.0, step=0.01, value=0.0, description='Posun v y'),
            zoom=FloatSlider(min=1.0, max=50.0, step=0.5, value=1.0, description='Priblížení'));