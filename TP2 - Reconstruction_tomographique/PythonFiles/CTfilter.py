#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np

## filtrer le sinogramme
## ligne par ligne
def filterSinogram(sinogram):
    for i in range(sinogram.shape[0]):
        sinogram[i] = filterLine(sinogram[i])

## filtrer une ligne (projection) via FFT
def filterLine(projection):
    fft_projection = np.fft.rfft(projection)

    # Créer un filtre rampe |u|
    freqs = np.fft.rfftfreq(len(projection)) # fréquences
    filtre_rampe = np.abs(freqs) # filtre rampe
    
    # Appliquer le filtre rampe sur la fft de la projection
    filtered_fft_projection = fft_projection * filtre_rampe
    
    # Retourner au domaine spatial
    filtered_projection = np.fft.irfft(filtered_fft_projection)
    
    return filtered_projection

