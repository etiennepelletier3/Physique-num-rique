#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np
import time

# local files
import geo as geo
import util as util
import CTfilter as CTfilter


## créer l'ensemble de données d'entrée à partir des fichiers
def readInput():
    # lire les angles
    [nbprj, angles] = util.readAngles(geo.dataDir + geo.anglesFile)

    print("nbprj:", nbprj)
    print("angles min and max (rad):")
    print("[" + str(np.min(angles)) + ", " + str(np.max(angles)) + "]")

    # lire le sinogramme
    [nbprj2, nbpix2, sinogram] = util.readSinogram(geo.dataDir + geo.sinogramFile)

    if nbprj != nbprj2:
        print("angles file and sinogram file conflict, aborting!")
        exit(0)

    if geo.nbpix != nbpix2:
        print("geo description and sinogram file conflict, aborting!")
        exit(0)

    return [nbprj, angles, sinogram]


## reconstruire une image TDM en mode rétroprojection
def laminogram():

    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox):  # colonnes de l'image
        print("working on image column: " + str(j + 1) + "/" + str(geo.nbvox))
        for i in range(geo.nbvox):  # lignes de l'image
            for a in range(len(angles)):
                # votre code ici...
                # le défi est simplement géométrique;
                # pour chaque voxel, trouver la position par rapport au centre de la
                # grille de reconstruction et déterminer la position d'arrivée
                # sur le détecteur d'un rayon partant de ce point et atteignant
                # le détecteur avec un angle de 90 degrés. Vous pouvez utiliser
                # le pixel le plus proche ou interpoler linéairement...Rappel, le centre
                # du détecteur est toujours aligné avec le centre de la grille de
                # reconstruction peu importe l'angle.
                # voxel_center = np.array([0, 0])

                voxel_x = geo.nbvox / 2 - i
                voxel_y = j - geo.nbvox / 2

                vec_voxel_ij = np.array([voxel_x, voxel_y]) * geo.voxsize

                vec_proj = np.array([np.cos(angles[a]), np.sin(angles[a])])

                t = np.dot(
                    vec_proj, vec_voxel_ij
                )  # distance t du centre du détecteur [cm]

                pix_detect = int(
                    round(t / geo.pixsize)
                )  # position du pixel sur le détecteur

                pix_sin = int(
                    pix_detect + geo.nbpix / 2
                )  # position du pixel dans le sinogramme

                pix_val = sinogram[a][pix_sin]  # valeur du pixel dans le sinogramme

                image[j, i] += pix_val

    util.saveImage(image, "CT")


## reconstruire une image TDM en mode retroprojection filtrée
def backproject():

    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    ### option filtrer ###
    CTfilter.filterSinogram(sinogram)

    import matplotlib.pyplot as plt

    # Plot the filtered sinogram
    plt.imshow(sinogram, cmap="gray", aspect="auto")
    plt.title("Sinogram filtré")
    plt.show()
    ######

    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox):  # colonnes de l'image
        print("working on image column: " + str(j + 1) + "/" + str(geo.nbvox))
        for i in range(geo.nbvox):  # lignes de l'image
            for a in range(len(angles)):
                voxel_x = geo.nbvox / 2 - i
                voxel_y = j - geo.nbvox / 2

                vec_voxel_ij = np.array([voxel_x, voxel_y]) * geo.voxsize

                vec_proj = np.array([np.cos(angles[a]), np.sin(angles[a])])

                t = np.dot(
                    vec_proj, vec_voxel_ij
                )  # distance t du centre du détecteur [cm]

                pix_detect = int(
                    round(t / geo.pixsize)
                )  # position du pixel sur le détecteur

                pix_sin = int(
                    pix_detect + geo.nbpix / 2
                )  # position du pixel dans le sinogramme

                pix_val = sinogram[a][pix_sin]  # valeur du pixel dans le sinogramme

                image[j, i] += pix_val
            #    pas mal la même chose que prédédemment
            # mais avec un sinogramme qui aura été préalablement filtré

    util.saveImage(image, "fbp")


## reconstruire une image TDM en mode retroprojection
def reconFourierSlice():

    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite, complexe
    # pour qu'elle puisse contenir sa version FFT d'abord
    IMAGE = np.zeros((geo.nbvox, geo.nbvox), "complex")

    # conteneur pour la FFT du sinogramme
    SINOGRAM = np.zeros(sinogram.shape, dtype=complex)

    # image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    for a in range(nbprj):
        # fft sur l'axe du détecteur avec recentrage
        SINOGRAM[a, :] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sinogram[a, :])))

    # Calculer l'axe fréquentiel pour les projections (utiliser d=geo.pixsize)
    freq1d = np.fft.fftshift(np.fft.fftfreq(geo.nbpix, d=geo.pixsize))
    pos_mask = freq1d >= 0
    freq1d_pos = freq1d[pos_mask]

    N = geo.nbvox
    freq2d = np.fft.fftshift(np.fft.fftfreq(N, d=geo.voxsize))
    U, V = np.meshgrid(freq2d, freq2d)

    # Module et angle (en radians) pour chaque point de la grille
    R = np.sqrt(U**2 + V**2)
    phi = np.arctan2(V, U)
    # Ramener les angles dans l'intervalle [0, 2*pi]
    phi_mod = np.mod(phi, 2 * np.pi)

    # Pour chaque point de la grille 2D, remplir IMAGE par interpolation dans la TF 1D
    for i in range(N):
        for j in range(N):
            r = R[i, j]
            # Si la fréquence dépasse la borne max disponible, on met 0
            if r > freq1d_pos[-1]:
                IMAGE[i, j] = 0
            else:
                # Trouver l'angle le plus proche
                diff = np.abs(angles - phi_mod[i, j])
                a_idx = np.argmin(diff)
                # Interpoler la TF 1D pour obtenir la valeur de la projection
                FT_proj = SINOGRAM[a_idx, :][pos_mask]
                real_val = np.interp(r, freq1d_pos, np.real(FT_proj))
                imag_val = np.interp(r, freq1d_pos, np.imag(FT_proj))
                IMAGE[i, j] = real_val + 1j * imag_val

    # Reconstituer l'image par TF−1 2D :
    image = np.fft.ifft2(np.fft.ifftshift(IMAGE))
    image = np.fft.fftshift(image)
    image = np.real(image)
    image = np.flip(image, axis=1)

    # Sauvegarder l'image reconstruite
    util.saveImage(image, "fft")


## main ##
start_time = time.time()
# laminogram()
# backproject()
reconFourierSlice()
print("--- %s seconds ---" % (time.time() - start_time))
