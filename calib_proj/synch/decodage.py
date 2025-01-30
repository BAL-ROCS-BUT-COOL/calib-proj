import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal




def detect_sync_sequence(received_signal, sync_sequence, seq_fps, cam_fps, threshold=0.5, plot=True):
    """
    Détecte la position de la séquence de synchronisation dans le signal reçu en trouvant l'index avec la meilleure corrélation positive,
    et retourne cet index uniquement si la meilleure corrélation dépasse le seuil spécifié. Enregistre et trace toutes les corrélations calculées.

    :param received_signal: Tableau des intensités reçues [I1, I2, ..., In].
    :param sync_sequence: Tableau des intensités de la séquence de synchronisation [S1, S2, ..., Sm].
    :param threshold: Seuil de corrélation pour la détection (par défaut 0.5).
    :return: Index de début de la séquence de synchronisation si la meilleure corrélation positive dépasse le seuil, sinon -1.
    """
    sync_length = len(sync_sequence)
    normalized_sync = (sync_sequence - np.mean(sync_sequence)) / np.std(sync_sequence)

    fps_ratio = cam_fps / seq_fps
    mes_length = int(sync_length * fps_ratio)

    correlations = []
    ref_signal = normalized_sync

    for i in range(len(received_signal) - mes_length + 1):

        methode = 1

        # substraction of the mean and normalization of the signal -> so that the correlation gives the ZNCC
        meas_signal = received_signal[i:i + mes_length]
        meas_signal = (meas_signal - np.mean(meas_signal)) / np.std(meas_signal)

        fs_ref = round(seq_fps)  # Fréquence d'échantillonnage de la séquence de synchronisation
        fs_meas = round(cam_fps)  # Fréquence d'échantillonnage du signal reçu
        if methode == 0:

            # Trouvez la fréquence d'échantillonnage commune (par exemple, le PPCM)
            fs_common = np.lcm(fs_ref, fs_meas)

            # Facteurs de suréchantillonnage
            up_ref = fs_common // fs_ref
            up_meas = fs_common // fs_meas

            # Suréchantillonnage des signaux
            ref_signal_upsampled = signal.resample_poly(ref_signal, up_ref, 1)
            meas_signal_upsampled = signal.resample_poly(meas_signal, up_meas, 1)

            # Temps pour les signaux suréchantillonnés
            t_ref_upsampled = np.arange(len(ref_signal_upsampled)) / fs_common
            t_meas_upsampled = np.arange(len(meas_signal_upsampled)) / fs_common

            corr = signal.correlate(ref_signal_upsampled, meas_signal_upsampled, mode='full')  
            lags = signal.correlation_lags(len(ref_signal_upsampled), len(meas_signal_upsampled), mode='full')
            lag = lags[np.argmax(corr)]
            time_lag = lag / fs_common
            correlation = np.max(corr) 
        
        elif methode == 1:
            from scipy import interpolate, signal

            
            # substraction of the mean and normalization of the signal -> so that the correlation gives the ZNCC
            meas_signal = received_signal[i:i + mes_length]
            meas_signal = (meas_signal - np.mean(meas_signal)) / np.std(meas_signal)

           
            # Création des axes temporels correspondants
            t_ref = np.arange(len(ref_signal)) / fs_ref
            t_mesure = np.arange(len(meas_signal)) / fs_meas

            # Définir une fréquence d'échantillonnage commune élevée pour l'interpolation
            freq_commune = max(fs_ref, fs_meas) * 100  # Par exemple, 10 fois la plus haute fréquence d'échantillonnage

            t_min = min(t_ref[0], t_mesure[0])
            t_max = max(t_ref[-1], t_mesure[-1])
            t_commune = np.arange(t_min, t_max, 1 / freq_commune)

            # Interpolation des deux signaux vers la grille temporelle commune
            interpolateur_ref = interpolate.interp1d(t_ref, ref_signal, kind='linear', fill_value="extrapolate")
            interpolateur_mesure = interpolate.interp1d(t_mesure, meas_signal, kind='linear', fill_value="extrapolate")

            signal_ref_interp = interpolateur_ref(t_commune)
            signal_mesure_interp = interpolateur_mesure(t_commune)

            # Calcul de la corrélation croisée entre les deux signaux interpolés
            corr = signal.correlate(signal_ref_interp, signal_mesure_interp, mode='full')
            correlation = np.max(corr) 
            lags = signal.correlation_lags(len(signal_ref_interp), len(signal_mesure_interp), mode='full')

            # Identification du décalage temporel correspondant au maximum de la corrélation
            decalage_echantillons = lags[np.argmax(corr)]
            time_lag = decalage_echantillons / freq_commune

        correlations.append((i, correlation, time_lag))

    # Recherche de la meilleure corrélation positive
    best_index = None
    best_correlation = -np.inf
    best_lag = None
    for index, corr, lag in correlations:
        if corr > best_correlation and corr > threshold:
            best_correlation = corr
            best_index = index
            best_lag = lag

    if 0:
        i = best_index
        meas_signal = received_signal[i:i + mes_length]
        meas_signal = (meas_signal - np.mean(meas_signal)) / np.std(meas_signal)

        fs_ref = round(seq_fps)  # Fréquence d'échantillonnage de la séquence de synchronisation
        fs_meas = round(cam_fps)  # Fréquence d'échantillonnage du signal reçu

        # Trouvez la fréquence d'échantillonnage commune (par exemple, le PPCM)
        fs_common = np.lcm(fs_ref, fs_meas)

        # Facteurs de suréchantillonnage
        up_ref = fs_common // fs_ref
        up_meas = fs_common // fs_meas

        # Suréchantillonnage des signaux
        ref_signal_upsampled = signal.resample_poly(ref_signal, up_ref, 1)
        meas_signal_upsampled = signal.resample_poly(meas_signal, up_meas, 1)

        # Temps pour les signaux suréchantillonnés
        t_ref_upsampled = np.arange(len(ref_signal_upsampled)) / fs_common
        t_meas_upsampled = np.arange(len(meas_signal_upsampled)) / fs_common

        corr = signal.correlate(ref_signal_upsampled, meas_signal_upsampled, mode='full')
        correlation = np.correlate(ref_signal_upsampled, meas_signal_upsampled)[0] / sync_length


        
        # Tracé des signaux
        plt.figure(figsize=(12, 8))

        # Signal de référence original
        plt.subplot(4, 1, 1)
        plt.stem(np.arange(len(ref_signal)) / fs_ref, ref_signal, basefmt=" ")
        plt.title('Signal de Référence Original')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')

        # Signal mesuré original
        plt.subplot(4, 1, 2)
        plt.stem(np.arange(len(meas_signal)) / fs_meas, meas_signal, basefmt=" ")
        plt.title('Signal Mesuré Original')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')

        # Signaux suréchantillonnés
        plt.subplot(4, 1, 3)
        plt.plot(t_ref_upsampled, ref_signal_upsampled, label='Signal de Référence Suréchantillonné')
        plt.plot(t_meas_upsampled, meas_signal_upsampled, label='Signal Mesuré Suréchantillonné', linestyle='dashed')
        plt.title('Signaux Suréchantillonnés')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    index_refined = best_index - best_lag * cam_fps

    return index_refined, mes_length

def process_video_to_gray_mean(input_video_filename):
    """
    Charge une vidéo enregistrée, calcule la moyenne des valeurs de pixels en nuances de gris
    pour chaque frame et renvoie un tableau numpy contenant des valeurs normalisées entre 0 et 1.

    :param input_video_filename: Chemin du fichier vidéo à traiter.
    :return: Tableau numpy contenant les moyennes normalisées entre 0 et 1.
    """
    # Initialiser la capture vidéo
    cap = cv2.VideoCapture(input_video_filename)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {input_video_filename}")

    mean_gray_values = []

    # Parcourir chaque frame de la vidéo
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        # Convertir en nuances de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculer la moyenne des valeurs de pixels
        mean_value = np.mean(gray_frame)
        mean_gray_values.append(mean_value)

    # Normaliser les valeurs entre 0 et 1
    mean_gray_values = np.array(mean_gray_values)
    normalized_values = (mean_gray_values - np.min(mean_gray_values)) / (np.max(mean_gray_values) - np.min(mean_gray_values))

    # Libérer les ressources
    cap.release()

    return normalized_values



