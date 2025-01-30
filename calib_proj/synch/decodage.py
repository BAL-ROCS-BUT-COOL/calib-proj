import numpy as np
import matplotlib.pyplot as plt
import cv2



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

    correlations = []

    for i in range(len(received_signal) - sync_length + 1):
        segment = received_signal[i:i + sync_length]
        normalized_segment = (segment - np.mean(segment)) / np.std(segment)

        correlation = np.correlate(normalized_sync, normalized_segment)[0] / sync_length
        correlations.append((i, correlation))

    # Recherche de la meilleure corrélation positive
    best_index = None
    best_correlation = -np.inf
    for index, corr in correlations:
        if corr > best_correlation and corr > threshold:
            best_correlation = corr
            best_index = index

    if plot:
        # Tracé des corrélations
        indices, corr_values = zip(*correlations)
        plt.figure(figsize=(10, 5))
        plt.plot(indices, corr_values, label='Corrélation')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Seuil')
        if best_index is not None:
            plt.axvline(x=best_index, color='g', linestyle='--', label='Meilleure correspondance')
        plt.xlabel('Index')
        plt.ylabel('Valeur de la corrélation')
        plt.title('Corrélation entre la séquence de synchronisation et le signal reçu')
        plt.legend()
        plt.show(block=False)

    return best_index

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



