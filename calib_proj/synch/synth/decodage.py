import numpy as np
import matplotlib.pyplot as plt

def generate_sync_sequence(length):
    """
    Génère une séquence de synchronisation binaire de longueur spécifiée.
    """
    return np.random.choice([0, 1], size=length)

def apply_affine_transformation(signal, alpha, beta):
    """
    Applique une transformation affine au signal.
    """
    return alpha * signal + beta

def add_sync_to_signal(signal, sync_sequence, insert_position):
    """
    Insère la séquence de synchronisation dans le signal à la position spécifiée.
    """
    signal_with_sync = signal.copy()
    signal_with_sync[insert_position:insert_position + len(sync_sequence)] = sync_sequence
    return signal_with_sync


# def detect_sync_sequence(received_signal, sync_sequence, threshold=0.5):
#     """
#     Détecte la position de la séquence de synchronisation dans le signal reçu en trouvant l'index avec la meilleure corrélation,
#     et retourne cet index uniquement si la meilleure corrélation dépasse le seuil spécifié.

#     :param received_signal: Tableau des intensités reçues [I1, I2, ..., In].
#     :param sync_sequence: Tableau des intensités de la séquence de synchronisation [S1, S2, ..., Sm].
#     :param threshold: Seuil de corrélation pour la détection (par défaut 0.8).
#     :return: Index de début de la séquence de synchronisation si la meilleure corrélation dépasse le seuil, sinon -1.
#     """
#     sync_length = len(sync_sequence)
#     normalized_sync = (sync_sequence - np.mean(sync_sequence)) / np.std(sync_sequence)

#     best_correlation = -np.inf
#     best_index = -1


#     for i in range(len(received_signal) - sync_length + 1):
#         segment = received_signal[i:i + sync_length]
#         normalized_segment = (segment - np.mean(segment)) / np.std(segment)

#         correlation = np.correlate(normalized_sync, normalized_segment)[0] / sync_length


#         if correlation > best_correlation:
#             best_correlation = correlation
#             best_index = i

#     if best_correlation > threshold:
#         return best_index
#     else:
#         return -1


def detect_sync_sequence(received_signal, sync_sequence, threshold=0.5):
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
    best_index = -1
    best_correlation = -np.inf
    for index, corr in correlations:
        if corr > best_correlation and corr > threshold:
            best_correlation = corr
            best_index = index

    # Tracé des corrélations
    indices, corr_values = zip(*correlations)
    plt.figure(figsize=(10, 5))
    plt.plot(indices, corr_values, label='Corrélation')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Seuil')
    if best_index != -1:
        plt.axvline(x=best_index, color='g', linestyle='--', label='Meilleure correspondance')
    plt.xlabel('Index')
    plt.ylabel('Valeur de la corrélation')
    plt.title('Corrélation entre la séquence de synchronisation et le signal reçu')
    plt.legend()
    plt.show()

    return best_index

    

# Paramètres
# seq_duration = 60 * 60 * 30
signal_length = 60 * 60 * 30
print(signal_length)
sync_length = 30
insert_position = 100000
alpha = 0.1
beta = 1
threshold = 0.8

# Génération de la séquence de synchronisation
sync_sequence = generate_sync_sequence(sync_length)

# Génération du signal de bruit
np.random.seed(42)  # Pour la reproductibilité
noise_signal = np.random.normal(0, 1, signal_length)
# Normalisation du signal de bruit entre 0 et 1
min_val = np.min(noise_signal)
max_val = np.max(noise_signal)
noise_signal = (noise_signal - min_val) / (max_val - min_val)

# Insertion de la séquence de synchronisation dans le signal de bruit
transformed_sync_sequence = apply_affine_transformation(sync_sequence, alpha, beta)
received_signal = add_sync_to_signal(noise_signal, transformed_sync_sequence, insert_position)

# Détection de la séquence de synchronisation
detected_index = detect_sync_sequence(received_signal, sync_sequence, threshold)

# Affichage des résultats
plt.figure(figsize=(15, 6))
plt.plot(received_signal, label='Signal Reçu')
plt.axvline(x=insert_position, color='g', linestyle='--', label='Position Réelle de la Sync')
if detected_index != -1:
    plt.axvline(x=detected_index, color='r', linestyle='--', label='Position Détectée de la Sync')
plt.legend()
plt.title('Détection de la Séquence de Synchronisation dans le Signal Reçu')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.show()
