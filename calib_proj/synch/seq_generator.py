import cv2
import numpy as np
import json


from calib_proj.synch.utils import plot_sequence, save_sequences_to_json, load_sequences_from_json

def generate_synch_sequence_video(output_filename, resolution, synch_sequence_encoding, proj_fps=30):
    """
    Génère une vidéo à partir d'une séquence de synchronisation encodée de 0 et 1.
    
    :param output_filename: Nom du fichier de sortie vidéo (avec extension .avi ou .mp4).
    :param resolution: Tuple (largeur, hauteur) de la résolution de chaque image.
    :param synch_sequence_encoding: Liste de 0 et 1 représentant la séquence de synchronisation.
    :param frame_rate: Nombre d'images par seconde (par défaut 10).
    """
    width, height = resolution



    # Initialisation du writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec vidéo
    video_writer = cv2.VideoWriter(output_filename, fourcc, proj_fps, (width, height), isColor=False)

    for value in synch_sequence_encoding:
        # Générer une image blanche (255) ou noire (0) en fonction de la valeur encodée
        intensity = 255 if value == 1 else 0
        image_array = np.full((height, width), intensity, dtype=np.uint8)

        
        # Ajouter le frame à la vidéo
        video_writer.write(image_array)

    # Libérer les ressources
    video_writer.release()
    print(f"Vidéo générée : {output_filename}")


def generate_sync_sequence(length):
    """
    Génère une séquence de synchronisation binaire de longueur spécifiée.
    """
    return np.random.choice([0, 1], size=length)


def generate_hamming_sequences(length=30):
    # Génère la première séquence aléatoire
    seq_start = generate_sync_sequence(length)
    
    # Génère la deuxième séquence en inversant les bits de manière aléatoire
    seq_end = 1 - seq_start  # Maximiser la différence initialement

    return seq_start, seq_end



def generate_sequences(seq_duration, seq_fps, proj_fps): 
    np.random.seed(0)

    seq_length = seq_fps * seq_duration  
    seq_start, seq_end = generate_hamming_sequences(seq_length)
    repeat_factor = proj_fps // seq_fps 
    seq_start = np.repeat(seq_start, repeat_factor)
    seq_end = np.repeat(seq_end, repeat_factor)
    sequences = {'start': seq_start.tolist(), 'end': seq_end.tolist()}

    return sequences


# # Exemple d'utilisation
# json_filename = "synch_sequence.json"
# save_sequences_to_json(json_filename, sequences)
# sequences = load_sequences_from_json(json_filename)

# generate_synch_sequence_video(output_filename, resolution, synch_sequence_encoding, proj_fps)

# output_filename
