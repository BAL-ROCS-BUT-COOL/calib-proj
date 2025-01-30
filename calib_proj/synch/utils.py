import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def save_sequences_to_json(filename, sequences):
    with open(filename, 'w') as f:
        json.dump(sequences, f, indent=4)

def load_sequences_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_sequence(sequence):
    """
    Trace la séquence de synchronisation binaire.
    
    :param sequence: Liste de 0 et 1 représentant la séquence de synchronisation.
    """
    plt.figure(figsize=(10, 2))
    plt.plot(sequence, drawstyle='steps-post')
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Synchronization Sequence')
    plt.grid(True)
    plt.show(block=False)


