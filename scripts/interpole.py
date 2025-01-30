import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Supposons que ref_signal et meas_signal soient vos signaux d'origine
# avec des fréquences d'échantillonnage de 30 Hz et 25 Hz respectivement.

# Exemple de signaux (à remplacer par vos données réelles)
ref_signal = np.sin(2 * np.pi * np.linspace(0, 1, 30))  # Signal de référence
meas_signal = np.sin(2 * np.pi * np.linspace(0, 1, 30))  # Signal mesuré

# Définissez les fréquences d'échantillonnage
fs_ref = 30  # Fréquence d'échantillonnage du signal de référence
fs_meas = 30  # Fréquence d'échantillonnage du signal mesuré

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

# Calcul de la corrélation croisée
corr = signal.correlate(ref_signal_upsampled, meas_signal_upsampled, mode='full')
lags = signal.correlation_lags(len(ref_signal_upsampled), len(meas_signal_upsampled), mode='full')
lag = lags[np.argmax(corr)]
time_lag = lag / fs_common

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

# Corrélation croisée
plt.subplot(4, 1, 4)
plt.plot(lags / fs_common, corr)
plt.title('Corrélation Croisée')
plt.xlabel('Décalage Temporel (s)')
plt.ylabel('Amplitude')
plt.axvline(x=time_lag, color='r', linestyle='--', label=f'Décalage = {time_lag:.2f} s')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Le décalage temporel estimé entre les deux signaux est de {time_lag:.2f} secondes.")
