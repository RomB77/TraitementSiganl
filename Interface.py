##Librairies
import tkinter as tk
from tkinter import filedialog, Toplevel,ttk
import pygame
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from midiutil import MIDIFile
from collections import Counter
import pandas as pd
import os
import random
from pydub import AudioSegment
from pydub.playback import play
import tempfile

## Dictionnaire des instruments
instruments= {
    'Piano': 0,
    'Piano Clair': 1,
    'Piano Électrique': 2,
    'Piano Honky-tonk': 3,
    'Clavier Électrique 1': 4,
    'Clavier Électrique 2': 5,
    'Clavecin': 6,
    'Clavinet': 7,
    'Célesta': 8,
    'Carillon': 9,
    'Boîte à Musique': 10,
    'Vibraphone': 11,
    'Marimba': 12,
    'Xylophone': 13,
    'Cloches Tubulaires': 14,
    'Dulcimer': 15,
    'Orgue Classique': 16,
    'Orgue Percussif': 17,
    'Orgue Rock': 18,
    'Orgue d’Église': 19,
    'Harmonium': 20,
    'Accordéon': 21,
    'Harmonica': 22,
    'Accordéon Tango': 23,
    'Guitare Classique': 24,
    'Guitare Acoustique': 25,
    'Guitare Jazz': 26,
    'Guitare Clean': 27,
    'Guitare Mutée': 28,
    'Guitare Saturée': 29,
    'Guitare Distorsion': 30,
    'Harmoniques Guitare': 31,
    'Contrebasse Acoustique': 32,
    'Basse Électrique (Doigt)': 33,
    'Basse Électrique (Médiator)': 34,
    'Basse Fretless': 35,
    'Basse Slap 1': 36,
    'Basse Slap 2': 37,
    'Basse Synth 1': 38,
    'Basse Synth 2': 39,
    'Violons': 40,
    'Alto': 41,
    'Violoncelle': 42,
    'Contrebasse': 43,
    'Cordes Tremolo': 44,
    'Cordes Pizzicato': 45,
    'Harpe': 46,
    'Timbales': 47,
    'Ensemble à Cordes 1': 48,
    'Ensemble à Cordes 2': 49,
    'Cordes Synth 1': 50,
    'Cordes Synth 2': 51,
    'Chœur Aahs': 52,
    'Chœur Oohs': 53,
    'Chœur Synthétique': 54,
    'Coup d’Orchestre': 55,
    'Trompette': 56,
    'Trombone': 57,
    'Tuba': 58,
    'Trompette Sourdine': 59,
    'Cor': 60,
    'Section Cuivres': 61,
    'Cuivres Synth 1': 62,
    'Cuivres Synth 2': 63,
    'Saxophone Soprano': 64,
    'Saxophone Alto': 65,
    'Saxophone Ténor': 66,
    'Saxophone Baryton': 67,
    'Hautbois': 68,
    'Cor Anglais': 69,
    'Basson': 70,
    'Clarinette': 71,
    'Piccolo': 72,
    'Flûte': 73,
    'Flûte à Bec': 74,
    'Flûte de Pan': 75,
    'Bouteille Soufflée': 76,
    'Shakuhachi': 77,
    'Sifflet': 78,
    'Ocarina': 79,
    'Lead (Carré)': 80,
    'Lead (Dent de Scie)': 81,
    'Lead (Calliope)': 82,
    'Lead (Chiff)': 83,
    'Lead (Charang)': 84,
    'Lead (Voix)': 85,
    'Lead (Quintes)': 86,
    'Lead (Basse + Lead)': 87,
    'Pad (New Age)': 88,
    'Pad (Chaleureux)': 89,
    'Pad (Polysynth)': 90,
    'Pad (Chœur)': 91,
    'Pad (Archet)': 92,
    'Pad (Métallique)': 93,
    'Pad (Halo)': 94,
    'Pad (Balayage)': 95,
    'FX (Pluie)': 96,
    'FX (Bande Son)': 97,
    'FX (Cristal)': 98,
    'FX (Atmosphère)': 99,
    'FX (Éclat)': 100,
    'FX (Gobelins)': 101,
    'FX (Échos)': 102,
    'FX (Science-Fiction)': 103,
    'Sitar': 104,
    'Banjo': 105,
    'Shamisen': 106,
    'Koto': 107,
    'Kalimba': 108,
    'Cornemuse': 109,
    'Violon Fiddle': 110,
    'Shanai': 111,
    'Clochettes': 112,
    'Agogo': 113,
    'Tambours Métalliques': 114,
    'Blocs de Bois': 115,
    'Tambour Taiko': 116,
    'Tom Mélodique': 117,
    'Tambour Synthétique': 118,
    'Cymbale Inversée': 119,
    'Bruit de Guitare': 120,
    'Bruit de Souffle': 121,
    'Bruit de Vagues': 122,
    'Chant d’Oiseaux': 123,
    'Sonnerie Téléphone': 124,
    'Hélicoptère': 125,
    'Applaudissements': 126,
    'Coup de Feu': 127
}

## Fenêtre de résultat

def result(audio,instru):
    y, sr = librosa.load(audio)

    # Adapter les paramètres en fonction du tempo
    nfft = 2048 # Si le tempo est rapide (arbitrairement au-dessus de 120 BPM) on réduire la taille de la fenêtre pour une meilleure résolution temporelle
    hop_length = int(nfft * 0.5)

    # Calcul du spectrogramme
    D = librosa.stft(y, n_fft=nfft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Affichage du spectrogramme
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogramme')
    #plt.show()

    # Détection des débuts (onsets) en temps
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=512, units='frames')
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # Ajouter un point final pour couvrir le dernier segment
    onset_times = np.append(onset_times, librosa.get_duration(y=y, sr=sr))

    # Liste pour stocker les notes détectées
    detected_notes = []

    def note_to_midi(note_name):
        # Liste des noms des notes
        note_base = {'C': 0, 'C♯': 1, 'D': 2, 'D♯': 3, 'E': 4, 'F': 5, 'F♯': 6,
                    'G': 7, 'G♯': 8, 'A': 9, 'A♯': 10, 'B': 11}

        # Extraction de la note et de l'octave
        note = note_name[:-1]  # Partie lettre (e.g., 'C', 'D#')
        octave = int(note_name[-1])  # Partie chiffre (e.g., 4)

        # Calcul du numéro MIDI
        midi_number = 12 * (octave + 1) + note_base[note]
        return midi_number

    def octave(note_prec, note_suiv) :
        if len(note_prec) == 2 and len(note_suiv) == 2 :
            if note_prec[0] == note_suiv[0] :
                return False
        if len(note_prec) == 3 and len(note_suiv) == 3 :
            if note_prec[:2] == note_suiv[:2] :
                return False
        else :
            return True

    # Diviser et analyser chaque segment
    for i in range(len(onset_times) - 1):
        start_time = onset_times[i]
        end_time = onset_times[i + 1]

        # Convertir le temps en échantillons
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        duree_time = end_time - start_time

        # Extraire le segment audio
        segment = y[start_sample:end_sample]

        # Détection de la fréquence dominante avec librosa.piptrack
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
        notes_in_segment = []

        if pitches.any():
            for t in range(pitches.shape[1]):  # Parcourir chaque frame
                index = magnitudes[:, t].argmax()  # Index de la magnitude maximale
                pitch_candidate = pitches[index, t]
                if pitch_candidate > 0:  # Vérifier que la fréquence est valide
                    note = librosa.hz_to_note(pitch_candidate)
                    notes_in_segment.append(note)

        # Trouver la note la plus fréquente dans ce segment
        if notes_in_segment :
            most_common_note = Counter(notes_in_segment).most_common(1)[0][0]
            if (not detected_notes) or octave(detected_notes[-1][0],most_common_note) :
                midi_note = note_to_midi(most_common_note)
                detected_notes.append((most_common_note, start_time, end_time,pitch_candidate,duree_time,midi_note))

    """
    #print(detected_notes)
    for note, start, end, freq in detected_notes:
        print(f"Note: {note}, Fréquence : {freq}, Start: {start:.2f}s, End: {end:.2f}s")
    """

    # Sauvegarder le fichier MIDI
    track = 0
    channel = 0
    volume = 100  # 0-127, selon la norme MIDI

    # Création du fichier MIDI
    midi = MIDIFile(1)
    program = instru
    midi.addProgramChange(track, channel, 0, program)

    # Ajout des notes au fichier MIDI
    for i, (note_name, start,end, freq,duree,midi_note) in enumerate(detected_notes): # frame = duree
        if i < len(detected_notes) - 1 :
            duree = detected_notes[i+1][1] - detected_notes[i][1]
        midi.addNote(track, channel, midi_note, start*2, duree*2, 100)  # Canal 0, vélocité 100

    # Écriture du fichier MIDI
    new_audio=audio.split('.')[0]
    with open(f"{new_audio}.mid", "wb") as output_file:
        midi.writeFile(output_file)


# Fonction pour ouvrir une nouvelle fenêtrequi joue le résultat
def changer_instrument():
    instrument = instrument_combobox.get()

    if instrument in instruments:
        try:
            result(audio_choisi,instruments[instrument])
            new_audio=audio_choisi.split('.')[0]+'.mid'

            # Charger le fichier audio (mais ne pas le jouer immédiatement)
            pygame.mixer.music.load(new_audio)

            # Ouvrir une nouvelle fenêtre
            nouvelle_fenetre = Toplevel(fenetre)
            nouvelle_fenetre.title(f"Changer pour l'instrument : {instrument}")
            nouvelle_fenetre.geometry("600x300")  # Définir la taille de la fenêtre principale

            # Gestionnaire pour quitter proprement l'application
            def quitter_application():
                pygame.mixer.music.stop()  # Arrêter la lecture en cours (si nécessaire)
                pygame.mixer.quit()        # Libérer les ressources de pygame
                nouvelle_fenetre.destroy()          # Fermer la fenêtre principale

            # Associer l'événement de fermeture de la fenêtre au gestionnaire
            nouvelle_fenetre.protocol("WM_DELETE_WINDOW", quitter_application)

            # Titre et information dans la fenêtre
            tk.Label(nouvelle_fenetre, text=f"{nom_audio} joué par un/une {instrument}").pack(pady=10)

            # Ajouter les boutons de contrôle audio
            tk.Button(nouvelle_fenetre, text="Jouer", command=lambda: pygame.mixer.music.play()).pack(pady=5)
            tk.Button(nouvelle_fenetre, text="Pause", command=lambda: pygame.mixer.music.pause()).pack(pady=5)
            tk.Button(nouvelle_fenetre, text="Reprendre", command=lambda: pygame.mixer.music.unpause()).pack(pady=5)
            tk.Button(nouvelle_fenetre, text="Arrêter", command=lambda: pygame.mixer.music.stop()).pack(pady=5)

        except pygame.error as e:
            label_instrument_erreur.config(text=f"Erreur audio : {e}")
    else:
        label_instrument_erreur.config(text="Veuillez entrer un nom d'instrument !")


##Fenêtre initiale
# Initialisation de pygame pour la gestion de l'audio
pygame.mixer.init()

# Fonction pour sélectionner un fichier audio
audio_choisi = None

def choisir_fichier():
    global audio_choisi  # Déclarer la variable globale
    global nom_audio
    audio_choisi = filedialog.askopenfilename(
        title="Choisir un fichier audio",
        filetypes=[("Fichiers audio", " *.wav")]
    )
    sep=audio_choisi.split('/')
    nom_audio=sep[len(sep)-1]
    if audio_choisi:
        label_fichier.config(text=f"Audio sélectionné : {nom_audio}")
        try:
            pygame.mixer.music.load(audio_choisi)
        except pygame.error as e:
            label_fichier.config(text=f"Erreur de chargement : {e}")
    else:
        label_fichier.config(text="Aucun fichier sélectionné")


def fichier_aleat():
    global audio_choisi  # Variable pour stocker le fichier sélectionné
    global nom_audio
    repertoire = filedialog.askdirectory(title="Choisir un répertoire")  # Sélection du répertoire
    if repertoire:
        # Récupérer tous les fichiers audio (.mp3, .wav, .mid) dans le répertoire
        fichiers_audio = [f for f in os.listdir(repertoire) if f.endswith(('.wav'))]
        if fichiers_audio:
            audio_choisi = os.path.join(repertoire, random.choice(fichiers_audio))  # Fichier aléatoire
            sep=audio_choisi.split('\\')
            nom_audio=sep[len(sep)-1]
            label_fichier.config(text=f"Fichier sélectionné : {nom_audio}")
            try:
                pygame.mixer.music.load(audio_choisi)  # Charger le fichier audio
            except pygame.error as e:
                label_fichier.config(text=f"Erreur de chargement : {e}")
        else:
            label_fichier.config(text="Aucun fichier audio trouvé dans ce répertoire.")
    else:
        label_fichier.config(text="Aucun répertoire sélectionné.")


# Fonction pour jouer le fichier audio
def jouer_audio():
    try:
        pygame.mixer.music.play()
    except pygame.error as e:
        label_fichier.config(text=f"Erreur : {e}")

# Fonction pour mettre en pause la lecture
def pause_audio():
    try:
        pygame.mixer.music.pause()
    except pygame.error as e:
        label_fichier.config(text=f"Erreur : {e}")

# Fonction pour reprendre la lecture
def reprendre_audio():
    try:
        pygame.mixer.music.unpause()
    except pygame.error as e:
        label_fichier.config(text=f"Erreur : {e}")

# Fonction pour arrêter la lecture
def arreter_audio():
    try:
        pygame.mixer.music.stop()
    except pygame.error as e:
        label_fichier.config(text=f"Erreur : {e}")

# Création de la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Lecteur Audio")
fenetre.geometry("600x500")  # Définir la taille de la fenêtre principale

# Gestionnaire pour quitter proprement l'application
def quitter_application():
    fenetre.destroy()          # Fermer la fenêtre principale
    exit()

# Associer l'événement de fermeture de la fenêtre au gestionnaire
fenetre.protocol("WM_DELETE_WINDOW", quitter_application)


# Bouton pour sélectionner un fichier
bouton_choisir = tk.Button(fenetre, text="Choisir un fichier audio", command=choisir_fichier)
bouton_choisir.pack(pady=10)

# Label pour afficher le nom du fichier sélectionné
label_fichier = tk.Label(fenetre, text="ou", wraplength=400)
label_fichier.pack(pady=10)

# Bouton pour sélectionner un fichier
bouton_choisir = tk.Button(fenetre, text="Sélectionner un audio aléatoire", command=fichier_aleat)
bouton_choisir.pack(pady=10)

# Label pour afficher le nom du fichier sélectionné
label_fichier = tk.Label(fenetre, text="Aucun fichier sélectionné", wraplength=400)
label_fichier.pack(pady=10)

# Boutons de contrôle audio
bouton_jouer = tk.Button(fenetre, text="Jouer", command=jouer_audio)
bouton_jouer.pack(pady=10)

bouton_pause = tk.Button(fenetre, text="Pause", command=pause_audio)
bouton_pause.pack(pady=5)

bouton_reprendre = tk.Button(fenetre, text="Reprendre", command=reprendre_audio)
bouton_reprendre.pack(pady=5)

bouton_arreter = tk.Button(fenetre, text="Arrêter", command=arreter_audio)
bouton_arreter.pack(pady=5)

# Texte déplacé sous les boutons de contrôle
label_instrument = tk.Label(fenetre, text="Choisir l'instrument pour changer le son :")
label_instrument.pack(padx=5, pady=10)  # Ajoute un espace avec pady

# Champ de texte pour entrer un instrument

instruments_list = list(instruments.keys())
instrument_combobox = tk.ttk.Combobox(fenetre, values=instruments_list, state="readonly")
instrument_combobox.pack(padx=10, pady=5)

# Définir une valeur par défaut pour la combobox
instrument_combobox.set("Piano")

label_instrument_erreur = tk.Label(fenetre, text="", fg="red")
label_instrument_erreur.pack()

# Bouton pour ouvrir une nouvelle fenêtre
bouton_changer = tk.Button(fenetre, text="Changer", command=changer_instrument)
bouton_changer.pack(pady=10)

# Boucle principale de l'interface graphique
fenetre.mainloop()




