import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import noisereduce as nr
from midiutil import MIDIFile             #http://midiutil.readthedocs.io/en/1.2.1/
import IPython.display as ipd
from collections import Counter

# Parameters
nb_note = 7 # Nombre de notes pour lesquelles on veut les fréquences

# Load Audio
filename = './son/Niveau 1 - Bach flute.wav'
duree = librosa.get_duration(filename=filename)
x, fs = librosa.load(filename, sr=None, mono=True, duration=duree)
print(duree)

instruments = {
    0: 'Acoustic Grand Piano',
    1: 'Bright Acoustic Piano',
    2: 'Electric Grand Piano',
    3: 'Honky-tonk Piano',
    4: 'Electric Piano 1',
    5: 'Electric Piano 2',
    6: 'Harpsichord',
    7: 'Clavinet',
    8: 'Celesta',
    9: 'Glockenspiel',
    10: 'Music Box',
    11: 'Vibraphone',
    12: 'Marimba',
    13: 'Xylophone',
    14: 'Tubular Bells',
    15: 'Dulcimer',
    16: 'Drawbar Organ',
    17: 'Percussive Organ',
    18: 'Rock Organ',
    19: 'Church Organ',
    20: 'Reed Organ',
    21: 'Accordion',
    22: 'Harmonica',
    23: 'Tango Accordion',
    24: 'Acoustic Guitar (nylon)',
    25: 'Acoustic Guitar (steel)',
    26: 'Electric Guitar (jazz)',
    27: 'Electric Guitar (clean)',
    28: 'Electric Guitar (muted)',
    29: 'Overdriven Guitar',
    30: 'Distortion Guitar',
    31: 'Guitar Harmonics',
    32: 'Acoustic Bass',
    33: 'Electric Bass (finger)',
    34: 'Electric Bass (pick)',
    35: 'Fretless Bass',
    36: 'Slap Bass 1',
    37: 'Slap Bass 2',
    38: 'Synth Bass 1',
    39: 'Synth Bass 2',
    40: 'Violin',
    41: 'Viola',
    42: 'Cello',
    43: 'Contrabass',
    44: 'Tremolo Strings',
    45: 'Pizzicato Strings',
    46: 'Orchestral Harp',
    47: 'Timpani',
    48: 'String Ensemble 1',
    49: 'String Ensemble 2',
    50: 'Synth Strings 1',
    51: 'Synth Strings 2',
    52: 'Choir Aahs',
    53: 'Voice Oohs',
    54: 'Synth Choir',
    55: 'Orchestra Hit',
    56: 'Trumpet',
    57: 'Trombone',
    58: 'Tuba',
    59: 'Muted Trumpet',
    60: 'French Horn',
    61: 'Brass Section',
    62: 'Synth Brass 1',
    63: 'Synth Brass 2',
    64: 'Soprano Sax',
    65: 'Alto Sax',
    66: 'Tenor Sax',
    67: 'Baritone Sax',
    68: 'Oboe',
    69: 'English Horn',
    70: 'Bassoon',
    71: 'Clarinet',
    72: 'Piccolo',
    73: 'Flute',
    74: 'Recorder',
    75: 'Pan Flute',
    76: 'Blown Bottle',
    77: 'Shakuhachi',
    78: 'Whistle',
    79: 'Ocarina',
    80: 'Lead 1 (square)',
    81: 'Lead 2 (sawtooth)',
    82: 'Lead 3 (calliope)',
    83: 'Lead 4 (chiff)',
    84: 'Lead 5 (charang)',
    85: 'Lead 6 (voice)',
    86: 'Lead 7 (fifths)',
    87: 'Lead 8 (bass + lead)',
    88: 'Pad 1 (new age)',
    89: 'Pad 2 (warm)',
    90: 'Pad 3 (polysynth)',
    91: 'Pad 4 (choir)',
    92: 'Pad 5 (bowed)',
    93: 'Pad 6 (metallic)',
    94: 'Pad 7 (halo)',
    95: 'Pad 8 (sweep)',
    96: 'FX 1 (rain)',
    97: 'FX 2 (soundtrack)',
    98: 'FX 3 (crystal)',
    99: 'FX 4 (atmosphere)',
    100: 'FX 5 (brightness)',
    101: 'FX 6 (goblins)',
    102: 'FX 7 (echoes)',
    103: 'FX 8 (sci-fi)',
    104: 'Sitar',
    105: 'Banjo',
    106: 'Shamisen',
    107: 'Koto',
    108: 'Kalimba',
    109: 'Bagpipe',
    110: 'Fiddle',
    111: 'Shanai',
    112: 'Tinkle Bell',
    113: 'Agogo',
    114: 'Steel Drums',
    115: 'Woodblock',
    116: 'Taiko Drum',
    117: 'Melodic Tom',
    118: 'Synth Drum',
    119: 'Reverse Cymbal',
    120: 'Guitar Fret Noise',
    121: 'Breath Noise',
    122: 'Seashore',
    123: 'Bird Tweet',
    124: 'Telephone Ring',
    125: 'Helicopter',
    126: 'Applause',
    127: 'Gunshot'
}

# Analyse du tempo du morceau
tempo, _ = librosa.beat.beat_track(y=x, sr=fs)
print(f"Detected tempo: ",tempo," BPM")

# Adapter les paramètres en fonction du tempo
nfft = 2048 # Si le tempo est rapide (arbitrairement au-dessus de 120 BPM) on réduire la taille de la fenêtre pour une meilleure résolution temporelle
hop_length = int(nfft * 0.5) 

# Calcul du spectrogramme
D = librosa.stft(x, n_fft=nfft, hop_length=hop_length)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Affichage du spectrogramme
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=fs, hop_length=hop_length, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme')
#plt.show()

def freq_to_note(freq):
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)
    if freq == 0:
        return None
    h = round(12 * np.log2(freq / C0))
    n = h % 12
    note_noms = ['DO', 'DO#', 'RE', 'RE#', 'MI', 'FA', 'FA#', 'SOL', 'SOL#', 'LA', 'LA#', 'SI']
    return note_noms[n],h

detected_notes = []
note_durations = []  # Liste pour stocker la durée des notes
prev_t = 0

# Détection des fréquences dominantes
pitches, magnitudes = librosa.piptrack(y=x, sr=fs, n_fft=nfft, hop_length=hop_length)

# Seuil minimum de magnitude pour éviter les fausses détections
magnitude_threshold = 0.05 * np.max(magnitudes)

for t in range(pitches.shape[1]):  # Parcours de chaque frame
    index = magnitudes[:, t].argmax()  # Trouver l'index de la fréquence dominante
    pitch = pitches[index, t]

    # Vérifier si la magnitude est au-dessus du seuil
    if pitch > 0 and magnitudes[index, t] > magnitude_threshold:
        note_name,note_midi = freq_to_note(pitch)  # Récupérer le nom de la note et son numéro MIDI
        if note_name :
            detected_notes.append((note_name, pitch,note_midi))  # Ajouter la note et son numéro MIDI
            if t > 0:
                # Calcul de la durée de la note en fonction de la différence de temps
                duration = librosa.frames_to_time(t - prev_t, sr=fs)
                note_durations.append(duration)
            prev_t = t


toute_notes = [[detected_notes[i][0],detected_notes[i][1],detected_notes[i][2],note_durations[i]] for i in range(len(detected_notes))
               if i == 0 or detected_notes[i][0] != detected_notes[i-1][0]]

notes_sans_repete = [[toute_notes[k][0],toute_notes[k][1],toute_notes[k][2],toute_notes[k][3]*60*k,toute_notes[k][3]*50] for k in range(len(toute_notes))]

toute_filtre = [detected_notes[i][0] for i in range(len(detected_notes))
               if i == 0 or detected_notes[i][0] != detected_notes[i-1][0]]

# Filtrer pour les notes les plus fréquentes
note_counts = Counter(toute_filtre)
most_common_notes = [note for note, count in note_counts.most_common(nb_note)]

final_notes = []

for note in notes_sans_repete :
    if note[0] in most_common_notes :
        final_notes.append(note)

# Sauvegarder le fichier MIDI
track = 0
channel = 0
tempo = tempo  # En BPM
volume = 100  # 0-127, selon la norme MIDI

# Création du fichier MIDI
midi = MIDIFile(1)
midi.addTempo(0, 0, tempo)
program = 0
midi.addProgramChange(track, channel, 0, program)


# Ajout des notes au fichier MIDI
for i, (note_name, freq, midi_note, start_time, frame) in enumerate(final_notes): # frame = duree
    print(note_name,start_time)
    #duration = note_durations[i] if i < len(note_durations) else 0.5  # Durée par défaut
    midi.addNote(track, channel, midi_note, start_time, frame, 100)  # Canal 0, vélocité 100

# Écriture du fichier MIDI
with open("output_notes.mid", "wb") as output_file:
    midi.writeFile(output_file)
