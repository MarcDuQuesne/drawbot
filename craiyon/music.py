"""
Creates 'music' from an image, by randomly sampling pixels
and mapping the values in their rgb or hvs channels to
a note (base) frequency, duration and octave.
"""

import cv2
import numpy as np
from scipy.io import wavfile

import logging

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100

class Scale:

    # Define tones. Upper case are white keys in piano. Lower case are black keys
    INTERVALS = [
        "A",
        "a",
        "B",
        "C",
        "c",
        "D",
        "d",
        "E",
        "F",
        "f",
        "G",
        "g",
    ]  # sounds more like Deboussy.
    # INTERVALS = ["C", "D", "E", "F", "G", "A", "B"]  # C scale (?) only white keys
    # INTERVALS = ["D", "E", "f", "G", "A", "B", "c"]  # D scale (?)

    def __init__(self, octave, key):

        # Load note dictionary
        self.octave = octave

        # Find index of desired key
        index = self.INTERVALS.index(key)
        # Redefine scale interval so that scale intervals begins with key
        self.intervals = (
            self.INTERVALS[index : len(self.INTERVALS)] + self.INTERVALS[:index]
        )

    @property
    def frequencies(self):
        freqs = []
        for i in range(len(self.intervals)):
            note = self.intervals[i] + str(self.octave)
            freqToAdd = self.piano_notes[note]
            freqs.append(freqToAdd)
        return freqs

    @property
    def piano_notes(self):
        # White keys are in Uppercase and black keys (sharps) are in lowercase
        octave = ["C", "c", "D", "d", "E", "F", "f", "G", "g", "A", "a", "B"]
        base_freq = 440  # Hz, Frequency of Note A4
        keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
        # Trim to standard 88 keys
        start = np.where(keys == "A0")[0][0]
        end = np.where(keys == "C8")[0][0]
        keys = keys[start : end + 1]

        note_freqs = dict(
            zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))])
        )
        note_freqs[""] = 0.0  # stop
        return note_freqs

    @staticmethod
    def get_sine_wave(frequency, duration, sample_rate=SAMPLE_RATE, amplitude=4096):
        t = np.linspace(0, duration, int(sample_rate * duration))  # Time axis
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        return wave


class Img2Song:

    HARMONY_SELECT = {
        "U0": 1,  # Unison
        "ST": 16 / 15,  # semitone
        "M2": 9 / 8,  #  major second
        "m3": 6 / 5,  # minor third
        "M3": 5 / 4,  # major third
        "P4": 4 / 3,  # perfect fourth
        "DT": 45 / 32,  # diatonic tritone
        "P5": 3 / 2,  # perfect fifth
        "m6": 8 / 5,  # minor sixth
        "M6": 5 / 3,  # major sixth
        "m7": 9 / 5,  # minor seventh
        "M7": 15 / 8,  # major seventh
        "O8": 2,  # Octave
    }

    def __init__(self, image_filename):
        orig = cv2.imread(image_filename)
        logger.info(f"Image Shape: {orig.shape}")
        self.hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

    def map_closest(self, hue, mapper):
        closest_threshold = min(mapper, key=lambda x: abs(x - hue))
        return mapper[closest_threshold]

    def transform(
        self,
        scale: Scale,
        sample_rate:int = 44100,
        note_base_duration:float = 0.2,
        note_varation_duration: list[float] = [0.5, 1, 2, 4],
        nPixels:int = 120,
        octaves: list[float] = [0.5, 1, 2, 4],
        harmonize: str = "M3",
        harmonize_octave: float = 0.5,
    ):

        hsv = self.hsv.reshape((-1, 3))
        random_pixels = hsv[np.random.choice(self.hsv.shape[0], nPixels, replace=False)]
        h = random_pixels[:, 0]
        s = random_pixels[:, 1]
        v = random_pixels[:, 2]

        hue_frequency = np.linspace(
            h.min(), h.max(), len(scale.frequencies), endpoint=False
        )
        self.HUE2FREQ = {
            color: frequency
            for color, frequency in zip(hue_frequency, scale.frequencies)
        }
        hue_durations = np.linspace(
            s.min(), s.max(), len(note_varation_duration), endpoint=False
        )
        self.HUE2DURATION = {
            color: duration
            for color, duration in zip(hue_durations, note_varation_duration)
        }
        hue_octave = np.linspace(v.min(), v.max(), len(octaves), endpoint=False)
        self.HUE2OCTAVE = {color: octave for color, octave in zip(hue_octave, octaves)}

        frequencies = np.vectorize(lambda x: self.map_closest(x, mapper=self.HUE2FREQ))(
            h.flatten()
        )
        durations = np.vectorize(
            lambda x: self.map_closest(x, mapper=self.HUE2DURATION)
        )(s.flatten())
        octaves = np.vectorize(lambda x: self.map_closest(x, mapper=self.HUE2OCTAVE))(
            v.flatten()
        )

        t = np.linspace(
            0, note_base_duration, int(note_base_duration * sample_rate), endpoint=False
        )  # time variable

        def sample(freq, duration, octave):
            # Make note and harmony note
            note = 0.5 * np.sin(2 * np.pi * freq * t * duration * octave)
            harmony_notes = 0.5 * np.sin(
                2
                * np.pi
                * freq
                * t
                * duration
                * self.HARMONY_SELECT[harmonize]
                * harmonize_octave
            )
            return note, harmony_notes

        channels = zip(
            *map(sample, frequencies, durations, octaves)
        )  # The first one is the song, the rest are harmonics
        combined = np.vstack([np.array(channel).flatten() for channel in channels]).T
        return combined


if __name__ == "__main__":

    combined = Img2Song(image_filename=r"C:\Users\627728\Projects\Making-Music-From-Images\data\images\catterina.jpg",).transform(
        scale=Scale(octave=2, key="C"),
        harmonize="M3",
        sample_rate=SAMPLE_RATE,
        nPixels=240,
        note_base_duration=0.15,
    )

    wavfile.write("song.wav", rate=SAMPLE_RATE, data=combined.astype(np.float32))
