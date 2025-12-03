"""
Creates 'music' from an image, by randomly sampling pixels
and mapping the values in their rgb or hvs channels to
a note (base) frequency, duration and octave.
"""

import cv2
import numpy as np
from scipy.io import wavfile

import logging
from pathlib import Path

from typing import Dict, List, Sequence, Optional, Tuple

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
IMAGES = Path(__file__).parent.parent / "images"

class Scale:
    """
    Simple representation of a musical scale based on a starting key and octave.

    Attributes:
        octave: base octave for this scale.
        intervals: ordered list of keys starting from the chosen key.
    """

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

    def __init__(self, octave: int, key: str) -> None:
        """"Initialize scale.

        Args:
            octave: Octave number (int).
            key: Key of the scale (str). Must exist in INTERVALS.
        """
    
        # Load note dictionary
        self.octave: int = octave

        # Find index of desired key
        index = self.INTERVALS.index(key)
        # Redefine scale interval so that scale intervals begins with key
        self.intervals = (
            self.INTERVALS[index : len(self.INTERVALS)] + self.INTERVALS[:index]
        )

    @property
    def frequencies(self) -> List[float]:
        """List of frequencies corresponding to the scale intervals."""
        freqs: List[float] = []
        for i in range(len(self.intervals)):
            note = self.intervals[i] + str(self.octave)
            freqToAdd = self.piano_notes[note]
            freqs.append(freqToAdd)
        return freqs

    @property
    def piano_notes(self) -> Dict[str, float]:
        """Return a mapping from piano key name (e.g. 'A4') to frequency (Hz)."""
        # White keys are in Uppercase and black keys (sharps) are in lowercase
        octave = ["C", "c", "D", "d", "E", "F", "f", "G", "g", "A", "a", "B"]
        base_freq = 440  # Hz, Frequency of Note A4
        keys = np.array([x + str(y) for y in range(0, 9) for x in octave])
        # Trim to standard 88 keys
        start = np.where(keys == "A0")[0][0]
        end = np.where(keys == "C8")[0][0]
        keys = keys[start : end + 1]

        note_freqs: Dict[str, float] = dict(
            zip(keys, [2 ** ((n + 1 - 49) / 12) * base_freq for n in range(len(keys))])
        )
        note_freqs[""] = 0.0  # stop
        return note_freqs

    @staticmethod
    def get_sine_wave(
        frequency: float,
        duration: float,
        sample_rate: int = SAMPLE_RATE,
        amplitude: float = 4096.0,
    ) -> np.ndarray:
        """Generate a sine wave for the given frequency and duration.

        Returns:
            A numpy array containing the waveform samples.
        """
        t = np.linspace(0, duration, int(sample_rate * duration))  # Time axis
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        return wave


class Img2Song:
    """
    Convert an image (HSV) into audio samples by mapping hue/saturation/value to
    frequency, duration and octave respectively.
    """

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

    def __init__(self, image_filename: str) -> None:
        """Read an image and store its HSV representation.

        Args:
            image_filename: Path to image file.
        """
        orig = cv2.imread(image_filename)
        logger.info(f"Image Shape: {orig.shape}")
        self.hsv: np.ndarray = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
        # runtime-created mappings (populated in transform)
        self.HUE2FREQ: Dict[float, float] = {}
        self.HUE2DURATION: Dict[float, float] = {}
        self.HUE2OCTAVE: Dict[float, float] = {}

    def map_closest(self, hue: float, mapper: Dict[float, float]) -> float:
        """
        Map a scalar hue value to the closest key in mapper and return the mapped value.

        Args:
            hue: Scalar value to map.
            mapper: Dictionary whose keys are scalar thresholds and values are targets.

        Returns:
            The value from mapper corresponding to the closest key.
        """
        closest_threshold = min(mapper, key=lambda x: abs(x - hue))
        return mapper[closest_threshold]

    def transform(
        self,
        scale: Scale,
        sample_rate: int = 44100,
        note_base_duration: float = 0.2,
        note_varation_duration: Sequence[float] = [0.5,1,2,4],
        nPixels: int = 120,
        octaves: Sequence[float] = [0.5,1,2,4],
        harmonize: str = "M3",
        harmonize_octave: float = 0.5,
    ) -> np.ndarray:
        """
        Transform sampled pixels from the image into a 2D numpy array of audio samples.

        Args:
            scale: Scale object providing frequency candidates.
            sample_rate: Sampling rate in Hz.
            note_base_duration: Base duration used to compute sampling time vector.
            note_varation_duration: Sequence of available note durations. Defaults to [0.5,1,2,4].
            nPixels: Number of pixels to randomly sample from the image.
            octaves: Sequence of octave multipliers. Defaults to [0.5,1,2,4].
            harmonize: Key of HARMONY_SELECT to create harmony tones.
            harmonize_octave: Multiplier for harmony octave.

        Returns:
            A 2D numpy array where columns correspond to the main melody and harmonics.
        """
        # avoid mutable default arguments
        if note_varation_duration is None:
            note_varation_duration = [0.5, 1, 2, 4]
        if octaves is None:
            octaves = [0.5, 1, 2, 4]

        hsv = self.hsv.reshape((-1, 3))
        # Use the flattened hsv length for correct random sampling
        random_pixels = hsv[np.random.choice(hsv.shape[0],nPixels, replace=False)]
        h = random_pixels[:, 0]
        s = random_pixels[:, 1]
        v = random_pixels[:, 2]

        hue_frequency = np.linspace(
            h.min(), h.max(), len(scale.frequencies), endpoint=False
        )
        self.HUE2FREQ = {
            float(color): float(frequency)
            for color, frequency in zip(hue_frequency, scale.frequencies)
        }
        hue_durations = np.linspace(
            s.min(), s.max(), len(note_varation_duration), endpoint=False
        )
        self.HUE2DURATION = {
            float(color): float(duration)
            for color, duration in zip(hue_durations, note_varation_duration)
        }
        hue_octave = np.linspace(v.min(), v.max(), len(octaves), endpoint=False)
        self.HUE2OCTAVE = {
            float(color): float(octave) for color, octave in zip(hue_octave, octaves)
        }

        frequencies = np.vectorize(
            lambda x: self.map_closest(float(x), mapper=self.HUE2FREQ)
        )(h.flatten())
        durations = np.vectorize(
            lambda x: self.map_closest(float(x), mapper=self.HUE2DURATION)
        )(s.flatten())
        octaves_mapped = np.vectorize(
            lambda x: self.map_closest(float(x), mapper=self.HUE2OCTAVE)
        )(v.flatten())

        t = np.linspace(
            0, note_base_duration, int(note_base_duration * sample_rate), endpoint=False
        )  # time variable

        def sample(freq: float, duration: float, octave: float) -> Tuple[np.ndarray, np.ndarray]:
            """"Return a (note, harmony) tuple of numpy arrays for the given params."""
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

    logging.basicConfig(level=logging.INFO)

    image = IMAGES / "1.original" / "crab.png"

    combined = Img2Song(image_filename=image.as_posix()).transform(
        scale=Scale(octave=1, key="D"),
        harmonize="M3",
        sample_rate=SAMPLE_RATE,
        nPixels=1240,
        note_base_duration=0.25,
    )

    wavfile.write(f"{image.stem}.wav", rate=SAMPLE_RATE, data=combined.astype(np.float32))