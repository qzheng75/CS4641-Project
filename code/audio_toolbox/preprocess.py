import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import librosa

class AudioVisualizer:
    def __init__(self, feature_to_visualize: str) -> None:
        try:
            self.feature_to_visualize = feature_to_visualize
            self.feature = getattr(librosa.feature, feature_to_visualize)
        except AttributeError:
            raise AttributeError(f"{feature_to_visualize} is not a supported feature in librosa.feature")
        
    def visualize(self, audio: np.ndarray, filename: str=None, **kwargs) -> Figure:
        vis = self.feature(y=audio, **kwargs)
        fig, ax = plt.subplots(figsize=(7, 3))
        if self.feature_to_visualize == 'chroma_stft':
            librosa.display.specshow(vis, ax=ax, x_axis='s', y_axis='chroma')
            ax.set(title=f'Chroma, File: {filename}' if filename is not None else '')
        else:
            librosa.display.specshow(vis, ax=ax, x_axis='s')
            ax.set(title=f'Feature: {self.feature_to_visualize}'
                + f', File: {filename}' if filename is not None else '', ylabel=self.feature_to_visualize)
        return fig

