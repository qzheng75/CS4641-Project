import os
from typing import List
from PIL import Image

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


feature_args = {
    'n_features': 12,
    'n_derivatives': 2 # number of maximum order of derivatives to take
}

class AudioOTFDataset(Dataset):
    def __init__(self,
                 root_folder: str,
                 filenames: List[str],
                 labels: List[str],
                 num_frames: int,
                 scaling_strategy: str,
                 name: str='Audio Dataset',
                 label_encoding: str='Onehot',
                 features_to_compute: list=None,
                 flatten_features: bool=True,
                 shuffle: bool=False,
                 random_state: int=0,
                 device: str='cpu',
                 **kwargs) -> None:
        """
        Init method for AudioOTFDataset

        Args:
            root_folder (str): root folder where data are stored
            filenames (List[str]): list of names of audio files
            labels (List[str]): list of labels of audio files
            num_frames (int): number of frames to be considered
            scaling_strategy (str): strategy for scaling processed data
            name (str, optional): name of the dataset. Defaults to 'Audio Dataset'.
            label_encoding (str, optional): method to encode raw labels. Defaults to 'Onehot'.
            shuffle (bool, optional): whether to shuffle the data. Defaults to False.
            random_state (int, optional): seed for shuffling the data. Defaults to 0.
            device (str, optional): device for tensors. Defaults to cpu.
        """
        super(AudioOTFDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset_name = name
        self.num_frames = num_frames
        self.device = device  
                
        self.filenames = filenames
        self.raw_labels = labels
        self.flatten_features = flatten_features
        self.kwargs = kwargs
        
        if label_encoding == 'Onehot':
            self.encoder = OneHotEncoder()
            self.labels = self.encoder.fit_transform(np.array(labels).reshape(-1, 1))
            self.num_classes = self.labels.shape[1]
        else:
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(np.array(labels))
            self.num_classes = np.max(self.labels) + 1
            
        self.labels = self.to_tensor(self.labels)
        
        assert len(filenames) == len(labels),\
            f'Number of files should match number of labels, but got {len(filenames)}'
            
        assert len(filenames) != 0,\
            f'The dataset shouldn\' be empty.'
        
        assert scaling_strategy is None or scaling_strategy in ('min-max', 'standard', 'max-abs'),\
            f'{scaling_strategy} is not supported now.'
        self.scaling_strategy = scaling_strategy
        
        if 'n_features' in kwargs.keys():
            feature_args['n_features'] = kwargs['n_features']
            
        self.load_info = [self.__load_from_folder(i) for i in tqdm(range(len(filenames)), desc=f"Loading audios for {self.dataset_name}")]
        self.X = self.load_info #TODO
        
        if features_to_compute is None or len(features_to_compute) == 0:
            self.features_to_compute = ['chroma', 'mfcc', 'tempogram', 'mel_spectrogram']
        else:
            self.features_to_compute = features_to_compute
        self.__process_raw_audio()
        if shuffle:
            torch.random.manual_seed(random_state)
            shuffled_index = torch.randperm(len(self.X))
            self.X = self.X[shuffled_index]
            self.labels = self.labels[shuffled_index]
            
    def __load_from_folder(self, idx: int) -> np.ndarray:
        """
        Load one audio based on index.

        Args:
            idx (int): index of audio to load

        Returns:
            np.ndarray: vectorized representation of the loaded audio
            int: sample rate of the audio
        """
        filename, label = self.filenames[idx], self.raw_labels[idx]
        audio_path = os.path.join(self.root_folder, 'genres_original', label, filename)
        x, sample_rate = librosa.load(audio_path)
        return x, sample_rate
    
    def __augment_data(self, x: np.ndarray, sample_rate, **kwargs) -> np.ndarray:
        """
        Augment one audio sample.

        Args:
            x (np.ndarray): original vectorized audio

        Returns:
            np.ndarray: the augmented audio
        """
        augmentations = []
        add_gaussian_noise = kwargs.get('add_gausian_noise', dict())
        time_stretch = kwargs.get('time_stretch', dict())
        pitch_shift = kwargs.get('pitch_shift', dict())
        shift = kwargs.get('shift', dict())
        
        if 'add_gausian_noise' in kwargs.keys():
            # args: min_amplitude, max_amplitude, p
            augmentations.append(AddGaussianNoise(**add_gaussian_noise))
        if 'time_stretch' in kwargs.keys():
            # args: min_rate, max_rate, p
            augmentations.append(TimeStretch(**time_stretch))
        if 'pitch_shift' in kwargs.keys():
            # args: min_semitones, max_semitones, p
            augmentations.append(PitchShift(**pitch_shift))
        if 'shift' in kwargs.keys():
            # args: min_fraction, max_fraction, p
            augmentations.append(Shift(**shift))
        augment = Compose(augmentations)
        return augment(samples=x, sample_rate=sample_rate)
    
    def __compute_features(self, x: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute features for input vectorized audio.

        Args:
            x (np.ndarray): original vectorized audio

        Returns:
            np.ndarray: concatenated features of the input audio
        """
        n_feature = feature_args['n_features']
        n_derivatives = feature_args['n_derivatives']
        
        features = []

        if 'mfcc' in self.features_to_compute:
            mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_feature)
            mfcc_deltas = np.stack([librosa.feature.delta(mfcc, order=i) for i in range(1, n_derivatives + 1)], axis=0)
            features.extend([mfcc[:, :self.num_frames][np.newaxis, :, :], mfcc_deltas[:, :, :self.num_frames]])
        
        if 'chroma' in self.features_to_compute:
            chroma = librosa.feature.chroma_stft(y=x, sr=sr, n_chroma=n_feature)
            chroma_deltas = np.stack([librosa.feature.delta(chroma, order=i) for i in range(1, n_derivatives + 1)], axis=0)
            features.extend([chroma[:, :self.num_frames][np.newaxis, :, :], chroma_deltas[:, :, :self.num_frames]])
        
        if 'mel_spectrogram' in self.features_to_compute:
            mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_feature)
            mel_deltas = np.stack([librosa.feature.delta(mel_spectrogram, order=i) for i in range(1, n_derivatives + 1)], axis=0)
            features.extend([mel_spectrogram[:, :self.num_frames][np.newaxis, :, :], mel_deltas[:, :, :self.num_frames]])
            
        if 'tempogram' in self.features_to_compute:
            tempogram = librosa.feature.tempogram(y=x, sr=sr, win_length=n_feature)
            temp_deltas = np.stack([librosa.feature.delta(tempogram, order=i) for i in range(1, n_derivatives + 1)], axis=0)
            features.extend([tempogram[:, :self.num_frames][np.newaxis, :, :], temp_deltas[:, :, :self.num_frames]])

        return np.concatenate(features, axis=0)
    
    def __scaling(self, x: np.ndarray) -> np.ndarray:
        """
        Scale the input vector.

        Args:
            x (np.ndarray): np array to scale

        Returns:
            np.ndarray: scaled array
        """
        if self.scaling_strategy == 'min-max':
            scaler = MinMaxScaler()
        elif self.scaling_strategy == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MaxAbsScaler()
        
        return scaler.fit_transform(X=x)
    
    def to_tensor(self, x: np.ndarray) -> torch.tensor:
        """
        Turn the input np array to a torch tensor.

        Args:
            x (np.ndarray): input np array

        Returns:
            torch.tensor: transformed tensor
        """
        return torch.tensor(x, device=self.device)
    
    def __len__(self) -> int:
        """
        Return number of samples of the dataset

        Returns:
            int: number of samples of the dataset
        """
        return len(self.filenames)
    
    def __process_raw_audio(self):
        """
        Process the raw audio and store the processed audios in self.X
        """
        vec_list = []
        for x, sr in tqdm(self.X, desc=f'Processing for {self.dataset_name}'):
            x = self.__compute_features(x, sr)
            if self.scaling_strategy is not None:
                x = self.__scaling(x)
            if self.flatten_features:
                x = self.to_tensor(x).view(1, -1).squeeze()
            else:
                x = self.to_tensor(x)
            vec_list.append(x)
        self.X = torch.stack(vec_list)
            
    
    def __getitem__(self, index: int) -> torch.tensor:
        """
        Get a sample from the dataset.

        Args:
            index (int): index of the sample

        Returns:
            torch.tensor: vectorized representation of the audio
        """
        return self.X[index], self.labels[index]
    
    def get_raw_label(self, index: int) -> np.ndarray:
        """
        Get the raw label of the sample of the input index.

        Args:
            index (int): index of the sample

        Returns:
            np.ndarray: the raw label of the sample
        """
        return self.raw_labels[index]

    def get_label(self, index: int) -> np.ndarray:
        """
        Get label of the sample of the input index.

        Args:
            index (int): index of the sample

        Returns:
            np.ndarray: the label of the sample
        """
        return self.labels[index].toarray()
    
    def get_filename(self, index: int) -> str:
        """
        Get the file name of the input index.

        Args:
            index (int): index of the sample

        Returns:
            str: the file name of the sample
        """
        return self.filenames[index]
    
    def __repr__(self):
        l = len(f"====== {self.dataset_name} ======")
        return f"====== {self.dataset_name} ======\n" +\
            f"Root folder: {self.root_folder}\n" +\
            f"Number of samples: {len(self)}\n" +\
            f"Shape of one sample: {self.__getitem__(0)[0].size()}\n" +\
            f"Number of classes: {self.num_classes}\n" +\
            f"Features included: {self.features_to_compute}\n" +\
            f"Scaling strategy: {self.scaling_strategy}\n" +\
            "=" * l
            
            
class AudioImageDataset(Dataset):
    def __init__(self, root_folder, filenames, labels,
                 label_encoding: str='Onehot',
                 name: str='Full dataset') -> None:
        super(AudioImageDataset, self).__init__()
        self.filenames = filenames
        self.raw_labels = labels
        self.dataset_name = name
        self.root_folder = root_folder
        
        self.X = [self.__load_from_folder(i) for i in tqdm(range(len(filenames)), desc=f"Loading audios for {self.dataset_name}")]
        
        if label_encoding == 'Onehot':
            self.encoder = OneHotEncoder()
            self.labels = self.encoder.fit_transform(np.array(labels).reshape(-1, 1))
            self.num_classes = self.labels.shape[1]
        else:
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(np.array(labels))
            self.num_classes = np.max(self.labels) + 1
            
        self.X = self.__generate_images()
            
    def __generate_images(self):
        images = []
        for x, sr in tqdm(self.X, desc=f'Processing for {self.dataset_name}'):
            mel_spectrogram = librosa.feature.melspectrogram(y=x, sr=sr)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            librosa.display.specshow(mel_spectrogram_db, sr=sr, fmax=8000)
            plt.axis('off')
            plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
            img = Image.open('temp.png')
            images.append(np.array(img))
        os.remove('temp.png')
        return np.stack(images)
        
    def __load_from_folder(self, idx: int) -> np.ndarray:
        """
        Load one audio based on index.

        Args:
            idx (int): index of audio to load

        Returns:
            np.ndarray: vectorized representation of the loaded audio
            int: sample rate of the audio
        """
        filename, label = self.filenames[idx], self.raw_labels[idx]
        audio_path = os.path.join(self.root_folder, 'genres_original', label, filename)
        x, sample_rate = librosa.load(audio_path)
        return x, sample_rate
    
    def to_tensor(self, **kwargs):
        return torch.tensor(self.X, **kwargs)
    
    def __getitem__(self, index: int) -> torch.tensor:
        """
        Get a sample from the dataset.

        Args:
            index (int): index of the sample

        Returns:
            torch.tensor: vectorized representation of the audio
        """
        return self.X[index], self.labels[index]

def visualize_confusion_matrices(conf_matrices, splits, suptitle):

    # Create subplots
    fig, axes = plt.subplots(1, len(splits), figsize=(15, 5))

    # Plot each confusion matrix
    for i, (conf_matrix, split) in enumerate(zip(conf_matrices, splits)):
        ax = axes[i]
        ax.imshow(conf_matrix, cmap='Blues', interpolation='nearest')

        # Add color bar
        cb = ax.figure.colorbar(ax.imshow(conf_matrix, cmap='Blues', interpolation='nearest'), ax=ax, fraction=0.046, pad=0.04)

        # Add labels
        ax.set_title(f'{split} Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.grid(False)

        # Add tick marks
        class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
            'metal', 'pop', 'reggae', 'rock']
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)

        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, format(conf_matrix[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")

        # plt.tight_layout()
        fig.suptitle(suptitle, fontsize=16)