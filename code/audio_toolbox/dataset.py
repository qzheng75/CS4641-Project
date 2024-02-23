import os
from typing import List

from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import librosa
import torch
from torch.utils.data import Dataset


feature_args = {
    'n_mfcc': 12, # number of mfcc to return
    'n_chroma': 12, # number of chroma bins to return
    'n_derivatives': 2 # number of maximum order of derivatives to take
}

class AudioDataset(Dataset):
    def __init__(self,
                 root_folder: str,
                 filenames: List[str],
                 labels: List[str],
                 num_frames: int,
                 scaling_strategy: str,
                 name: str='Audio Dataset',
                 label_encoding: str='Onehot',
                 **kwargs) -> None:
        """
        Init method for AudioDataset

        Args:
            root_folder (str): root folder where data are stored
            filenames (List[str]): list of names of audio files
            labels (List[str]): list of labels of audio files
            num_frames (int): number of frames to be considered
            scaling_strategy (str): strategy for scaling processed data
            name (str, optional): name of the dataset. Defaults to 'Audio Dataset'.
        """
        super(AudioDataset, self).__init__()
        self.root_folder = root_folder
        self.dataset_name = name
        self.num_frames = num_frames  
                
        self.filenames = filenames
        self.raw_labels = labels
        self.kwargs = kwargs
        
        if label_encoding == 'Onehot':
            self.encoder = OneHotEncoder()
            self.labels = self.encoder.fit_transform(np.array(labels).reshape(-1, 1))
            self.num_classes = self.labels.shape[1]
        else:
            self.encoder = LabelEncoder()
            self.labels = self.encoder.fit_transform(np.array(labels))
            self.num_classes = np.max(self.labels) + 1
        
        assert len(filenames) == len(labels),\
            f'Number of files should match number of labels, but got {len(filenames)}'
            
        assert len(filenames) != 0,\
            f'The dataset shouldn\' be empty.'
        
        assert scaling_strategy is None or scaling_strategy in ('min-max', 'standard', 'max-abs'),\
            f'{scaling_strategy} is not supported now.'
        self.scaling_strategy = scaling_strategy
        
        if 'n_mfcc' in kwargs.keys():
            feature_args['n_mfcc'] = kwargs['n_mfcc']
        if 'n_chroma' in kwargs.keys():
            feature_args['n_chroma'] = kwargs['n_chroma']
        if 'n_derivatives' in kwargs.keys():
            feature_args['n_derivatives'] = kwargs['n_derivatives']
        assert feature_args['n_mfcc'] == feature_args['n_chroma']\
            , f"Out channels must be the same, but got {feature_args['n_mfcc']} for mfcc and {feature_args['n_chroma']} for chroma"
            
        self.X = [self.__load_from_folder(i) for i in tqdm(range(len(filenames)), desc=f"Loading audios for {self.dataset_name}")]
        self.__process_raw_audio()
            
    def __load_from_folder(self, idx: int) -> np.ndarray:
        """
        Load one audio based on index.

        Args:
            idx (int): index of audio to load

        Returns:
            np.ndarray: vectorized representation of the loaded audio
        """
        filename, label = self.filenames[idx], self.raw_labels[idx]
        audio_path = os.path.join(self.root_folder, 'genres_original', label, filename)
        x, _ = librosa.load(audio_path, **self.kwargs)
        return x
    
    def __compute_features(self, x: np.ndarray) -> np.ndarray:
        """
        Compute features for input vectorized audio.

        Args:
            x (np.ndarray): original vectorized audio

        Returns:
            np.ndarray: concatenated features of the input audio
        """
        n_mfcc = feature_args['n_mfcc']
        n_chroma = feature_args['n_chroma']
        n_derivatives = feature_args['n_derivatives']
        
        # Compute MFCC features and their deltas
        # Shape: (n_mfcc, num_frames)
        mfcc = librosa.feature.mfcc(y=x, n_mfcc=n_mfcc)
        
        # Shape: (n_mfcc, num_frames)
        chroma = librosa.feature.chroma_stft(y=x, n_chroma=n_chroma)
        
        # Delta terms Shape: (n_derivatives * n_mfcc, num_frames)
        mfcc_deltas = np.concatenate([librosa.feature.delta(mfcc, order=i) for i in range(1, n_derivatives + 1)], axis=0)
        chroma_deltas = np.concatenate([librosa.feature.delta(chroma, order=i) for i in range(1, n_derivatives + 1)], axis=0)
        
        # Concatenate all features together
        features = np.concatenate([mfcc[:, :self.num_frames],
                                   mfcc_deltas[:, :self.num_frames],
                                   chroma[:, :self.num_frames],
                                   chroma_deltas[:, :self.num_frames]], axis=0)
        
        # Final feature shape: ((2 * n_derivatives + 2) * n_mfcc, num_frames)
        return features
    
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
        return torch.tensor(x)
    
    def __len__(self) -> int:
        """
        Return number of samples of the dataset

        Returns:
            int: number of samples of the dataset
        """
        return len(self.filenames)
    
    def __process_raw_audio(self):
        vec_list = []
        for x in tqdm(self.X, desc=f'Processing for {self.dataset_name}'):
            x = self.__compute_features(x)
            if self.scaling_strategy is not None:
                x = self.__scaling(x)
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
        return self.X[index]
    
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
            f"Shape of one sample: {self.__getitem__(0).size()}\n" +\
            f"Number of classes: {self.num_classes}\n" +\
            f"Features:\n\tn_mfcc: {feature_args['n_mfcc']}\n\tn_chroma: {feature_args['n_chroma']}\n" +\
            f"\tn_derivatives: {feature_args['n_derivatives']}\n" +\
            f"Scaling strategy: {self.scaling_strategy}\n" +\
            "=" * l