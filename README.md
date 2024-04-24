## Project files

All code for the project is stored in the `code` folder

### Strategy 1: Stacked MFCC & Chroma

Files related to strategy one include:

- `audio_toolbox/dataset.py`: contains the class `AudioOTFDataset`, used for generating vectorized representations of the audios stored as *.wav* files on the fly.
- `notebooks/dataset_generation/generate_full_dataset.ipynb`: the notebook to generate and save the datasets for strategy one.
- `notebooks/strategy_one.ipynb`: the notebook to preprocess and fit ML models on the dataset.
- `notebooks/mlp_strategy_one.ipynb`: the notebook to use MLP fit the dataset.

### Strategy 2: Manual Feature Engineering

Files related to strategy two include:

- `notebooks/dataset_generation/generate_30sec_tabular_dataset.ipynb`: the notebook to generate features for audios and save the datasets for strategy 2.1.
- `notebooks/tabular_ml_modeling.ipynb.ipynb`: the notebook to preprocess and fit ML models on the generated datasets with strategy 2.1.
- `audio_toolbox/datasets.py`: contains `SplitedDataset`, which involved audio spliting and feature computation methods.
- `notebooks/dataset_generation/generate_split_dataset.ipynb`: the notebook to generated splited audios and compute features for both the original audio and the audio segments for strategy 2.2.
- `notebooks/split_weight_training_3_sec.ipynb` and `notebooks/split_weight_training_6_sec.ipynb`: the notebook for fitting ML models with strategy 2.2.
  
### Strategy 3: Two-Step VGG fine-tuning

Files related to strategy three include:

- `audio_toolbox/datasets.py`: contains `AudioImageDataset`, which is used to generate spectrograms for audios.
- `notebooks/dataset_generation/generate_image_dataset.ipynb`: the notebook to generate spectrograms for audios and save the datasets for strategy 3.
- `notebooks/VGG_fine_tuning.ipynb`: the notebook for fine-tuning VGG according to strategy 3.

## To run docker container:

- docker-compose up --build
- Go to http://localhost:4000 to see your live site!
