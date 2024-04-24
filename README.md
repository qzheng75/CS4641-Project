## Project files

All code for the project is stored in the `code` folder

### Strategy 1: Stacked MFCC & Chroma

Files related to strategy one include:

- `audio_toolbox/dataset.py`: contains the class `AudioOTFDataset`, used for generating vectorized representations of the audios stored as *.wav* files on the fly.
- `notebooks/dataset_generation/generate_full_dataset.ipynb`: the notebook to generate and save the datasets for strategy one.
- `notebooks/strategy_one.ipynb`: the notebook to preprocess and fit ML models on the dataset.
- `notebooks/mlp_strategy_one.ipynb`: the notebook to use MLP fit the dataset.

### Strategy 2: Manual Feature Engineering

Files related to strategy one include:

- `audio_toolbox/dataset.py`: contains the class `AudioOTFDataset`, used for generating vectorized representations of the audios stored as *.wav* files on the fly.
- `notebooks/dataset_generation/generate_full_dataset.ipynb`: the notebook to generate and save the datasets for strategy one.
- `notebooks/strategy_one.ipynb`: the notebook to preprocess and fit ML models on the dataset.
- `notebooks/mlp_strategy_one.ipynb`: the notebook to use MLP fit the dataset.

### Jupyter notebooks

Several jupyter notebooks are provided to demo tasks within the project.

- `3_sec_tabular_ml_modeling.ipynb`: modeling on tabular data extracted from 3-second audio clips
- `explorations.ipynb`: demo plotting generated features of audios.
- `generate_30sec_tabular_dataset.ipynb`: generation of a tabular dataset from 30-second audio clips.
- `generate_3sec_tabular_dataset.ipynb`: generation of a tabular dataset from 3-second audio clips.
- `generate_full_dataset.ipynb`: compile the full dataset used for training.
- `generate_image_dataset.ipynb`: prepare an image-based dataset from audio data.
- `mlp_training.ipynb`: training of a Multilayer Perceptron (MLP) on the generated datasets.
- `slicing_random_forest.ipynb`: application of the Random Forest algorithm on sliced audio data.
- `strategy_one.ipynb`: implementation of the first strategy for audio analysis.
- `tabular_ml_modeling.ipynb`: machine learning modeling on tabular data.

## To run docker container:

- docker-compose up --build
- Go to http://localhost:4000 to see your live site!
