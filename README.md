## Project files

All code for the project is stored in the `code` folder

### Audio Toolbox

This folder contains different components used in the project.

- `dataset.py`: contains the class `AudioOTFDataset`, used for generating tensor representations of the audios stored as *.wav* files on the fly. It currently contains basic feature engineering techniques for audios (**will contain more feature engineering techniques**).
- `metrics.py`: contains helper methods to calculate metrics for the classification task.
- `models.py`: **will** contain deep learning models developed for the project.
- `trainer.py`: contains the trainer used to train deep learning models under the pytorch framework.
- `visualize.py`: includes tools for visualizing audio data and model results.

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
