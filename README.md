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

- `explorations.ipynb`: demo plotting generated features of audios.
- `generate_full_dataset.ipynb`: compile the full dataset used for training.
- `generate_image_dataset.ipynb`: prepare an image-based dataset from audio data.
- `generate_tabular_dataset.ipynb`: generate a tabular dataset from audio features.
- `mlp_training.ipynb`: (started) training a Multilayer Perceptron on the audio data
- `tabular_ml_modeling.ipynb`: modeling with traditional machine learning techniques on tabular data.
- `traditional_ml_modeling.ipynb`: demo fitting some machine learning models for the classification task
- `VGG_ft.ipynb`: (started) fine-tuning a pre-trained VGG network on audio-derived image data.

## To run docker container:

- docker-compose up --build
- Go to http://localhost:4000 to see your live site!
