## Project files

All code for the project is stored in the `code` folder

### Audio Toolbox

This folder contains different components used in the project.

- `dataset.py`: contains the class `AudioOTFDataset`, used for generating tensor representations of the audios stored as *.wav* files on the fly. It currently contains basic feature engineering techniques for audios (**will contain more feature engineering techniques**).
- `metric.py`: contains helper methods to calculate metrics for the classification task.
- `model.py`: **will** contain deep learning models developed for the project.
- 'preprocess.py': contains visualization tool for generated features of the audios.
- `trainer.py`: contains the trainer used to train deep learning models under the pytorch framework.

### Jupyter notebooks

Several jupyter notebooks are provided to demo tasks within the project.

- `test_trainer.ipynb`: demo the whole pipeline for training a simple linear model
- `explorations.ipynb`: demo plotting generated features of audios
- `traditional_ml_modeling.ipynb`: demo fitting some machine learning models for the classification task

## To run docker container:

- docker-compose up --build
- Go to http://localhost:4000 to see your live site!
