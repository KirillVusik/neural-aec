# neural-aec
Attempt to implement an Neural Arithmetic Expression Calculator

Limitations:
  * Only integers are allowed
  * Numbers are in a range [-99, 99]
  * Max numbers in expression: 4
  * Allowed operations: +, -
  * Max expression length (w/o whitespaces): 16

## Install
### Install required Python packages

CPU: ```pip install -r requirements.txt```

GPU: ```pip install -r requirements-gpu.txt```

Note: the only difference of the ```requirements-gpu.txt``` from the standard one ```tensorflow-gpu```

### Train a model from scratch:
	
Note: Please feel free to skip this section if you would like to run with the pre-trained model

1. Generate dataset for training:

   ```python generate_dataset.py --count <SAMPLES_COUNT> --output_path <DATASET_PATH>```
2. Train models:

   ```python train_models.py --datasets_path <DATASET_PATH> --output <MODEL PATH>```
   
   Note: there are parameters in the beginning of the train_models.py for a grid search training configuration.
3. Set the parameters in ```config.py``` to point to the saved model you neen, e.g:

   ```
   MODEL_ARCHITECTURE = './trained_models/some-architecture.json'
   MODEL_WEIGHTS = './trained_models/some-weights.json'
   ```
   
   Note: I am not using checkpoint here to allow loading a CPU compatible models with weights trained on a GPU
   
### Run application:

1. Make sure that the ```MODEL_ARCHITECTURE``` and the ```MODEL_WEIGHTS``` in the ```config.py``` are valid paths to model
2. Set the current directory to be the repository root
3. python app.py
