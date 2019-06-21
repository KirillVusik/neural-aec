# neural-aec
Attempt to implement an Neural Arithmetic Expression Calculator

Limitations:
  * Only integers are allowed
  * Numbers are in range [-99, 99]
  * Max numbers in expression: 4
  * Allowed operations: +, -
  * Max expression length (w/o whitespaces): 16

## Install
### Install required pyhton packages

CPU: ```pip install requirements.txt```

GPU: ```pip install requirements-gpu.txt```

Note: the only difference of between the ```requirements-gpu.txt``` and the standard one is a tensorflow-gpu

### Train a model from scratch:
	
Note: Please feel free to skip this section if you would like to run with the pre-trained model

1. Generate dataset for training:

   ```python generate_datasets.py --count <SAMPLES_COUNT> --output_path <DATASET_PATH>```
2. Train models:

	 ```python train_models.py --datasets_path <DATASET_PATH> --output <MODEL PATH>```
   
   Note: there are parameters in the beginning of the train_models.py for a grid search training
3. Set the parameters in ```config.py``` to point to the saved model, e.g:

   ```
	 MODEL_ARCHITECTURE = './trained_models/architecture.json'
   MODEL_WEIGHTS = './trained_models/weights.json'
   Note: I am not using checkpoint here to allow loading a CPU compatible models with weights trained on a GPU
   ```


### Run application:

1. Make sure that the ```MODEL_PATH``` is a valid path to model
2. Set the current directory to be the repository root
3. python app.py
