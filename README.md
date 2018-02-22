# tensorflow-template
My template for Tensorflow project.

## Directory organization
```
/
├─ src
│    ├─ main.py
│    └─ utils 
│         └ tictoc.py
├─ data  
└─ README.md
```

## Features

1. **System call parsing**  
    1.1. Syntax  
    ```
    python main.py -i <inputfile> -l <labelfile> [-m <modelpath> -e <epochs> | -t -m <modelpath>]

    -i, --inputfile=: File path to input data.
    -l, --labelfile=: File path to label data.
    -m, --modelpath=: File path to saved model. Including hyper-parameters for rebuilding Tensorflow graphs.
    -e, --epochs=: Number of epochs to train.
    -t, --test: Test a model.
    ```  
    1.2. Examples  
    ```
    Train a model from scratch 
        Note: Model must be defined in the source code, or use configuration file (See below).
        -i ../data/inputfile -l ../data/labelfile -e 1000
    
    Restore a trained model and continue training:
        -m ../models/model -i ../data/inputfile -l ../data/labelfile -e 1000
    
    Test a trained model:
        -t -m ../models/model -i ../data/inputfile -l ../data/labelfile
    ```


2. **Configuration files**  
    2.1. Syntax
    ```
    python main.py config_file.conf
    ```
    2.2. File format  
        Use JSON format, this will be interpreted to system call (see above). Model will be build w.r.t. to "hyperparameters".  
        E.g.,  
    ```
    Train a model from scratch: 
    {
      "test_flag": false,
      "input_data_path": "../data/inputfile",
      "label_data_path": "../data/labelfile",
      "epochs": 1000,
      "model_path": "",
      "hyperparameters":
      {
        "neuron_numbers": [128, 256, 128],
        "batch_size": 1000,
        "learning_rate": 0.0001
      }
    }
    ```
    ```
    Restore a model and continue training: 
    {
      "test_flag": false,
      "input_data_path": "../data/inputfile",
      "label_data_path": "../data/labelfile",
      "epochs": 1000,
      "model_path": "../models/model",
      "hyperparameters": null
    }
    ```

3. **Training time estimation**  
Mimicking the tic/toc syntax in Matlab, elapsed time is recorded (inspired by [this answer](https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python)). ETA (Estimated Time of Arrival, _i.e._, rest running time.) will be calculated w.r.t. the last epoch time and rest number of epochs.

## Coding with this template

1. **Control flow in main()**  
```python
# System call parsing.
...
# Using context management for Tensorflow session.
with TFLearner(hyperparameters) as tfl:
    # Load data.
    tfl.load_data(input_data_path, label_data_path)
    # Build Tensorflow graph.
    tfl.build_graph()
    # Train model from scratch.
    tfl.train(num_epoch=num_epoch)
    # Or restore model to train.
    tfl.restore(model_path, num_epoch=num_epoch)
    # Or test a model.
    test_result = tfl.test_model(
        input_data_path=input_data_path, 
        label_data_path=label_data_path, 
        model_path=model_path)
```  
2. **Logging, and saving/restoring models**  
While training, logs should be written by [Tensorflow Summary Writer](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter), and checkpoints should be stored by [Tensorflow Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver).  
However, references to Tensors and Graphs will be lost, unless the [Tensor names are known](https://www.tensorflow.org/versions/r0.12/api_docs/python/framework/core_graph_data_structures), after restoring the [metagraph](https://www.tensorflow.org/api_guides/python/meta_graph). </br></br>
To solve this issue, hyper-parameters are stored together with checkpoints, with file extension _.hyperparameters_. To restore hyper-parameters, use python build-in [pickle.load](https://docs.python.org/3/library/pickle.html) function (or JSON). Tensorflow Graph should be rebuild w.r.t. hyper-parameters **before** restoring the model.
```python
# Example for restoring a model (taken from main.py).
class TFLearner(object):
    ...
    def restore(self, model_path, *, num_epoch=0):
    """Restore and continue training the network.

    Args:
        model_path (str): Path to the saved model files.
        num_epoch (`obj':int, optional): Number of epochs to train. Defaults to 0.

    Returns:
        None.
    """

    # Build graph.
    with open(model_path + ".hyperparams", "rb") as f:
        self._hyperparameters = pickle.load(f)
    self.build_graph()

    # Restore model.
    self._tfsaver.restore(self._tfsess, model_path)

    # Train if num_epoch > 0.
    if num_epoch > 0:
        self.train(num_epoch, init_flag=False)
```
```python
# Example for restoring a model from hyper-parameters.
with open("model.hyperparameters", "rb") as f:
    hyperparameters = pickle.load(f)

with TFLearner(hyperparameters) as tfl:
    ...
```

3. **Coding style, comment, and docstring**  
Use [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), or follow your own style.  


-- gl hf

