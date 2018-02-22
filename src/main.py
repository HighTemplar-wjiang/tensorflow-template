"""Template for Tensorflow projects.
Created by Weiwei Jiang on 20180222.

Usage:
python main.py config_file
    [See default.conf for Configuration file example.]

with TFLearner(hyperparameters) as tfl:
    ...
    tfl.train()
    ...
"""

import sys
import getopt
import os.path
import pickle
import numpy as np
import tensorflow as tf
from utils.tictoc import *


class TFLearner(object):
    """Learning something.

    Hyper-Parameters:
        hyper-param1: ...
        hyper-param2: ...

    Graph:
        0. Dimensions:
            Axis 0: Index of batch.
            Axis 1: Index of input/output.
            Axis 2: Index of feature map/intermediate values.
                E.g.: [i][j][k] returns a scalar of input/output j of feature map k in batch i.
        1. Input: ...
        2. ...
    """

    def __init__(self, *, hyperparameters=None):
        """Initialize the instance.

        Args:
             hyperparameters (:obj:`dict`): Dictionary of hyperparameters. Defaults to None.

        Attributes:
            _hyperparameters (dict): Hyper parameters for the network structure.
            _tfgrphs (dict): Tensorflow graphs.
            _data (dict): Data for training and testing.
            _tfsaver (tf.Saver): Tensorflow model saver.
            _tfsess (tf.Session): Tensorflow session. Initialize in __enter__.
        """
        if hyperparameters is None:
            self._hyperparameters = {
                # TODO: Default hyper-parameters.
            }
        else:
            self._hyperparameters = hyperparameters

        self._tfgraphs = {}  # Tensorflow graphs.
        self._tfsaver = None  # Tensorflow model saver.

        # Taining data and test data set.
        self._data = {
            "train_input": np.array([], dtype=np.float32),
            "train_label": np.array([], dtype=np.float32),
            "test_input": np.array([], dtype=np.float32),
            "test_label": np.array([], dtype=np.float32)
        }

    def __enter__(self):
        # Tensorflow session.
        self._tfsess = tf.Session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close Tensorflow session.
        self._tfsess.close()

    def build_graph(self):
        """Build tensor graph.
        """
        # TODO: Define Tensor graphs.

        # Clear built graphs.
        tf.reset_default_graph()

        print("INFO: Building Tensor graph.")

        # TODO: Unpack hyper-parameters.

        # TODO: Inputs and labels.
        # inputs = tf.placeholder(tf.float32, shape=(None, ?), name="InputData")
        # labels = tf.placeholder(tf.int32, shape=(None, ?), name="LabelData")

        # TODO: Learning rate.
        # learning_rate = tf.placeholder(tf.float32, shape=(), name="LearningRate")

        # TODO: Build Tensorflow model.
        with tf.name_scope("Model"):
            pass

        # TODO: Build output layer.
        with tf.name_scope("Output"):
            pass

        # TODO: Predict graph.
        with tf.name_scope("Predict"):
            pass

        # TODO: Loss function.
        with tf.name_scope("Loss"):
            pass
            # Add summary.
            # tf.summary.scalar("Loss_sum", loss_sum)

        # TODO: Define optimizer.
        with tf.name_scope("Train"):
            pass
            # train = tf.train.AdamOptimizer(
            #     learning_rate=learning_rate,
            #     beta1=0.9,
            #     beta2=0.999,
            #     name="Adam"
            # ).minimize(loss_sum)

        # TODO: Calculate accuracy.
        with tf.name_scope("Accuracy"):
            # Train set accuracy.
            pass
            # Add summary.
            # tf.summary.scalar("Train_accuracy", train_accuracy)
            # Merge all train summaries.
            # summary_train = tf.summary.merge_all()

            # Calculate test accuracy.
            # Note: Train and test summaries must be separated.
            # Test summary.
            # summary_test = tf.summary.scalar("Test_accuracy", test_accuracy)

        # Store to attributes.
        self._tfgraphs = {
            "input": None,
            "label": None,
            "predict": None,
            "loss": None,
            "learning_rate": None,
            "train": None,
            "accuracy": {
                "train": None,
                "test": None
            },
            "summary": {
                "train": None,
                "test": None
            }
        }

        # Instantiate a tensorflow saver.
        self._tfsaver = tf.train.Saver(max_to_keep=1000)

        # Restart session.
        self._tfsess = tf.Session()

        print("\tDone.")

    def load_data(self, input_data_path, label_data_path, *, test_ratio=0.1, otherparams=None):
        """Load training and testing data.

        Data format (for both inputs and labels):
            ...

        Args:
            input_data_path (str): Path to input data file.
            label_data_path (str): Path to label data file.
            test_ratio (float, optional): Ratio of data amount for testing. Defaults to 0.1.
            otherparams: ...

        Returns:
            None. The data will be stored to the object itself.
        """

        print("INFO: Loading data from {} and {}...".format(input_data_path, label_data_path))

        try:
            # Structure input data.
            with open(input_data_path, "r") as f:
                inputdata = f.read()
            # TODO: Organize input data.

            # Structure label data.
            with open(label_data_path, "r") as f:
                labeldata = f.read()
            # TODO: Organize label data.

            # TODO: Other process, i.e., normalizations.

            # TODO: Store data to attributes.
            self._data["train_input"] = None
            self._data["train_label"] = None
            self._data["test_input"] = None
            self._data["test_label"] = None

            # TODO: Print info.
            print("INFO: {} data loaded, {} for training, {} for testing.".format(0, 0, 0))

            return

        except FileNotFoundError:
            print("ERROR: File {} or {} not found!".format(input_data_path, label_data_path))

    def train(self, num_epoch: int, *, init_flag=True):
        """Train the network.

        Args:
            num_epoch (int): Number of epochs to train.
            init_flag (`obj`:bool): Flag for variable initialization, defaults to True.
        """

        # TODO: Hyper-parameters in running time.
        # learning_rate = self._hyperparameters["learning_rate"]

        # TODO: Get data info.
        batch_size = self._hyperparameters["batch_size"]
        num_train = self._data["train_input"].shape[0]
        num_batch = num_train // batch_size

        # TODO: op to write logs to Tensorboard.
        # summary_writer = tf.summary.FileWriter("../logs/", graph=tf.get_default_graph())

        # TODO: Get test data, reshape if necessary.
        # test_input = np.expand_dims(self._data["test_input"], axis=2)
        # test_label = np.expand_dims(self._data["test_label"], axis=2)

        # Initialize variables.
        if init_flag is True:
            print("INFO: Variables initialized.")
            self._tfsess.run(tf.global_variables_initializer())

        # Training.
        elapse_time_total = 0.0
        for idx_epoch in range(num_epoch):
            print("Epoch {}/{}".format(idx_epoch + 1, num_epoch))

            # _accuracy_train = 0.0
            elapse_time_epoch = 0.0
            for idx_batch in range(num_batch):
                print("Epoch {}/{}".format(idx_epoch + 1, num_epoch))
                print("\tBatch {}/{}".format(idx_batch + 1, num_batch))

                # Record time.
                tic()

                # TODO: Get batch data, reshape and/or shuffle if necessary.
                # idx_start = idx_batch * batch_size
                # idx_end = idx_start + batch_size
                # batch_input = np.expand_dims(self._data["train_input"][idx_start:idx_end, :], axis=2)
                # batch_label = np.expand_dims(self._data["train_label"][idx_start:idx_end, :], axis=2)

                # TODO: Training... Good luck!
                # _, _loss, _summary_train = self._tfsess.run(
                #     [self._tfgraphs["train"],
                #      self._tfgraphs["loss"],
                #      self._tfgraphs["summary"]["train"]],
                #     feed_dict={
                #         self._tfgraphs["input"]: batch_input,
                #         self._tfgraphs["label"]: batch_label,
                #         self._tfgraphs["learning_rate"]: learning_rate
                #     })

                # TODO: Get train accuracy.
                # _predicts_train, _accuracy_train = self._tfsess.run([self._tfgraphs["predict"],
                #                                                      self._tfgraphs["accuracy"]["train"]],
                #                                                     feed_dict={
                #                                                         self._tfgraphs["input"]: batch_input,
                #                                                         self._tfgraphs["label"]: batch_label,
                #                                                         self._tfgraphs["learning_rate"]: learning_rate
                #                                                     })
                # TODO: Get test accuracy.
                # _predicts_test, _accuracy_test, _summary_test = self._tfsess.run(
                #     [self._tfgraphs["predict"],
                #      self._tfgraphs["accuracy"]["test"],
                #      self._tfgraphs["summary"]["test"]],
                #     feed_dict={
                #         self._tfgraphs["input"]: test_input,
                #         self._tfgraphs["label"]: test_label,
                #         self._tfgraphs["learning_rate"]: cur_rate
                #     })

                # Record time.
                elapse_time_batch = toc()
                elapse_time_epoch += elapse_time_batch
                elapse_time_total += elapse_time_batch
                print("\tBatch time: {}".format(seconds_to_readable(elapse_time_batch)))
                print("\tEpoch time: {}".format(seconds_to_readable(elapse_time_epoch)))
                print("\tTotal time: {}".format(seconds_to_readable(elapse_time_total)))
                print("\tETA: {}\n".format(seconds_to_readable(
                    elapse_time_batch * num_batch * num_epoch - elapse_time_total)))

                # print("\tTrain accuracy: {}\n"
                #       "\tPredict sample:\n"
                #       "\tInput:  {}\n"
                #       "\tLabel:  {}\n"
                #       "\tSoftmax:\n{}".format(_accuracy_train, batch_input[0, :3, 0],
                #                               batch_label[0, :3, 0], np.array(_predicts_train)[:3, 0, :]))
                #
                # print("\tTest accuracy: {}\n"
                #       "\tPredict sample:\n"
                #       "\tInput:  {}\n"
                #       "\tLabel:  {}\n"
                #       "\tSoftmax:\n{}".format(_accuracy_test, test_input[0, :3, 0],
                #                               test_label[0, :3, 0], np.array(_predicts_test)[:3, 0, :]))

                # TODO: Write logs.
                # if idx_batch % 20 == 0:
                #     # Write log every 10 batches.
                #     summary_writer.add_summary(_summary_train, idx_epoch * num_batch + idx_batch)
                #     summary_writer.add_summary(_summary_test, idx_epoch * num_batch + idx_batch)

                # print("Loss: " + str(_loss))

            # TODO: Adaptive learning rate, if necessary.
            #   New rate = current rate * multiplier
            # error_rate_train = 1.0 - _accuracy_train
            # multiplier = 2.0 ** (-np.floor(-np.log10(error_rate_train)))
            # # multiplier = 1.0
            # cur_rate = learning_rate * multiplier

            # TODO: Save checkpoints periodically. Also dump hyper-parameters for restoring model if needed.
            # if idx_epoch % 10 == 0:
            #     print("INFO: Saving checkpoint.")
            #     self._tfsaver.save(self._tfsess, "../models/model_{:d}".format(idx_epoch))
            #     with open("../models/model_{:d}.hyperparams".format(idx_epoch), "wb") as f:
            #         pickle.dump(self._hyperparameters, f)

        # TODO: Save the model at the end of training.
        # print("INFO: Saving checkpoint.")
        # self._tfsaver.save(self._tfsess, "../models/model_{:d}".format(num_epoch))
        # with open("../models/model_{:d}.hyperparams".format(num_epoch), "wb") as f:
        #     pickle.dump(self._hyperparameters, f)

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

    def test_model(self, *, input_data_path=None, label_data_path=None, model_path=None,
                   test_ratio=1.0, otherparams=None):
        """Restore and test a model.

        Args:
            input_data_path (str, optional): Path to input data file. Defaults to None, use data stored in attributes.
            label_data_path (str, optional): Path to label data file. Defaults to None, use data stored in attributes.
            model_path (str, optional): Path to the saved model files. Defaults to None, use model stored in attributes.
            test_ratio (float, optional): Ratio of data amount for testing. Defaults to 1.0.
            otherparams: ...

        Returns:
            Dict: A dictionary of dictionary that contains:
                {"accuracy": float, "loss": float}
        """

        # TODO: Load test file.
        if input_data_path is not None and label_data_path is not None:
            self.load_data(input_data_path, label_data_path, test_ratio=test_ratio)

        # TODO: Get test data, reshape if necessary.
        # test_input = np.expand_dims(self._data["test_input"], axis=2)
        # test_label = np.expand_dims(self._data["test_label"], axis=2)

        # TODO: Get test accuracy.
        # _predicts_test, _accuracy_test, _loss_test = self._tfsess.run(
        #     [self._tfgraphs["predict"],
        #      self._tfgraphs["accuracy"]["test"],
        #      self._tfgraphs["loss"]],
        #     feed_dict={
        #         self._tfgraphs["input"]: test_input,
        #         self._tfgraphs["label"]: test_label
        #     })

        # TODO: Dictionary to return as results.
        result_dict = {"accuracy": None, "cost": None}

        return result_dict


def main(argv):
    """Parse system call arguments and execute."""

    print("Main function received arguments: {}.".format(argv))

    command_str = "main.py -i <inputfile> -l <labelfile> [-m <modelpath> -e <epochs> | -t -m <modelpath>] "

    # Arguments for running.
    test_flag = False
    input_data_path = ""
    label_data_path = ""
    model_path = ""
    num_epochs = 1000
    config_file_path = "./default.conf"  # Default config file.
    hyperparameters = None

    if len(argv) < 2:
        # Using config file.
        if len(argv) == 1:
            config_file_path = argv[0]

        with open(config_file_path, "r") as f:
            config = json.load(f)
        # Parse configs.
        input_data_path = config["input_data_path"]
        label_data_path = config["label_data_path"]
        model_path = config["model_path"]
        test_path = config["test_path"]
        num_epochs = int(config["epochs"])
        hyperparameters = config["hyperparameters"]
    else:
        try:
            opts, args = getopt.getopt(argv, "ti:l:m:e:", ["test", "inputfile=", "labelfile=", "modelpath=", "epochs="])
        except getopt.GetoptError:
            print(command_str)
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-i", "--inputfile"):
                if os.path.isfile(arg):
                    input_data_path = arg
                else:
                    print("ERROR: File {} does not exist!".format(arg))
                    sys.exit()
            elif opt in ("-l", "--labelfile"):
                if os.path.isfile(arg):
                    label_data_path = arg
                else:
                    print("ERROR: File {} does not exist!".format(arg))
                    sys.exit()
            elif opt in ("-t", "--test"):
                test_flag = True
            elif opt in ("-m", "--modelpath"):
                if os.path.isfile(arg + ".index"):
                    model_path = arg
                else:
                    print("ERROR: Model {} does not exist!".format(arg))
            elif opt in ("-e", "--epochs"):
                try:
                    num_epochs = int(arg)
                except ValueError:
                    print("ERROR: Epoch value {} is not an integer!".format(arg))
            else:
                print(command_str)
                sys.exit()

    # Model path must be indicated for testing.
    if test_flag and model_path == "":
        print("ERROR: Test model path required!")

    with TFLearner(hyperparameters=hyperparameters) as cl:
        # Load data.
        cl.load_data(input_data_path, label_data_path)

        # Not testing.
        if not test_flag:
            if model_path == "":
                # Train from scratch.
                cl.build_graph()
                cl.train(num_epochs)
            else:
                # Restore model and continue training.
                cl.restore(model_path, num_epoch=num_epochs)
        else:  # Testing
            cl.restore(model_path, num_epoch=0)
            test_result = cl.test_model(
                input_data_path=input_data_path, label_data_path=label_data_path, model_path=model_path)
            print(test_result)


if __name__ == "__main__":
    import json

    main(sys.argv[1:])
