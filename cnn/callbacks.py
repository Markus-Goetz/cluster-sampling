from keras.callbacks import Callback
import numpy as np


DELIMITER = '\t'


class TrackTrainHistory(Callback):
    """
    Callback class that tracks training accuracy and loss for each epoch
    """
    def __init__(self, out_file):
        """
        Constructor for the TrackTrainHistory callback.

        :param out_file: str, target location of the history CSV file.
        """
        Callback.__init__(self)

        self.accuracies = []
        self.losses = []
        self.out_file = out_file

    def on_epoch_end(self, epoch, logs=None):
        """
        Override of the parent hook that is called after an epoch ends. Extracts the training loss and target metric and
        stores the entire history in the output file.

        :param epoch:   int, incremental number of the current epoch number
        :param logs:    dict, the training history of loss and target metric
        """
        self.accuracies.append(float(logs.get('acc')))
        self.losses.append(float(logs.get('loss')))

        # write the history to disk
        np.savetxt(self.out_file, np.vstack((self.accuracies, self.losses,)).T, delimiter=DELIMITER)


class TrackTestHistoryAndModel(Callback):
    """
    Callback to benchmark the trained model, i.e. determine the target metric, for every n-th epoch.
    Saves the performance history and the model weights, given that they outperform the previous version.
    """
    def __init__(self, test_generator, test_batches, weights_file, history_file, n=10):
        """
        Constructor for the TrackTestHistoryAndModel callback.

        :param test_generator:  generator, iterator that generates batched test data image patches.
        :param test_batches:    int, number of complete batches in the generator.
        :param weights_file:    str, target path for the model weights/parameters to be stored in HDF5 format.
        :param history_file:    str, target path for the history CSV file.
        :param n:               int, marks after how many epochs the model is being evaluated on the test data.
        """
        Callback.__init__(self)

        self.test_generator = test_generator
        self.test_batches = test_batches
        self.weights_file = weights_file
        self.history_file = history_file
        self.n = n

        self.accuracies = []
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        """
        Override of the parent hook that is called after an epoch ends. Extracts the training loss and target metric and
        stores the entire history in the output file. Additionally tests the performance of the model on the test data
        every n-th epoch.

        :param epoch:   int, incremental number of the current epoch number
        :param logs:    dict, the training history of loss and target metric
        """
        # evaluate the model only for every n-th epoch
        if epoch % self.n != self.n - 1:
            return

        # calculate new test loss and test accuracy:
        new_loss, new_accuracy = self.model.evaluate_generator(self.test_generator, steps=self.test_batches, verbose=1)

        # store the model weights in case we have not yet stored any or the new model is better
        if not self.losses or new_loss < np.min(self.losses):
            self.model.save_weights(self.weights_file)

        # remember the obtained accuracy and loss
        self.accuracies.append(new_accuracy)
        self.losses.append(new_loss)

        # write the history to disk
        np.savetxt(self.history_file, np.vstack((self.accuracies, self.losses,)).T, delimiter=DELIMITER)
