import os
import time
import traceback
from typing import Tuple, List, Optional

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.python.keras.layers import Flatten, Dropout, BatchNormalization

DATA_FOLDER = "data"
OUTPUT_FOLDER = os.getcwd() + "/output"


ENABLE_GPU = False
if ENABLE_GPU:
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def define_model(layers: List[int], img_height: int, img_width: int,
                 data_augmentation_layers: Optional[List[PreprocessingLayer]] = None, dropout: Optional[int] = None,
                 batch_normalization: bool = False, activation_function: str = "relu"):
    """
    Creates the CNN model.

    :param layers: A list of filter sizes. For each entry, an additional Conv2D layers
     with the specified filter and a MaxPooling2D layer will be added.
    :param img_height: The height of each image. Images that do not match this weight will be rescaled.
    :param img_width: The width of each image. Images that do not match this width will be rescaled.
    :param data_augmentation_layers: The list of data augmentation strategies to apply.
    :param dropout: The dropout value to use [0, 1]. 0 to disable.
    :param batch_normalization: Whether to apply batch normalization to each combination.
    :param activation_function: The activation function to use in the final layer.
    :return: The compiled model.
    """
    print("Creating CNN...")

    model = Sequential()

    if data_augmentation_layers is not None:
        for data_augmentation_layer in data_augmentation_layers:
            model.add(data_augmentation_layer)

    # Scale the input images to [0-1] instead of [0-255] to improve performance a bit.
    model.add(preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)))

    for filter_size in layers:
        model.add(Conv2D(filters=filter_size, kernel_size=3, activation="relu", padding="SAME"))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(MaxPooling2D())

    if dropout is not None:
        model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(128, activation="relu")),

    model.add(Dense(10, activation=activation_function))

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    return model


def read_data(img_height: int, img_width: int, batch_size: int) \
        -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Reads the dataset.

    :param img_height: The height of the images. Scaling is used in case the actual input height is not the same.
    :param img_width: The width of the images. Scaling is used in case the actual input width is not the same.
    :param batch_size: The batch size to use.
    :return: The training, validation, and test datasets in a 60/20/20 split.
    """

    dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_FOLDER,
        validation_split=0.4,
        subset="training",
        seed=12345,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    dataset_validation_test = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_FOLDER,
        validation_split=0.4,
        subset="validation",
        seed=12345,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Gets the total number of batches in the validation/test set.
    validation_test_size = tf.data.experimental.cardinality(dataset_validation_test).numpy()
    validation_size = int(validation_test_size / 2)

    print("validation_test_size: {}, validation_size: {}".format(validation_test_size, validation_size))

    # Take the first half of the validation/test set for the validation set
    # And the second half for the test set.
    dataset_validation = dataset_validation_test.take(validation_size)
    dataset_test = dataset_validation_test.skip(validation_size)

    print("validation size: {}, test size: {}".format(
        tf.data.experimental.cardinality(dataset_validation).numpy(),
        tf.data.experimental.cardinality(dataset_test).numpy()))

    return dataset_train, dataset_validation, dataset_test


def run_model(model: Sequential, dataset_train: tf.data.Dataset, dataset_validation: tf.data.Dataset,
              dataset_test: tf.data.Dataset, batch_size: int, epochs: int, output_dir: Optional[str] = None)\
        -> int:
    """
    Trains and tests the provided model.

    :param model: The model to train and test.
    :param dataset_train: The dataset to use for the training phase.
    :param dataset_validation: The dataset to use for validation during the training phase.
    :param dataset_test: The dataset to use for testing the accuracy of the model.
    :param batch_size: The batch size to use for training and testing the model
    :param epochs: The number of epochs to train the model for.
    :param output_dir: The directory to store the results in.
    :return: The final accuracy of the model.
    """

    print("Fitting model...")
    history = model.fit(dataset_train, validation_data=dataset_validation, batch_size=batch_size,
                        epochs=epochs, verbose=1)

    print("Model fitted!")
    print(model.summary())
    if output_dir is not None:
        f = open(output_dir + "/model_summary", "w")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.close()

    score, accuracy = model.evaluate(dataset_test, batch_size=batch_size)

    print("Test Score: ", score)
    print("Test Accuracy: ", accuracy)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")

    if output_dir is not None:
        plt.savefig(output_dir + "/plot.pdf")
        f = open(output_dir + "/results", "w")
        f.write("Accuracy = {}\n".format(accuracy))
        f.write("Score = {}".format(score))
        f.close()
    else:
        plt.show()

    return accuracy


def get_data_augmentation_layers(rotation: bool = False, flip: bool = False,
                                 zoom: bool = False, contrast: bool = False) -> List[PreprocessingLayer]:
    """
    Creates a list of augmentation layers which can be applied later.

    :param rotation: Data Augmentation: Whether to apply random rotation to the images.
    :param flip: Data Augmentation: Whether to apply random horizontal flip to the images.
    :param zoom: Data Augmentation:  Whether to apply random zoom to the images.
    :param contrast: Data Augmentation: Whether to apply random contrast enhancement to the images.
    :return: The list of data augmentation layers.
    """
    data_augmentation = []
    if rotation:
        data_augmentation.append(preprocessing.RandomRotation(factor=(1/6)))  # Between +/- 30deg
    if flip:
        data_augmentation.append(preprocessing.RandomFlip("horizontal"))
    if zoom:
        data_augmentation.append(preprocessing.RandomZoom(height_factor=0.2))  # Zoom +/- 20%
    if contrast:
        data_augmentation.append(preprocessing.RandomContrast(factor=0.1))

    return data_augmentation


def run_experiment(batch_size: int, epochs: int, img_height: int, img_width: int, output_dir: Optional[str] = None,
                   rotation: bool = False, flip: bool = False, zoom: bool = False,
                   contrast: bool = False, dropout: Optional[int] = None, activation_function: str = 'relu',
                   batch_normalization: bool = False, layers: List[int] = None) -> int:
    """
    Runs the experiment using the provided configuration.

    :param batch_size: The batch size to use.
    :param epochs: The number of epochs to train the model for.
    :param img_height: The height of the images. Scaling is used in case the actual input height is not the same.
    :param img_width: The width of the images. Scaling is used in case the actual input width is not the same.
    :param output_dir: The directory to store the results in.
    :param rotation: Data Augmentation: Whether to apply random rotation to the images.
    :param flip: Data Augmentation: Whether to apply random horizontal flip to the images.
    :param zoom: Data Augmentation:  Whether to apply random zoom to the images.
    :param contrast: Data Augmentation: Whether to apply random contrast enhancement to the images.
    :param dropout: The dropout value to use for the last layer. Must be within [0-1] or None to disable it.
    :param activation_function: The activation function to use for the last dense layer. E.g. 'relu'.
    :param batch_normalization: Whether to apply batch normalization.
    :param layers: An array of layer sizes. Each entry creates a new layer combination with the number of filters.
    :return: The final accuracy of the model.
    """

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        f = open(output_dir + "/config", "w")

        f.write("\"epochs\": {},\n"
                "\"batch_size\": {},\n"
                "\"img_height\": {},\n"
                "\"img_width\": {},\n"
                "\"layers\": [{}],\n"
                "\"rotation\": {},\n"
                "\"flip\": {},\n"
                "\"zoom\": {},\n"
                "\"contrast\": {},\n"
                "\"dropout\": {},\n"
                "\"activation_function\": \"{}\",\n"
                "\"batch_normalization\": {}\n".format(epochs, batch_size, img_height, img_width,
                                                       ', '.join([str(layer) for layer in layers]),
                                                       rotation, flip, zoom, contrast,
                                                       dropout, activation_function, batch_normalization))
        f.close()

    dataset_train, dataset_validation, dataset_test = read_data(batch_size=batch_size,
                                                                img_height=img_height, img_width=img_width)

    model = define_model(layers, img_height=img_height, img_width=img_width,
                         data_augmentation_layers=get_data_augmentation_layers(flip=flip, zoom=zoom,
                                                                               contrast=contrast, rotation=rotation),
                         dropout=dropout, activation_function=activation_function,
                         batch_normalization=batch_normalization)

    return run_model(model, dataset_train, dataset_validation, dataset_test, output_dir=output_dir,
                     epochs=epochs, batch_size=batch_size)


def main():
    batch_size = 128
    epochs = 40
    img_height = 256
    img_width = 256


    #-------Perceptron implementation-----------#


    configs = [
        {  # 0
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "relu", "batch_normalization": False
        },

        {  # 1
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": 0.2, "activation_function": "relu", "batch_normalization": False
        },
        {  # 2
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "relu", "batch_normalization": False
        },
        {  # 3
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": 0.4, "activation_function": "relu", "batch_normalization": False
        },
        {  # 4
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": 0.5, "activation_function": "relu", "batch_normalization": False
        },
        {  # 5
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "relu", "batch_normalization": True
        },

        {  # 6
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": True, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "relu", "batch_normalization": False
        },
        {  # 7
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": True, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "relu", "batch_normalization": False
        },
        {  # 8
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": True, "contrast": False,
            "dropout": None, "activation_function": "relu", "batch_normalization": False
        },
        {  # 9
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": True,
            "dropout": None, "activation_function": "relu", "batch_normalization": False
        },

        {  # 10
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "sigmoid", "batch_normalization": False
        },
        {  # 11
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "elu", "batch_normalization": False
        },
        {  # 12
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "softmax", "batch_normalization": False
        },
        {  # 13
            "epochs": epochs, "batch_size": batch_size, "img_height": img_height, "img_width": img_width, "layers": [32, 64, 128],
            "rotation": False, "flip": False, "zoom": False, "contrast": False,
            "dropout": None, "activation_function": "tanh", "batch_normalization": False
        },
    ]


    # Every time we run this, we store all results in their own subdirectory.
    # This way we won't override any previous experiments.
    existing_subdir_count = 0
    for folder in os.listdir(OUTPUT_FOLDER):
        if folder.startswith("run_"):
            existing_subdir_count += 1
    output_subdir = OUTPUT_FOLDER + "/run_" + str(existing_subdir_count)

    accuracies = []
    times = []

    for idx, entry in enumerate(configs):
        output_dir = output_subdir + "/" + str(idx)
        start_time = time.time()
        try:
            tf.keras.backend.clear_session()
            tf.random.set_seed(12345)

            accuracies.append(run_experiment(output_dir=output_dir, **entry))
        except Exception as e:
            with open(output_dir + "/error", "w") as f:
                stacktrace = traceback.format_exc()
                f.write(stacktrace)
                f.close()

                print(stacktrace)

            accuracies.append(-1)

        times.append(time.time() - start_time)

    with open(output_subdir + "/comparison", "w") as f:
        for idx, result in enumerate(zip(accuracies, times)):
            line = "Configuration: {:>2}, Accuracy: {:1.16f}, Duraction: {:>6} s".format(idx, result[0], int(result[1]))
            print(line)
            f.write(line)
            f.write("\n")


if __name__ == "__main__":
    main()
