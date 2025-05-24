import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GlobalAveragePooling2D, Concatenate, MaxPooling2D, AvgPool2D
from tensorflow.keras import Model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121


# ---------------------------------------------------------------------------
class PretrainedModels:
    """
    Class that contains the six pretrained models, and a function for choosing their training strategy.

    Possible training strategies:
    1. 'no'         : the whole model is frozen -> no parameters are updated during the optimization process
    2. 'last_block' : the whole model is frozen, except the last block -> only the last block's parameters are updated
                      during the optimization process
    3. 'all'        : the whole model is unfrozen -> all parameters are updated during the optimization process
    """

    @staticmethod
    def pretraining_models_method(pretrained_model, model_name, training_basemodel='last_block'):
        """
        Function for unfreezing the pre chosen layers of the pretrained base CNN model.

        :param pretrained_model: tensorflow.keras.applications, the current pretrained base CNN model
        :param model_name: str, the name of the pretrained base CNN model
        :param training_basemodel: str, the training strategy of the pretrained base CNN model
        :return: tensorflow.keras.applications, the current pretrained base CNN model after unfreezing
                                                (or freezing) its layers
        """

        # First choice: freezing the whole pretrained model
        if training_basemodel == 'no':
            # Freezing the base model
            pretrained_model.trainable = False

        # Second choice: unfreezing only the last block of the pretrained model
        elif training_basemodel == 'last_block':

            # The names of the layers contain the index of the block that they are a part of, so we can save the
            # starting index of each block in a list

            # List with the indices of the starting layers for every block
            blocks_index = []

            # Starting at block 1 we keep the index of the starting layer at each block
            block = 1
            for i, layer in enumerate(pretrained_model.layers):

                if model_name in ['resnet50', 'resnet50v2', 'densenet121']:

                    if f'conv{block}' in layer.name:
                        blocks_index.append(i)
                        block += 1

                elif model_name in ['vgg16', 'vgg19']:

                    if f'block{block}' in layer.name:
                        blocks_index.append(i)
                        block += 1

                elif model_name in ['inception_v3']:

                    if 'mixed9_0' in layer.name:
                        blocks_index.append(i)
                        break

            # Finding the last blocks starting index (last index saved)
            last_block_index = blocks_index[-1]

            # Freezing every layer except the last block
            for layer in pretrained_model.layers[:last_block_index]:
                layer.trainable = False

        # Third choice: unfreezing the whole pretrained model
        elif training_basemodel == 'all':
            pretrained_model.trainable = True

        return pretrained_model

    @staticmethod
    def resnet50(input_image, training_basemodel='last_block'):
        """
        Function returning the ResNet50 architecture, as presented in the original paper, after removing the
        pretrained top layer.

        This model uses a preprocessing function that convert the input images from RGB to BGR, then will zero-center
        each color channel with respect to the ImageNet dataset, without scaling.

        :param input_image: tensorflow.keras.layers.Input, the input tensor for the ResNet50
        :param training_basemodel: str, the training strategy of the model
        :return: the base model
        """

        # Setting the pretrained model
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=input_image,
        )

        # Unfreezing (or freezing) the base model's layers
        base_model = PretrainedModels.pretraining_models_method(base_model, 'resnet50', training_basemodel)

        # Adding a global average pooling layer
        base_model = GlobalAveragePooling2D()(base_model.output)

        return base_model

    @staticmethod
    def resnet50v2(input_image, training_basemodel='last_block'):
        """
        Function returning the ResNet50 architecture, as presented in the original paper, after removing the
        pretrained top layer.

        This model uses a preprocessing function that scales the inputs image's intensities between -1 and 1,
        sample-wise.

        :param input_image: tensorflow.keras.layers.Input, the input tensor for the ResNet50
        :param training_basemodel: str, the training strategy of the model
        :return: the base model
        """

        # Setting the pretrained model
        base_model = ResNet50V2(
            weights="imagenet",
            include_top=False,
            input_tensor=input_image,
        )

        # Unfreezing (or freezing) the base model's layers
        base_model = PretrainedModels.pretraining_models_method(base_model, 'resnet50v2',training_basemodel)

        # Adding a global average pooling layer
        base_model = GlobalAveragePooling2D()(base_model.output)

        return base_model

    @staticmethod
    def vgg16(input_image, training_basemodel='last_block'):
        """
        Function returning the VGG16 architecture, as presented in the original paper, after removing the
        pretrained top layer.

        This model uses a preprocessing function that convert the input images from RGB to BGR, then will zero-center
        each color channel with respect to the ImageNet dataset, without scaling.

        :param input_image: tensorflow.keras.layers.Input, the input tensor for the VGG16
        :param training_basemodel: str, the training strategy of the model
        :return: the base model
        """

        # Setting the pretrained model
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_tensor=input_image,
        )

        # Unfreezing (or freezing) the base model's layers
        base_model = PretrainedModels.pretraining_models_method(base_model, 'vgg16', training_basemodel)

        # Adding a global average pooling layer
        base_model = GlobalAveragePooling2D()(base_model.output)

        return base_model

    @staticmethod
    def vgg19(input_image, training_basemodel='last_block'):
        """
        Function returning the VGG19 architecture, as presented in the original paper, after removing the
        pretrained top layer.

        This model uses a preprocessing function that convert the input images from RGB to BGR, then will zero-center
        each color channel with respect to the ImageNet dataset, without scaling.

        :param input_image: tensorflow.keras.layers.Input, the input tensor for the VGG19
        :param training_basemodel: str, the training strategy of the model
        :return: the base model
        """

        # Setting the pretrained model
        base_model = VGG19(
            weights="imagenet",
            include_top=False,
            input_tensor=input_image,
        )

        # Unfreezing (or freezing) the base model's layers
        base_model = PretrainedModels.pretraining_models_method(base_model, 'vgg19', training_basemodel)

        # Adding a global average pooling layer
        base_model = GlobalAveragePooling2D()(base_model.output)

        return base_model

    @staticmethod
    def inception_v3(input_image, training_basemodel='last_block'):
        """
        Function returning the VGG19 architecture, as presented in the original paper, after removing the
        pretrained top layer.

        This model uses a preprocessing function that scales the inputs image's intensities between -1 and 1,
        sample-wise.

        :param input_image: tensorflow.keras.layers.Input, the input tensor for the VGG19
        :param training_basemodel: str, the training strategy of the model
        :return: the base model
        """

        # Setting the pretrained model
        base_model = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_tensor=input_image,
        )

        # Unfreezing (or freezing) the base model's layers
        base_model = PretrainedModels.pretraining_models_method(base_model, 'inception_v3', training_basemodel)

        # Adding a global average pooling layer
        base_model = GlobalAveragePooling2D()(base_model.output)

        return base_model

    @staticmethod
    def densenet121(input_image, training_basemodel='last_block'):
        """
        Function returning the VGG19 architecture, as presented in the original paper, after removing the
        pretrained top layer.

        This model uses a preprocessing function that scales the inputs image's intensities between -1 and 1,
        sample-wise.

        :param input_image: tensorflow.keras.layers.Input, the input tensor for the VGG19
        :param training_basemodel: str, the training strategy of the model
        :return: the base model
        """

        # Setting the pretrained model
        base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_tensor=input_image,
        )

        # Unfreezing (or freezing) the base model's layers
        base_model = PretrainedModels.pretraining_models_method(base_model, 'densenet121', training_basemodel)

        # Adding a global average pooling layer
        base_model = GlobalAveragePooling2D()(base_model.output)

        return base_model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def BaseModel(input_image, pretrained_model='resnet50', training_basemodel='last_block'):
    """
    Function for fetching the chosen pretrained model and applying its training method.

    :param input_image: tensorflow.keras.layers.Input, the input tensor for the pretrained base model
    :param pretrained_model: str, the name of the chosen model
    :param training_basemodel: str, the training method to be applied
    :return: the base model
    """

    # Dictionary connecting the available models with their respective names
    pretrained_models = {
        'resnet50': PretrainedModels.resnet50,
        'resnet50v2': PretrainedModels.resnet50v2,
        'vgg16': PretrainedModels.vgg16,
        'vgg19': PretrainedModels.vgg19,
        'inception_v3': PretrainedModels.inception_v3,
        'densenet121': PretrainedModels.densenet121,
    }

    # The fetched chosen model
    base_model = pretrained_models[pretrained_model](input_image, training_basemodel)

    return base_model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def Multi_Input_Model(
        optimizer,
        width=224,
        height=224,
        n_features=6,
        pretrained_model='resnet50',
        training_basemodel='last_block',
        num_classes=3,
):
    """
    The function returns the compiled multi input model using 2D slices from MRIs and their respective metadata.

    :param optimizer: tensorflow.keras.optimizers, the chosen optimizer

    :param width: int, the input images' width used in the CNN model
    :param height: int, the input images' height used in the CNN model

    :param n_features: int, the number of features/metadata of the MRIs used in the ANN model

    :param pretrained_model: str, the name of the pretrained base model used in the CNN model for feature extraction
    :param training_basemodel: str, the training method of the base model

    :param num_classes: int, the number of classes

    :return: the compiled model
    """
    # --- Defining the input layers ---
    input_image = Input(shape=(width, height, 3))   # MRI input
    input_metadata = Input(shape=(n_features,))     # MRI metadata input

    # --- Defining the models for the different inputs ---

    # Defining the pretrained CNN model for the MRI data
    pretrained_base_model = BaseModel(
        input_image=input_image,
        pretrained_model=pretrained_model,
        training_basemodel=training_basemodel
    )

    # Adding a dense layer on top
    pretrained_base_model = Dense(128, activation='relu')(pretrained_base_model)

    # Defining the ANN model for the metadata
    metadata_model = Dense(128, activation='relu')(input_metadata)

    # --- Concatenating the outputs of the two models ---
    concatenated = Concatenate()([pretrained_base_model, metadata_model])

    # --- Adding classification layers ---
    flatten = Flatten()(concatenated)
    dense = Dense(1024, activation='relu')(flatten)
    dense = Dropout(0.2)(dense)
    dense = Dense(512, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation='relu')(dense)

    # --- Configuring output layer and loss function according to the number of classes ---

    if num_classes > 2:
        output_configuration = {
            'units': num_classes,
            'activation': 'softmax'
        }
        loss_function = 'sparse_categorical_crossentropy'
    else:
        output_configuration = {
            'units': 1,
            'activation': 'sigmoid'
        }
        loss_function = 'binary_crossentropy'

    # --- Setting
    output = Dense(**output_configuration)(dense)

    # --- Creating the model ---
    model = Model(inputs=[input_image, input_metadata], outputs=output)

    # --- Compiling the model ---
    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

    return model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def learning_rate_schedule(epoch, lr, min_lr=5e-5):
    """
    Function for scheduling learning rate during training

    :param epoch: int, the current epoch
    :param lr: float, the initial learning rate
    :param min_lr: float, the minimum learning rate during training
    :return: float, the learning rate for the current epoch
    """
    print(100*'-')
    print(f"Current learning rate = {lr:.2e}")
    print(100*'-')

    # Schedule:
    # 1. Learning rate must not be smaller than the given one
    if lr < min_lr:
        return lr
    # 2. Every 5 epochs it is decayed exponentially
    elif epoch % 5 == 0:
        return lr * tf.math.exp(-0.1)
    # 3. Return the original lr in case of none of the above
    else:
        return lr
# ---------------------------------------------------------------------------
