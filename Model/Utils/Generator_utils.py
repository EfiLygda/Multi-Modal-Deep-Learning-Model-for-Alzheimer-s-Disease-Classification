import numpy as np

# ---------------------------------------------------------------------------
def images_generator(generator, df, batch_size, target_width, target_height):
    """
    Wrapper function for the 'flow_from_dataframe' of keras by passing the ImageDataGenerator, used for splitting the
    images in batches and using the original generator for augmentation.

    :param generator: tensorflow.python.keras.preprocessing.image.ImageDataGenerator, the generator to wrap
    :param df: pandas.Dataframe, the labeled images dataset
    :param batch_size: int, the batch size
    :param target_width: int, the images' target width
    :param target_height: int, the images' target height

    :return: ImageDataGenerator.flow_from_dataframe, the generator for splitting the images in batches
    """
    return generator.flow_from_dataframe(
                        df,
                        x_col='image_path',
                        y_col='label',
                        target_size=(target_width, target_height),
                        color_mode='rgb',
                        class_mode='raw',  # labels already encoded
                        batch_size=batch_size,
                        shuffle=False,
                    )
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
class MetadataGenerator:

    def __init__(self, df, batch_size, mode='train'):
        """
        Generator for iterating the metadata df in batches

        :param df: pandas.Dataframe, the metadata dataframe
        :param batch_size: int, the batch size
        :param mode: str, the name of the subset (train, val, test)
        """
        self.df = df
        self.batch_size = batch_size
        self.index = 0  # Batch index
        self.mode = mode
        self.total_batches = int(np.ceil(len(self.df) / float(batch_size)))  # Total number of batches

    def __iter__(self):  # For iterating the generator
        return self

    def __next__(self):  # For fetching the next batch, when iterating

        # --- Automatically reset the generator ---
        # So that it will be used in each epoch during training, without raising an error

        # In case the batch index is bigger than the total batches, the index is reset to 0
        if self.index >= self.total_batches:
            self.index = 0

        # --- Start and end index in the metadata dataframe for the current batch ---

        # The starting index of the current batch
        start_index = self.index * self.batch_size

        # The ending index of the current batch
        # In case the last batch has smaller size than the batch size the last row's index is used
        end_index = min(start_index + self.batch_size, len(self.df))

        # --- Separating the data from the labels ---

        # Current batch's data
        batch_x = self.df.iloc[start_index:end_index, 1:-1].to_numpy()  # removing the 1st column with image id
                                                                        # and the last with the label
        # Current batch's labels
        batch_y = self.df.iloc[start_index:end_index]['label'].to_numpy()

        # --- Increase the batch index ---
        self.index += 1

        return batch_x, batch_y

    def __len__(self):
        return self.total_batches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
class FlowFromDataframes:

    # WARNINGS:

    # 1) A batch will be loaded before the end of the epoch so that the model will not be slowed down, but no batch
    # will be used more than one time during an epoch: https://stackoverflow.com/a/53780042

    # 2) The messages cannot be disabled: https://stackoverflow.com/a/65351797

    def __init__(self, aug_generator, images_df, metadata_df, batch_size, target_width, target_height, mode):
        """
        Generator for iterating the dataframes used in the multi input dataset in batches

        :param aug_generator: tensorflow.python.keras.preprocessing.image.ImageDataGenerator.flow_from_dataframe,
                              the iterator for the labeled images' batches

        :param images_df: pandas.Dataframe, the labeled images' data frame
        :param metadata_df: pandas.Dataframe, the metadata data frame

        :param batch_size: int, the batch size

        :param target_width: int, the images' target width
        :param target_height: int, the images' target height

        :param mode: str, the subset of the dataset (train, val, test)
        """

        self.images_df = images_df
        self.metadata_df = metadata_df

        self.target_width = target_width
        self.target_height = target_height

        self.mode = mode

        # Image augmentation generator
        self.aug_generator = aug_generator

        # Images generator for iteration in batches
        self.images_generator = images_generator(aug_generator, images_df,
                                                 batch_size,
                                                 target_width, target_height)

        # Metadata generator for iteration in batches
        self.metadata_generator = MetadataGenerator(metadata_df, batch_size, mode)

        self.batch_size = batch_size
        self.index = 0  # Batch index
        self.total_batches = int(np.ceil(len(self.images_df) / float(batch_size))) # Total number of batches

    def __iter__(self):  # For iterating the generator
        return self

    def __next__(self):   # For fetching the next batch, when iterating

        # --- Automatically reset the generator ---
        # So that it will be used in each epoch during training, without raising an error

        # In case the batch index is bigger than the total batches, the index is reset to 0
        if self.index >= self.total_batches:
            self.index = 0

            # Restarting the images and metadata generators
            self.images_generator = images_generator(
                self.aug_generator, self.images_df,
                self.batch_size,
                self.target_width, self.target_height
            )

            self.metadata_generator = MetadataGenerator(self.metadata_df, self.batch_size, self.mode)

        # Fetching the next augmented images' batch
        x_1, y_1 = next(self.images_generator)

        # Fetching the next metadata batch
        x_2, y_2 = next(self.metadata_generator)

        # Increasing the batch index
        self.index += 1

        return [x_1, x_2], y_2  # y_1 -> original groups, y_2 -> diagnosis

    def __len__(self):
        return self.total_batches
# ---------------------------------------------------------------------------
