import os
import tensorflow
from tensorflow.keras.optimizers import SGD

ROOT_DIR = '../'

# ---------------------------------------------------------------------------
# --- Dataset setup ---
PATIENTS = 'all_patients'
DATASET = 'orthogonal'
PLANE = 'axial'

# --- The directories for the datasets ---
DATASETS_DIRS = {
    'all_patients': f'{ROOT_DIR}/Datasets/Standarized_dataset_FINAL/All_Patients',
    'stable_patients': f'{ROOT_DIR}/Datasets/Standarized_dataset_FINAL/Stable_Patients',

    'orthogonal': 'Orthogonal_Slices',
    'shannon_1_top_slice': 'Shannon_Slices_Top_1',
    'shannon_8_top_slice': 'Shannon_Slices_Top_8',

    'axial': 'ADNI_labeled_axial',
    'coronal': 'ADNI_labeled_coronal',
    'sagittal': 'ADNI_labeled_sagittal',
    }

# --- Setting the basic paths for the dataset ---
PATHS = {
    'labeled_image_dir': os.path.join(DATASETS_DIRS[PATIENTS],
                                      DATASETS_DIRS[DATASET],
                                      DATASETS_DIRS[PLANE]),
    'metadata_dir': f'{ROOT_DIR}/csv',
}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Ordinal encoding of the labels ---
# 'CN' : Cognitively Normal
# 'MCI': Mild Cognitive Impairment
# 'AD' : Alzheimer's Disease
CLASS_ENCODINGS = {
    'CN_AD': {'CN': 0, 'AD': 1},
    'CN_MCI': {'CN': 0, 'MCI': 1},
    'MCI_AD': {'MCI': 0, 'AD': 1},
    'CN_MCI_AD': {'CN': 0, 'MCI': 1, 'AD': 2}
}

MODEL = 'CN_MCI_AD'
CLASS_INDICES = CLASS_ENCODINGS[MODEL]
NUM_CLASSES = len(CLASS_INDICES.items())
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Optimizer setup ---
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
EPOCHS = 100
BATCH_SIZE = 16
OPTIMIZER = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)

DATA_LEAKAGE = False

# --- Early stopping patience ---
PATIENCE = 10
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- Basemodel setup ---
BASE_MODEL = 'resnet50'
TRAINING_BASEMODEL = 'last_block'
INPUT_IMAGE_WIDTH = 224
INPUT_IMAGE_HEIGHT = 224

# --- Setting up the preprocessing function for the base model ---
PREPROCESSING_FUNCTIONS = {
        'resnet50': tensorflow.keras.applications.resnet50.preprocess_input,
        'resnet50v2': tensorflow.keras.applications.resnet_v2.preprocess_input,
        'vgg16': tensorflow.keras.applications.vgg16.preprocess_input,
        'vgg19': tensorflow.keras.applications.vgg19.preprocess_input,
        'inception_v3': tensorflow.keras.applications.inception_v3.preprocess_input,
        'densenet121': tensorflow.keras.applications.densenet.preprocess_input,
    }
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# --- The augmentations for the training, validation and test subset ---
TRAINING_CONFIG = {
        # 'rescale': 1.0 / 255,
        'preprocessing_function': PREPROCESSING_FUNCTIONS[BASE_MODEL],
        'rotation_range': 45,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        # 'shear_range': 0.1,
        # 'zoom_range': 0.1,
        'horizontal_flip': True,
        # 'vertical_flip': True
    }

TEST_CONFIG = {
    # 'rescale': 1.0 / 255,
    'preprocessing_function': PREPROCESSING_FUNCTIONS[BASE_MODEL],

}

VAL_CONFIG = {
    # 'rescale': 1.0 / 255,
    'preprocessing_function': PREPROCESSING_FUNCTIONS[BASE_MODEL],
}
# ---------------------------------------------------------------------------
