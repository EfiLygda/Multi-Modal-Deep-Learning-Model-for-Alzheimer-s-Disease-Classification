import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import rcParams
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Dictionary for setting the global plotting parameters
rc_fonts = {
        "text.usetex": True,
        "font.size": 15,
        "font.weight": 'bold',
        'mathtext.default': 'regular',
        'axes.titlesize': 16,
        "axes.labelsize": 16,
        'axes.linewidth': 1,
        "legend.fontsize": 15,
        'figure.titlesize': 16,
        'figure.figsize': (8.27, 11.69),
        "font.family": "serif",
        "font.serif": "computer modern roman",
        'figure.autolayout': True,
        'axes.labelweight': 'bold',
    }

# Setting the global plotting parameters
rcParams.update(rc_fonts)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dictionary with the hex codes of the colors used for plotting
colors_dict = {
    'latex_purple': '#6f00b7',
    'latex_purple!50': '#b780db',
    'latex_teal': '#008080',
    'latex_teal!50': '#80BFBF',
}

# Setting the dictionary with the RGBA codes of the colors used for plotting
colors_dict = {
    'loss': to_rgba(colors_dict['latex_purple']),
    'val_loss': to_rgba(colors_dict['latex_purple!50']),
    'acc': to_rgba(colors_dict['latex_teal']),
    'val_acc': to_rgba(colors_dict['latex_teal!50']),
}
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def prepare_training_figure():
    """
    Function for constructing the figure containing the axes for presenting the training process
    :return: tuple, containing the figure object and its axes abjects
    """

    # Constructing the main figure and its axes (5 rows for each CV fold and 2 columns for loss and accuracy axes)
    training_figure, training_axis = plt.subplots(nrows=5, ncols=2)

    # Creating pseudo lines for using their formatting in legends for the whole figure
    loss_line = Line2D([0], [0], color=colors_dict['loss'])
    val_loss_line = Line2D([0], [0], color=colors_dict['val_loss'], linestyle='dashed')
    acc_line = Line2D([0], [0], color=colors_dict['acc'])
    val_acc_line = Line2D([0], [0], color=colors_dict['val_acc'], linestyle='dashed')

    # Saving the pseudo lines in a list, so that they will be used as handles for the figure legend
    handles = [loss_line, val_loss_line, acc_line, val_acc_line]

    # Creating a legend for the loss and val loss lines
    leg_train = training_figure.legend(
        handles=handles[:2],
        title=r'',
        labels=['loss', 'val_loss'],
        bbox_to_anchor=(0.26, -0.05),
        loc="lower center",
        ncol=2
    )

    # Formatting the legend's appearance
    leg_train.get_frame().set_linewidth(1)
    leg_train.get_frame().set_edgecolor('black')
    leg_train.get_frame().set_boxstyle('Square', pad=0.2)

    # Creating a legend for the accuracy and val accuracy lines
    leg_val = training_figure.legend(
        handles=handles[2:],
        title=r'',
        labels=['acc', 'val_acc'],
        bbox_to_anchor=(0.75, -0.05),
        loc="lower center",
        ncol=2
    )

    # Formatting the legend's appearance
    leg_val.get_frame().set_linewidth(1)
    leg_val.get_frame().set_edgecolor('black')
    leg_val.get_frame().set_boxstyle('Square', pad=0.2)

    return training_figure, training_axis
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def prepare_ConfMatrices_figure():
    """
    Function for constructing a figure containing the confusion matrices of the 5 folds
    :return: tuple, containing the figure object and its axes abjects
    """

    # Constructing the main figure and its axes (3 rows for each CV fold and 2 columns for the total 5 matrices)
    CM_figure, CM_axis = plt.subplots(nrows=3, ncols=2)

    # Flattening the list containing all the axis objects (useful for adding the plots later to each axis)
    CM_axis = CM_axis.flatten()

    # Removing the axis from the last one, since it will not be used
    CM_axis[-1].set_axis_off()

    return CM_figure, CM_axis
# ---------------------------------------------------------------------------
