from collections import namedtuple

# Instanciate the namedtuple Label
Label = namedtuple(
    "Label",
    [
        "name",
        "id",
        "trainId",
        "category",
        "categoryId",
        "hasInstances",
        "ignoreInEval",
        "color",
    ],
)

# Define the list of labels namedtuples
labels = [
    Label("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    Label("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    Label("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    Label("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]

# Define the TARGET_SIZE VAR
TARGET_SIZE = (256, 512)

# Define the columns order for the comparative metrics table
COL_ORDER = ['experiment_folder',
 'train_Dice',
 'train_IoU_class_0',
 'train_IoU_class_1',
 'train_IoU_class_2',
 'train_IoU_class_3',
 'train_IoU_class_4',
 'train_IoU_class_5',
 'train_IoU_class_6',
 'train_IoU_class_7',
 'train_MeanIoU',
 'train_Pixel_Accuracy',
 'train_loss',
 'val_Dice',
 'val_IoU_class_0',
 'val_IoU_class_1',
 'val_IoU_class_2',
 'val_IoU_class_3',
 'val_IoU_class_4',
 'val_IoU_class_5',
 'val_IoU_class_6',
 'val_IoU_class_7',
 'val_MeanIoU',
 'val_Pixel_Accuracy',
 'val_loss',
 'test_Dice',
 'test_IoU_class_0',
 'test_IoU_class_1',
 'test_IoU_class_2',
 'test_IoU_class_3',
 'test_IoU_class_4',
 'test_IoU_class_5',
 'test_IoU_class_6',
 'test_IoU_class_7',
 'test_MeanIoU',
 'test_Pixel_Accuracy',
 'test_loss',
 'best_epoch',
 'n_epochs',
 'n_steps',
 'model_name',
 'inference_time',
 'classes',
 'train_samples',
 'val_samples',
 'test_samples',
 'train_batches',
 'augmentations',
 'normalize',
 'loss_fn',
 'optimizer',
 'learning_rate']

# Annotation des barplots ou autre
def show_nums_axes(ax, orient="v", fmt=".0g", extra="", stacked=False):
    """
    Affiche les valeurs numériques sur les barres d'un graphique à barres.

    Args:
        ax (matplotlib.axes.Axes): L'axe du graphique.
        fmt (str, optional): Format d'affichage des nombres
        orient (str, optional): L'orientation des barres. 'v' pour vertical (par défaut), 'h' pour horizontal.
        extra (str, optional): Texte additionnel à afficher sur les annotations. Ex. : " %"
        stacked (bool, optionnal): S'adapte pour un barstackedplot

    Returns:
        None
    """
    # Error handling
    if orient not in ["h", "v"]:
        raise ValueError("orient doit être égal à 'h' ou 'v si spécifié")
    try:
        format(-10.5560, fmt)
    except ValueError:
        raise "Erreur: le format spécifié dans fmt n'est pas correct."
    if not isinstance(stacked, bool):
        raise ValueError("stacked doit être un booléen")
    # Body
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if orient == "v":
            if not stacked:
                ax.annotate(
                    f"{height:{fmt}}{extra}" if height != 0 else "",
                    (x + width / 2.0, height),
                    ha="center",
                    va="bottom",
                )
            else:
                ax.annotate(
                    f"{height:{fmt}}{extra}",
                    (x + width - 4, y + height / 2),
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="top",
                )
        else:
            if not stacked:
                ax.annotate(
                    f"{width:{fmt}}{extra}" if width != 0 else "",
                    (width, y + height / 2.0),
                    ha="left",
                    va="center",
                )
            else:
                ax.annotate(
                    f"{width:{fmt}}{extra}",
                    (x + width - 1, y + height / 2),
                    fontsize=10,
                    fontweight="bold",
                    ha="right",
                    va="center",
                )