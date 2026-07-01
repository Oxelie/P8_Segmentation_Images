import tensorflow as tf
import keras
from typing import Any, Optional, Dict

# déclara la classe pour notre combo loss qui hérite de tf.keras.losses.Loss.
#@tf.keras.utils.register_keras_serializable()
@keras.saving.register_keras_serializable()
class DiceFocalLoss(tf.keras.losses.Loss):
    """
    calcule la combo loss Dice + Focal, adaptée à la segmentation multi-classes avec déséquilibre.
    utilisable directement dans model.compile(loss=DiceFocalLoss(...))
    """
    def __init__(
        self,
        num_classes: int = 8,
        dice_smooth: float = 1e-5, # facteur de lissage pour éviter la division par zéro
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        dice_weight: float = 0.5, # poids du composant dans le calcul de la combo loss
        focal_weight: float = 0.5, # poids du composant dans le calcul de la combo loss
        name: str = "DiceFocalLoss",
        **kwargs: Any,
    ):
        super().__init__(name=name, **kwargs) # call du contructeur parent
        self.num_classes = num_classes
        self.dice_smooth = dice_smooth
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def call(self, y_true, y_pred):
        # Dice Loss
        # Convertit les labels sparse en one-hot
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        # Applique softmax sur les prédictions
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        # Aplati les masques pour le calcul
        y_true_flat = tf.reshape(y_true_onehot, [tf.shape(y_true_onehot)[0], -1, self.num_classes])
        y_pred_flat = tf.reshape(y_pred_softmax, [tf.shape(y_pred_softmax)[0], -1, self.num_classes])
        # Calcule intersection et union
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        # Calcule le Dice coefficient et la perte associée.
        dice = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        dice_loss = 1.0 - tf.reduce_mean(dice)

        # Focal Loss
        # Évite les log(0) avec un clipping
        epsilon = tf.keras.backend.epsilon()
        y_pred_clipped = tf.clip_by_value(y_pred_softmax, epsilon, 1.0 - epsilon)
        # Calcule la cross-entropy pixel par pixel
        cross_entropy = -y_true_onehot * tf.math.log(y_pred_clipped)
        # Applique le modulateur focal (plus de poids aux exemples difficiles)
        focal = self.focal_alpha * tf.pow(1 - y_pred_clipped, self.focal_gamma) * cross_entropy
        # moyenne la perte sur le batch
        focal_loss = tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

        # Combo
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

    # sauvegarde de l'état et des paramètres de la loss pour pour le reload du modèle (=sérialisation de la loss)
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "dice_smooth": self.dice_smooth,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "dice_weight": self.dice_weight,
            "focal_weight": self.focal_weight,
        })
        return config
    
    
    
## Dice coefficient pour suivre l'évolution du dice coefficient à chaque époch de train et pour l'évaluation du modèle sur le test, je vais créer une classe qui héritera de tf.keras.metrics.Metric.

#@tf.keras.utils.register_keras_serializable(package="custom", name="DiceMetric") # Le décorateur permet de sauvegarder/charger la métrique dans un modèle Keras.
@keras.saving.register_keras_serializable()
class DiceMetric(tf.keras.metrics.Metric): # déclaration de la classe avec héritage de tf.keras.metrics.Metric
    """
    Dice coefficient pour segmentation sémantique avec labels sparse.
    Utilisable comme métrique ou fonction de perte personnalisée.
    """
    def __init__(
        self,
        num_classes: int = 8,
        smooth: float = 1e-5, # facteur de lissage pour éviter la division par zéro
        name: str = "Dice",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.smooth = smooth
        self.total_dice = self.add_weight(name="total_dice", initializer="zeros") # somme des scores Dice
        self.count = self.add_weight(name="count", initializer="zeros") # nombre total d'image traités dans tous les batchs

    # méthode appelée à chaque batch pour mettre à jour les scores
    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ) -> None:
        # Convertit les labels sparse en one-hot
        y_true_onehot = tf.one_hot(y_true, depth=self.num_classes)
        # Prend la classe prédite (argmax sur les probabilités softmax) et convertit en one-hot
        y_pred_indices = tf.argmax(y_pred, axis=-1)
        y_pred_onehot = tf.one_hot(y_pred_indices, depth=self.num_classes)
        # aplatit les maqsues pour le calcul (batch, H*W, num_classes)
        y_true_flat = tf.reshape(y_true_onehot, [tf.shape(y_true_onehot)[0], -1, self.num_classes])
        y_pred_flat = tf.reshape(y_pred_onehot, [tf.shape(y_pred_onehot)[0], -1, self.num_classes])
        # calcul intersection et union pour chaque batch
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        # score Dice pour chaque classe et moyenne sur les classes pour chaque image
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_per_image = tf.reduce_mean(dice, axis=-1)
        # enregistre la somme des scores Dice et le nombre d’images traitées 
        self.total_dice.assign_add(tf.reduce_sum(dice_per_image))
        # maj le nb d’images traitées sur l'epoch
        self.count.assign_add(tf.cast(tf.size(dice_per_image), tf.float32))

    # score Dice moyen sur tout le dataset
    def result(self) -> tf.Tensor:
        return self.total_dice / self.count

    # réinitialise les variables d’état (scores) au début d’une nouvelle epoch
    def reset_states(self) -> None:
        self.total_dice.assign(0.0)
        self.count.assign(0.0)

    # sérialisation de la métrique pour la sauvegarde et le reload du modèle
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "smooth": self.smooth,
        })
        return config
    
