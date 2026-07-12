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

    def call(self, y_true, y_pred, sample_weight=None):
        # paramètres de la loss pondérée pixel par pixel, pour diminuer l'impact du déséquilibre des classes dans la loss finale ('human' plus rare que 'background' mais plus important à détecter)
        ## y_true : shape (batch, H, W) avec des labels entiers (sparse) ici : (4, 256, 128) — un entier 0=>7 par pixel
        ## y_pred : shape (batch, H, W, num_classes) avec des logits  ici : (4, 256, 128, 8) — un vecteur de 8 valeurs par pixel  (8 scores bruts par pixel)
        ## sample_weight : shape (batch, H, W) ou None, pour pondérer la loss pixel par pixel ici :(4, 256, 128) — un float par pixel
        
        #### Dice Loss
        # Convertion one-hot des labels sparse, chaque entier en un vecteur de 8 cases ici la shape : (4, 256, 128) => (4, 256, 128, 8)
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)
        # conversion des logits bruts sans contraintes en proba avec softmax sur la dernière dimension de la pred, proba evidemment toutes positives, dont la somme fait 1 sur les 8 classes
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        # pour chaque image du batch, aplatir les masques réels et pred pour simplifier le calcul matriciel en factorisant les dimensions spatiales H×W en une seule dimension (la surface de l'image)
        # Avant : (4, 256, 128, 8) => 4 images, 256 lignes, 128 colonnes, 8 classes // Après : (4, 256*128, 8) => 4 images, 32768 pixels, 8 classes
        # syntaxe : tf.reshape(tensor à reshape, [batch_size, -1 (calcul : aplati ce qui reste à cette position de la liste), num_classes (dernière dimension de la shape d'origine ici 8)]) 
        y_true_flat = tf.reshape(y_true_onehot, [tf.shape(y_true_onehot)[0], -1, self.num_classes])
        y_pred_flat = tf.reshape(y_pred_softmax, [tf.shape(y_pred_softmax)[0], -1, self.num_classes])
        # Formule du Dice coefficient : 2 × |A ∩ B| / (|A| + |B|)
        # les tenseurs y_true_flat et y_pred_flat sont de shape (batch, H*W, num_classes) 
        # Calcule intersection et union
        # intersection : produit pixel par pixel entre les masques réels et prédits, puis somme sur les pixels de chaque image du batch, pour chaque classe, => shape (batch, num_classes) 
        # Le 0 du one-hot annule automatiquement toutes les mauvaises classes, seule la vraie classe contribue. Plus le modèle est confiant et correct, plus l'intersection est élevée (y_true=1, y_pred=0,85 résultat => 0,85 // y_true =0, y_pred=0,85 résultat => 0)
        intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
        # union : somme des masques réels et prédits sur les pixels de chaque image du batch, pour chaque classe, => shape (batch, num_classes)
        # somme sur les pixels séparément les deux tenseurs réels et pred, puis on fait après une addition scalaire sur les 2 tenseurs => shape (batch, num_classes) 
        union = tf.reduce_sum(y_true_flat, axis=1) + tf.reduce_sum(y_pred_flat, axis=1)
        # Calcule le Dice coefficient et la perte associée
        # formule = 2 × intersection / union oblige le score à être élevé seulement quand le réel et le prédit s'accordent
        # dice_smooth (1e-5) évite la division par zéro si une classe est absente de l'image
        dice = (2.0 * intersection + self.dice_smooth) / (union + self.dice_smooth)
        dice_loss = 1.0 - tf.reduce_mean(dice) # le Dice coefficient parfait vaut 1.0, donc la loss vaut 0.0 si parfait, 1.0 si nul

        ### Focal Loss avec pondération pixel par pixel
        # Évite les log(0) qui donnerait -inf avec un clipping
        epsilon = tf.keras.backend.epsilon() # 1e-7
        # Si une proba vaut exactement 0.0, on la remplace par 1e-7, si elle vaut exactement 1.0, on la remplace par 1 - 1e-7
        y_pred_clipped = tf.clip_by_value(y_pred_softmax, epsilon, 1.0 - epsilon) 
        # erreur par pixel par classe : shape (batch, H, W, num_classes), cross-entropy standard par pixel et par classe
        # log(proba) est négatif (car proba ∈ ]0,1[), donc le - devant le rend positif. 
        # La multiplication par y_true_onehot annule toutes les classes sauf la vraie, elle seule contribue au calcul de la loss 
        # si modèle est confiant et correct, -log(proba) (la loss calculée) est faible
        # si modèle confiant et faux, la loss calculée  est très élevée et le modèle sera plus pénalisé
        # si le modèle est incertain (proba proche de 0.5) et correct, la Loss est modérée => le modèle sera pénalisé pour son manque de confiance, il doit encore s'améliorer
        # si le modèle est incertain et faux, la Loss est élévée mais moins que lorsque confiant et faux => pénalisé, mais moins fort que si le modèle était sur de lui et faux
        cross_entropy = -y_true_onehot * tf.math.log(y_pred_clipped)
        focal_map = self.focal_alpha * tf.pow(1 - y_pred_clipped, self.focal_gamma) * cross_entropy
        # une erreur par pixel individuel : somme sur les 8 classes pour obtenir une seule valeur d'erreur par pixel, toujours grâce au onehot, seule la vraie classe contribue shape (batch, H, W)
        # Shape : (4, 256, 128, 8) => (4, 256, 128) 
        focal_per_pixel = tf.reduce_sum(focal_map, axis=-1)

        # applique les poids par pixel AVANT la réduction globale (on multiplie l'erreur de chaque pixel par son poids de classe, avant de faire la moyenne)
        if sample_weight is not None:
            # erreur_finale_pixel = erreur_focal_pixel × poids_classe_du_pixel. 
            
            focal_per_pixel = focal_per_pixel * tf.cast(sample_weight, focal_per_pixel.dtype) # cast assure que les types correspondent (float32)

        # moyenne la perte sur le batch
        # les classes rares ayant un poids plus élevé, leurs pixels contribuent plus à la loss finale que les pixels des classes fréquentes
        focal_loss = tf.reduce_mean(focal_per_pixel)

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
        
        # # somme sur les classes → shape (batch, H, W)
        
        
        # # applique les poids par pixel AVANT la réduction globale
        
        
        # # Calcule la cross-entropy pixel par pixel
        # cross_entropy = -y_true_onehot * tf.math.log(y_pred_clipped)
        # # Applique le modulateur focal (plus de poids aux exemples difficiles)
        # focal = self.focal_alpha * tf.pow(1 - y_pred_clipped, self.focal_gamma) * cross_entropy
        # # moyenne la perte sur le batch
        # focal_loss = tf.reduce_mean(tf.reduce_sum(focal, axis=-1))

        # # Combo
        # return self.dice_weight * dice_loss + self.focal_weight * focal_loss

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
    
