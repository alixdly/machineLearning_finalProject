# Machine Learning Course - Christophe Cerisara

### Rapport de projet : Arthur Mouchot, Alix Delannoy
14 décembre 2020 - École des Mines de Nancy
https://members.loria.fr/CCerisara/#courses/machine_learning/


# Partie 1 : Les données

### Généralités

Les données proviennent du site [Kaggle](https://www.kaggle.com/vinayshanbhag/radar-traffic-data "Kaggle"). <br/>
Il s'agit des données de traffic collectées par des capteurs radar à Austin aux États-Unis. Notre but est d'utiliser ces données pour entrainer un modèle de deep learning qui sera capable de prévoir le traffic futur. <br/> 

Voici un aperçu des données : 


<img width="1121" alt="Capture d’écran 2020-11-25 à 15 20 51" src="https://user-images.githubusercontent.com/47599816/100239583-dba72780-2f31-11eb-9775-0e7c7e41ae6e.png">


Il y a trois types d'informations dans chacune des lignes : 

*LOCATION* : Les 3 premières colonnes décrivent les emplacements de capteurs de traffic <br/>
*TIME* : Les 7 suivantes concernent le moment ou la donnée de traffic a été captée <br/>
*TRAFFIC* : Les 2 dernières concernent la donnée du traffic (direction et volume) <br/>

De plus, voici un tableau qui résume les données : 

<img width="1027" alt="Capture d’écran 2020-11-25 à 15 03 09" src="https://user-images.githubusercontent.com/47599816/100239588-dcd85480-2f31-11eb-9a9f-4fee6d65cc02.png">

**A RETENIR** : <br/>
* Il n'y a pas de "trous" dans nos données, toutes les lignes sont complètes (attention, cela ne veut pas dire que nous disposons des mêmes données pour chaque emplacement, ni qu'il ne manque pas de mesures. Mais chaque mesure est complète)
-> par exemple, il peut manquer des jours, ou des heures, ou une minute... on considère que cela signifie que le traffic était égal à 0 à ce moment là (ce qui est surement faux dans la réalité)
* Nous disposons de 4 603 861 lignes
* Nous nous intéressons à une période de 3 ans (de 2017 à 2019)
* Nous nous intéressons à 23 emplacements (location_name)
* **Le but est de prévoir des données de traffic (colonne Volume) pour chaque emplacement dans chaque direction à cet emplacement i.e. par couple (location_name,Direction) sur une période de temps à définir.**


# Partie 2 : Un premier modèle : Convolutional Neural Network (CNN)

## Explications
Dans cette partie, nous avons décidé tout d'abord de nous intéresser à la donnée de traffic jour par jour pour un emplacement et une direction et de construire un CNN capable de prédire correctement le traffic futur. Nous avons décidé d'utiliser un modèle CNN en premier lieu car même si ce n'est pas le modèle le plus adapté pour la prédiction d'une série temporelle, il s'agit du modèle avec lequel nous étions le plus à l'aise à l'issue du cours. 
Source : Nous avons utilisé le notebook Kaggle suivant pour comprendre et implémenter le CNN - https://www.kaggle.com/hanjoonchoe/cnn-time-series-forecasting-with-pytorch

Nous préparons trois sets de données : 
- train_set : le set de données qui va servir à l'entrainement des paramètres du modèle 
- valid_set : le set de validation, aussi appelé set de développement, qui va servir à évaluer les hyperparamètres du modèle
- test_set : le set sur lequel on va tester notre modèle

Emplacement : ' CAPITAL OF TEXAS HWY / LAKEWOOD DR' 
<img width="595" alt="Capture d’écran 2020-12-13 à 16 16 06" src="https://user-images.githubusercontent.com/47599816/102015924-9458cd80-3d5e-11eb-8a69-66c24228e390.png">

Direction : South Bound

Train_set : de septembre 2017 à septembre 2018 \
Valid_set : de octobre 2018 à décembre 2018 \ 
Test_set : Janvier 2019

Pour utiliser un modèle CNN pour prédire des données à 1 dimension, nous procédons de la manière suivante : \
Nous séparons nos données en des sous-séquences de taille 3 (input) auxquelles nous associons une séquence de taille 1 (ouptput). Notre modèle doit apprendre à prédire l'output en fonction de l'input. Nous allons essayer plusieurs architectures de CNN et d'hyperparamètres sur nos données. 

## CNN
Notre CNN, le plus simple possible, est construit de la manière suivante : 

1 couche de convolution \
1 couche d'activation non linéaire \
1 'fully connected' layer 

**Résultats : **

*lr=0.1, epochs=100*

<img width="488" alt="Capture d’écran 2020-12-13 à 17 07 46" src="https://user-images.githubusercontent.com/47599816/102017229-db968c80-3d65-11eb-841e-0aed08227c74.png">

Notre modèle n'apprend pas. Nous allons donc diviser le learning rate par un facteur 10. 

*lr=0.01, epochs = 100* 

<img width="488" alt="Capture d’écran 2020-12-13 à 17 10 08" src="https://user-images.githubusercontent.com/47599816/102017268-2adcbd00-3d66-11eb-98cd-1dd879f3307d.png">

On obtient un meilleur résultat. Nous allons essayer avec d'encore meilleurs learning rates, pour voir si nous obtenons de meilleurs résultats. 

lr = 0.001, epochs=100

<img width="482" alt="Capture d’écran 2020-12-13 à 17 15 09" src="https://user-images.githubusercontent.com/47599816/102017371-ea317380-3d66-11eb-8c07-312b4b54cf37.png">

On peut voir que le training loss remonte avant de recommencer à descendre, ce qui n'est pas satisfaisant, mais on obtient tout de même à l'issue des 100 époques un modèle aussi satisfaisant que précedemment.

lr = 0.0001, epochs=100

<img width="497" alt="Capture d’écran 2020-12-13 à 17 16 09" src="https://user-images.githubusercontent.com/47599816/102017372-eb62a080-3d66-11eb-90e2-3f5c30c4275d.png">

Cette fois-ci, nous obtenons un apprentissage satisfaisant. Pour un learning rate de 0.0001 et 100 époques, nous traçons également l'erreur sur le jeu de données de validation : 

<img width="468" alt="Capture d’écran 2020-12-13 à 17 27 41" src="https://user-images.githubusercontent.com/47599816/102017672-99bb1580-3d68-11eb-99c3-8c1b894d3950.png">

Si on souhaite augmenter le nombre d'époques, par exemple à 1000 on obtient les résultats suivants : 

<img width="523" alt="Capture d’écran 2020-12-13 à 17 34 46" src="https://user-images.githubusercontent.com/47599816/102017830-82305c80-3d69-11eb-9090-70abd091ad87.png">

qui augmentent le temps de calcul mais ne sont pas significativement meilleur. 

Enfin, si on applique notre modèle entrainé aux données du mois de janvier 2019, nous obtenons les résultats suivants : 

<img width="668" alt="Capture d’écran 2020-12-13 à 17 47 52" src="https://user-images.githubusercontent.com/47599816/102018181-e05e3f00-3d6b-11eb-87af-97720e6add88.png">

Notre modèle semble assez performant pour capter la tendance générale du traffic. Cependant, il est un peu "en retard" sur le pic de la 5ème date, et de manière générale. Il fluctue beaucoup quand la courbe réelle est assez "lisse". Nous pouvons donc tester d'autres modèles qui seront meilleurs pour prédire le traffic futur.

## Résultats du CNN sur d'autres couples (emplacement,direction)

Pour information, voici les résultats obtenus par notre modèle sur d'autres couples (emplacement, direction), avec les mêmes valeurs de learning rate et epochs : (0.01 et 100)

Emplacement : ' CAPITAL OF TEXAS HWY / LAKEWOOD DR', Direction : North Bound 

<img width="500" alt="Capture d’écran 2020-12-13 à 18 10 10" src="https://user-images.githubusercontent.com/47599816/102018621-a0e52200-3d6e-11eb-9c65-8d958998aa37.png">

<img width="683" alt="Capture d’écran 2020-12-13 à 18 10 17" src="https://user-images.githubusercontent.com/47599816/102018623-a3477c00-3d6e-11eb-9a08-668056c897d6.png">


Emplacement : '1612 BLK S LAMAR BLVD (Collier)' ,Direction : North Bound

<img width="480" alt="Capture d’écran 2020-12-13 à 18 14 56" src="https://user-images.githubusercontent.com/47599816/102018705-354f8480-3d6f-11eb-9e12-41b738bade4c.png">

<img width="672" alt="Capture d’écran 2020-12-13 à 18 15 03" src="https://user-images.githubusercontent.com/47599816/102018708-37b1de80-3d6f-11eb-8bf8-c3306d536d6d.png">

Emplacement : '400 BLK AZIE MORTON RD (South of Barton Springs Rd)' ,Direction : South Bound

<img width="482" alt="Capture d’écran 2020-12-13 à 18 19 59" src="https://user-images.githubusercontent.com/47599816/102018803-d6d6d600-3d6f-11eb-986c-e4a002f51643.png">

<img width="668" alt="Capture d’écran 2020-12-13 à 18 20 04" src="https://user-images.githubusercontent.com/47599816/102018804-d8080300-3d6f-11eb-9e20-7c86492901c7.png">

Par exemple, pour cette instance, on peut voir que notre modèle n'est pas du tout performant puisqu'il prévoit le pic de traffic bas du 5ème jour trop tard, mais aussi un deuxième pic inexistant 2 jours après. 

Nous allons donc tester un autre modèle de deep learning. 

# Partie 3 : Une autre modèle : le LSTM

## Explications : 

On remarque dans la littérature aujourd'hui que les réseaux de neurones récurrents à mémoire court-et-long terme (LSTM) sont particulièrement adaptés pour réaliser des prédictions futures à partir de séries temporelles. En effet les réseaux de neurones réccurents sont par définition efficaces pour réaliser des prédiction de séries temporelles car ils prennent en compte le passer pour prédire le futur. Le problème qui se pose advient lors de la rétropogagation de l'erreur, qui à cause de valeur de gradient trop faibles, à du mal à impacter les décisions sur les entrées les plus anciennes, on appelle cela le problème du gradient evenscent. Le LSTM, par le biais d'une architecture de cellule différente, entend régler ce problème et donc promet une meilleure prise en compte des évenements anciens.

Dans un RNN, on retrouve un architecture semblable à celle d'un réseau de neurone "classique" sauf que les neurones "bouclent" sur eux-même ce revient à chaque étape à considérer une entrée supplémentaire qui est l'état propre du neurone d'où le caractère réccurent du réseau.
![image](https://user-images.githubusercontent.com/74898266/102105941-457b6880-3e30-11eb-9ddb-cbe0d97fe024.png)

Au sein du neurone LSTM, on va retrouver 3 portes (la Porte d'oubli, la Porte d'entrée et la Porte de sortie) en plus de deux états (l'état de la cellule et l'état caché). Les portes vont servir à "filtrer" les données pour ne garder que les données utiles à l'apprentissage et ainsi limiter considérablement les problèmes de graient evanessant.
![image1](https://user-images.githubusercontent.com/74898266/102106037-6217a080-3e30-11eb-905a-a3ded411547b.png)

## Démarche suivie

Nous nous proposons de contruire un modèle LSTM permettant à partir de l'historique du traffic en un lieu et une direction, de prédire les deux prochains mois de circulation. Le modèle construit doit être fonctionnel pour n'importe quel lieu et quelle direction. Nous séparons la série temporelle en un jeu d'apprentissage et un jeu de validation avec un rapport de taile que nous nous autoriserons à faire varier. Pour pouvoir traiter les données de manière "relativement" rapide, nous agglomérons les mesures à la journée, les prédictions seront donc également des prédictions journalières.

## LSTM : 

Notre modèle se compose d'une couche LSTM à laquelle il faut définir un état initial et le nombre de couches cachées suivie d'une couche fully connected. A chaque epoch, un aprrentissage est réalisé sur le jeu d'apprentissage (plus ancien) avec rétropopagation de l'erreur suivi d'une validation sur le jeu de validation avec à nouveau une rétropopagation de l'erreur. Ces deux courbes seront superposées lors de la présentation des résultats.
Il nous faut aussi définir lors de chaque essai la valeur du taux d'apprentisage (learning rate - lr) et le nombre d'epochs à réaliser.

## Résultats 

TO DO : détailler ce qui est bien et pas bon, ce qu'on tente pour améliorer 

Les différents essais de prédiction sont détaillés ici avec les hyper-paramètres utilisés. On commentera les courbes d'erreurs obtenus.

**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0001 \
Epochs = 150 \
Test_size = data_size / 5 \
Hidden_layer_size = 25

![image2](https://user-images.githubusercontent.com/74898266/102111829-26cca000-3e37-11eb-9605-1d85c5754203.png)
![image3](https://user-images.githubusercontent.com/74898266/102111856-32b86200-3e37-11eb-85a2-d214a72fea90.png)

Notre modèle n'apprend pas du tout assez au vu des courbes d'apprentissages qui se superposent extremement bien néanmoins. 
Nous allons essayer d'augmnter le taux d'apprentissage d'un facteur 10 car nous pouvaon être en présence d'un minimum local.


**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.001 \
Epochs = 150 \
Test_size = data_size / 5 \
Hidden_layer_size = 25

![image4](https://user-images.githubusercontent.com/74898266/102114921-13bbcf00-3e3b-11eb-998a-bf20dc6fd913.png)
![image5](https://user-images.githubusercontent.com/74898266/102114948-1dddcd80-3e3b-11eb-98d8-5ec6f3787e07.png)

Cette fois la courbe d'apprentissage est bien globalement décroissante mais présente de fortes oscillations imprévisibles. De plus la tendance observée de la prédiction semble avoir une moyenne inférieure à celle précédente. Nous décidons maintenant d'augmenter d'un facteur 2 la taille du jeu de validation.

**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0005 \
Epochs = 250 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 25 
 
![Sans titre](https://user-images.githubusercontent.com/47599816/102079232-937e7500-3e0c-11eb-87bc-5c2fa6cd853f.png)

![Sans titre2](https://user-images.githubusercontent.com/47599816/102079236-95483880-3e0c-11eb-87fa-28936aa1dd52.png)

Si la prédiction otenue semble suivre des tendances, l'erreur lors de l'apprentissage et de la validation n'est pas celle attendue avec, là encore, de fortes oscillations observées à partir du 150ème epoch.

Nous essayons de réduire le nombre d'epochs à 150.


**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.001 \
Epochs = 150 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 25 
 
![Sans titre3](https://user-images.githubusercontent.com/47599816/102079382-e48e6900-3e0c-11eb-8dff-9279c8c57c03.png)
 
![Sans titre4](https://user-images.githubusercontent.com/47599816/102079394-e9ebb380-3e0c-11eb-8d56-5a77ce6aa9ad.png)

La démarche n'a pas forcément eu l'effet escompté et l'on retrouve ces sauts irréguliers mais avec une tendance sur les prédictions obtenues qui semble extremement réaliste par rapport à la série de départ.


**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0005 \
Epochs = 250 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 50 

![Sans titre5](https://user-images.githubusercontent.com/47599816/102079395-eb1ce080-3e0c-11eb-8a78-2007a9d79820.png)

![Sans titre6](https://user-images.githubusercontent.com/47599816/102079396-ece6a400-3e0c-11eb-9d44-b0b845873de2.png)

On peut également essayer d'augmenter le nombre de couches cachées mais celà n'a pas énormément d'impact et l'on retrouve le même type d'erreur que précédemment avec ces sauts et oscillations irréguliers à partir cette fois du 100ème epoch.
 
**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0005 \ 
Epochs = 100 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 50 

![Sans titre7](https://user-images.githubusercontent.com/47599816/102079409-f40db200-3e0c-11eb-8e87-b02680bfafde.png)

![Sans titre8](https://user-images.githubusercontent.com/47599816/102079414-f708a280-3e0c-11eb-81ad-73067e54586a.png) 

En réduisant le nombre d'epochs on obtient certes une courbe d'erreur lisse mais avec un modèle qui aprend peu et une tendance de prédiction qui semble moins réaliste.
 
**Hyper parameters**

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0001 \
Epochs = 500 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 50
 
![Sans titre9](https://user-images.githubusercontent.com/47599816/102079421-fbcd5680-3e0c-11eb-9ba2-4a717d6808ac.png)

![Sans titre10](https://user-images.githubusercontent.com/47599816/102079432-00920a80-3e0d-11eb-952f-606c9f99683d.png)

Un dernier essai a été fait avec un nombre d'epochs au maximum des capacités de la machine et un taux d'erreur très bas. La courbe d'apprentissage est très acceptable avec des petites oscillations mais la tendance observée sur les prédictions ne correspond pas et la courbe est trop lisse par rapport aux données passés. 

On pourrait conclure après les trois derniers essais qu'augmenter le nombre de couches cachées amène à une prédiction plus lisse par rapport aux oscillations observées sur les données réelles.


