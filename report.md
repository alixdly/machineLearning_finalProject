# Machine Learning - Mines Nancy

### Rapport de projet : Arthur Mouchot, Alix Delannoy



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
* Nous disposons de 4 603 861 lignes
* Nous nous intéressons à une période de 3 ans (de 2017 à 2019)
* Nous nous intéressons à 23 emplacements (location_name)
* **Le but est de prévoir des données de traffic (colonne Volume) pour chaque emplacement dans chaque direction à cet emplacement i.e. par couple (location_name,Direction) sur une période de temps à définir.**


# Partie 2 : Un premier modèle : Convolutional Neural Network (CNN)

## Explications
Dans cette partie, nous avons décidé tout d'abord de nous intéresser à la donnée de traffic jour par jour pour un emplacement et une direction et de construire un CNN capable de prédire correctement le traffic futur. 

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

**Hyper parameters **

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0005 \
Epochs = 250 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 25 
 
 ![image](https://user-images.githubusercontent.com/47599816/102024023-9c7d3100-3d8f-11eb-83f5-131dfd964d7e.png)
 ![image](https://user-images.githubusercontent.com/47599816/102024043-c46c9480-3d8f-11eb-922e-d33c528bc1ec.png)

**Hyper parameters **

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.001 \
Epochs = 150 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 25 
 
 ![image](https://user-images.githubusercontent.com/47599816/102024051-d1898380-3d8f-11eb-85ee-9c9b3d413476.png)
![image](https://user-images.githubusercontent.com/47599816/102024056-d3ebdd80-3d8f-11eb-9c92-ca1c247f9648.png)
 
**Hyper parameters **

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0005 \
Epochs = 250 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 50 

![image](https://user-images.githubusercontent.com/47599816/102024062-e49c5380-3d8f-11eb-90a5-9565fec6ce21.png)
![image](https://user-images.githubusercontent.com/47599816/102024063-e6fead80-3d8f-11eb-8784-3808198c0be2.png)

 
**Hyper parameters **

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0005 \ 
Epochs = 100 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 50 

![image](https://user-images.githubusercontent.com/47599816/102024070-f41b9c80-3d8f-11eb-95ad-a03dfee38cd6.png)
![image](https://user-images.githubusercontent.com/47599816/102024072-f67df680-3d8f-11eb-8468-f657253983ca.png)
 
 
**Hyper parameters **

Train_window : 60 \
Fut_pred : 60 \
Lr = 0.0001 \
Epochs = 500 \
Test_size = data_size / 2.5 \
Hidden_layer_size = 50
 
![image](https://user-images.githubusercontent.com/47599816/102024075-009ff500-3d90-11eb-850d-9801d9784f56.png)
![image](https://user-images.githubusercontent.com/47599816/102024076-03024f00-3d90-11eb-98e9-ec756cccef98.png)



