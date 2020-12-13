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

Dans cette partie, nous avons décidé tout d'abord de nous intéresser à la donnée de traffic tous les quarts d'heure pour un emplacement et une direction et de construire un CNN capable de prédire correctement le traffic futur. 

Nous préparons trois sets de données : 
- train_set : le set de données qui va servir à l'entrainement des paramètres du modèle 
- valid_set : le set de validation, aussi appelé set de développement, qui va servir à évaluer les hyperparamètres du modèle
- test_set : le set sur lequel on va tester notre modèle

Emplacement : ' CAPITAL OF TEXAS HWY / LAKEWOOD DR' 
<img width="595" alt="Capture d’écran 2020-12-13 à 16 16 06" src="https://user-images.githubusercontent.com/47599816/102015924-9458cd80-3d5e-11eb-8a69-66c24228e390.png">

Direction : North Bound

Train_set : de septembre 2017 à septembre 2018 \
Valid_set : de octobre 2018 à décembre 2018 \ 

Notre CNN, le plus simple possible, est construit de la manière suivante : 

1 couche de convolution \
1 couche d'activation non linéaire \
1 'fully connected' layer 

*Résultats :* 

*lr=0.1, epochs=100*

<img width="488" alt="Capture d’écran 2020-12-13 à 17 07 46" src="https://user-images.githubusercontent.com/47599816/102017229-db968c80-3d65-11eb-841e-0aed08227c74.png">

Notre modèle n'apprend pas. Nous allons donc diviser le learning rate par un facteur 10. 

*lr=0.01, epochs = 100* 

<img width="488" alt="Capture d’écran 2020-12-13 à 17 10 08" src="https://user-images.githubusercontent.com/47599816/102017268-2adcbd00-3d66-11eb-98cd-1dd879f3307d.png">

On obtient un meilleur résultat. Nous allons essayer avec d'encore meilleurs learning rates, pour voir si nous obtenons de meilleurs résultats. 

lr = 0.001, epochs=100

<img width="482" alt="Capture d’écran 2020-12-13 à 17 15 09" src="https://user-images.githubusercontent.com/47599816/102017371-ea317380-3d66-11eb-8c07-312b4b54cf37.png">

On peut voir que le training loss remonte avant de recommencer à descendre, ce qui n'est pas satisfaisant.

lr = 0.0001, epochs=100

<img width="497" alt="Capture d’écran 2020-12-13 à 17 16 09" src="https://user-images.githubusercontent.com/47599816/102017372-eb62a080-3d66-11eb-90e2-3f5c30c4275d.png">

Cette fois-ci, nous obtenons un apprentissage satisfaisant. Pour un learning rate de 0.01 et 100 époques, nous traçons également l'erreur sur le jeu de données de validation : 

<img width="468" alt="Capture d’écran 2020-12-13 à 17 27 41" src="https://user-images.githubusercontent.com/47599816/102017672-99bb1580-3d68-11eb-99c3-8c1b894d3950.png">


