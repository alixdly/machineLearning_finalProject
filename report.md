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
* **Notre but est de prévoir des données de traffic (colonne Volume) pour chaque emplacement dans chaque direction à cet emplacement i.e. par couple (location_name,Direction) à une fréquence que nous avons encore à définir**


# Partie 2 : Un premier modèle : Convolutional Neural Network (CNN)

En premier "essai", nous avons décidé d'implémenter le CNN le plus "basique", celui de la correction du dernier DM, pour d'abord être sur de bien avoir compris la correction mais aussi pour prendre en main les données et faire plusieurs essais : prévoir la prochaine heure en fonction de la journée, prévoir le prochain jour en fonction de la semaine, etc... pour mieux connaître les données.



