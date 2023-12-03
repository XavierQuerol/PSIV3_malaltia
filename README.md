# PSIV3_malaltia


## Estructura
```
├───config.py: Fitxer de configuració
├───utils.py: Funcions de plots i dataset
├───baseline.ipynb: Hi ha tot el pipeline del basline
├───main_autoencoder.py: Per entrenar l'autoencoder
├───model_autoencoder.py: Model propi d'autoencoder
├───main_CNN_MlpClassifier.py: Per entrenar un classificador (propi o amb finetuning d'una resnet)
├───model_CNN_MlpClassfier.py: Model propi de classificador
├───final.py: Crossvalidation amb dades de Cropped per pacient. Avaluem model propi i resnet
├───gradient_boosting.ipynb: Ensemble de patches per pacient fent un Gradient Boosting
```