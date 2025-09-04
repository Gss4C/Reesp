import data_func as daf
import pandas as pd

# Parameters #
features   = ["Id", "YearBuilt", "Heating", "GarageCars", "SalePrice", "Fireplaces", "LandSlope", "Street", "LotArea", "OverallCond"]
''' Features mapping
id: id,
YearBuilt: Anno di costruzione: year
Fireplaces: Numero camini
LotArea: Metri quadri
OverallCond: Condizioni
GarageCars: Bagni
Heating: Sistema di riscaldamento
Street: Ascensore S/N
LandSlope: Centro, Periferia, Mare
'''
dataset_id ={
    "path": "./data/HP-ADVT/train.csv",
    "img_path": "./data/images/house_data/"
}

if __name__ == '__main__':
    print("Hi, I'm the data-builder\nI will manipulate a standard dataset and an appropriate labelled image folder\nI will create the database for the dashboard")
    df = daf.read_clean_dataset(dataset_id["path"], features)
