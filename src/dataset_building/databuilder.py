import data_func as daf
import pandas as pd

# Parameters #
features   = ["Id", "YearBuilt", "Heating", "GarageCars", "SalePrice", "Fireplaces", "LandSlope", "Street", "LotArea", "OverallCond"]
''' Features mapping
id: id,
--- Pezzi pre-montati --------------------
OverallCond: Condizioni
LandSlope:   Centro, Periferia, Mare
--- Presi dal database -------------------
GarageCars: stanze        - rooms_desc
LotArea:    Metri quadri  - sqm
SalePrice:  Prezzo vend.  - price_desc
------------------------------------------
YearBuilt:  Anno          - year
Fireplaces: Numero camini - fireplaces
Heating:    Riscaldamento - heating
Street:     Ascensore S/N - elevator
'''
dataset_id ={
    "path": "./data/HP-ADVT/train.csv",
    "img_path": "./data/images/house_data/"
}

if __name__ == '__main__':
    print("Hi, I'm the data-builder\nI will manipulate a standard dataset and an appropriate labelled image folder\nI will create the database for the dashboard")
    df = daf.read_clean_dataset(dataset_id["path"], features)
