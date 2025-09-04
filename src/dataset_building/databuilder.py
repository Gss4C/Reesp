import random
import data_func as daf
import generator as gen
import pandas as pd

# Parameters #
features   = ["Id", "YearBuilt", "Heating", "GarageCars", "SalePrice", "Fireplaces", "LandSlope", "Street", "LotArea", "OverallCond"]
''' Features mapping
id: id
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
    desc_generator = gen.Descriptor()

    descriptions = []
    for idx, row in df.iterrows():
        if idx < 200: #test
            house_data = row.to_dict()
            print(type(house_data))
            description = desc_generator.description_assembly(house_data)
            descriptions.append({
                'property_id': idx+1,
                'description': description,
                'length': len(description.split())
            })

        # Print progresso
        if idx % 50 == 0:
            print(f'generated {idx} descriptions')
    
    description_df = pd.DataFrame(descriptions)
    description_df.to_csv('data/HP-ADVT/descriptions.csv', index = False)

    # Quality check
    print(f"\n=== GENERATION STATS ===")
    print(f"Total descriptions: {len(descriptions)}")
    print(f"Average length: {description_df['length'].mean():.1f} words")
    print(f"Length range: {description_df['length'].min()}-{description_df['length'].max()} words")

    #Samples
    print(f"\n=== SAMPLE DESCRIPTIONS ===")
    for i in range(3):
        idx = random.randint(0, len(descriptions)-1)
        print(f"\n{i+1}. {descriptions[idx]['description']}")
        