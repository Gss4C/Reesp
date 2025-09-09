import random
#import json
import data_func as daf
import generator as gen
import pandas as pd

def quality_check(description_df: pd.DataFrame):
    '''Controllo delle descrizioni'''
    print("QUALITY METRICS:")
    print(f"Total descriptions: {len(description_df)}")
    #print(f"Avg word count: {description_df['word_count'].mean():.1f}")
    #print(f"Word count range: {description_df['word_count'].min()}-{description_df['word_count'].max()}")
    
    # Check for duplicates
    duplicates = description_df['description'].duplicated().sum()
    print(f"Duplicate descriptions: {duplicates} ({duplicates/len(description_df)*100:.1f}%)")
    
    # Sample random descriptions
    print("\RANDOM SAMPLES:")
    samples = description_df.sample(5)
    for idx, row in samples.iterrows():
        print(f"\n{row['Id']}: {row['description']}")
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
#output_paths='data/output-datasets/train_img-desc-data.csv'
output_paths = ['data/output-datasets/total_df.csv', 'data/output-datasets/house_df.csv', 'data/output-datasets/desc_df.csv', 'data/output-datasets/img_df.csv']

if __name__ == '__main__':
    print("Hi, I'm the data-builder\nI will manipulate a standard dataset and an appropriate labelled image folder\nI will create the database for the dashboard")
    df = daf.read_clean_dataset(dataset_id["path"], features)
    desc_generator = gen.Descriptor()

    descriptions = []
    for idx, row in df.iterrows():
        #if idx < 200: #test
        house_data = row.to_dict()
        description = desc_generator.description_assembly(house_data)
        descriptions.append({
            'Id': idx+1,
            'description': description,
            #'length': len(description.split())
        })

        # Print progresso
        if idx % 100 == 0:
            print(f'generated {idx} descriptions')
    
    description_df = pd.DataFrame(descriptions)
    description_df.to_csv('data/HP-ADVT/descriptions.csv', index = False)
    quality_check(description_df)

    test_obj = gen.DatasetMerger(image_folder = 'data/images/house_data', 
                                 dataframes   = ['data/HP-ADVT/train.csv', 'data/HP-ADVT/descriptions.csv'],
                                 features     = features)
    df_list = test_obj.dataset_merging()
    
    for single_df, output_path in zip(df_list, output_paths):
        single_df.to_csv(output_path)
        print('Salvato il dataset in: ', output_path)    

    #df_test.to_csv(output_path)
    #print('Salvato il dataset in: ', output_path)