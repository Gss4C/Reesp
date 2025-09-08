import random
import pandas as pd
import os
import re
from pathlib import Path
import data_func as dfx

class Descriptor:
    '''
    Object able to create description for houses in a combinatory random way
    '''
    def __init__(self):
        #initialize string
        self.templates = [
            #TODO migliorare con tutte le funzioni fatte
            "{intro} di {sqm} mq {location_desc}. {condition_desc} {rooms_desc} {features_desc} {price_desc}",
            "{property_type} {condition_desc} di {sqm} metri quadrati {location_desc}. {rooms_desc} {features_desc} {price_desc}",
            "In vendita {intro} {location_desc}. {sqm} mq, {rooms_desc}, {condition_desc}. {features_desc} {price_desc}"
        ]
        self.intros = [
            "Splendido appartamento", "Luminoso appartamento", "Affascinante dimora", "Esclusiva proprietà", "Mirabolante immobile", "Strepitoso cementoide"
        ]
        self.conditions = {
            'Ottimo': ['in ottime condizioni', 'perfettamente ristrutturato', 'in condizioni impeccabili'],
            'Buono': ['in buone condizioni', 'ben tenuto', 'pronto da abitare'],
            'Da ristrutturare': ['da ristrutturare', 'con potenziale di personalizzazione', 'da rimodernare']
        }
        self.location_descriptors = {
            'Centro': ['nel prestigioso centro storico', 'nel cuore della città', 'in zona centrale'],
            'Periferia': ['in zona residenziale', 'in tranquilla zona periferica', 'in area verde'],
            'Mare': ['a pochi passi dal mare', 'in zona costiera', 'con vista mare']
        }
    def _generate_features_description(self, data: dict):
        '''
        Genera descrizione per: camini, ascensore, riscaldamento
        '''
        features = []

        #anno costruzione
        features.append(f'Anno di costruzione: {data['YearBuilt']}')
        #riscaldamento
        features.append(f'riscaldamento a {data['Heating']}')
        #ascensore
        if data['Street'] == "Pave":
            features.append('servito da ascensore')
        #camini
        if data['Fireplaces'] != 0:
            if data['Fireplaces'] == 1:
                features.append('con un camino')
            else:
                features.append(f'con {data['Fireplaces']} camini')
        return ', '.join(features)
    def _generate_price_description(self, price) -> str: 
        ''' Genera descrizione del prezzo'''
        if price > 400000:
            return f"prezzo richiesto: {price:,}€. Investimento prestigioso di alta qualità"
        elif price > 200000:
            return f"prezzo proposto di {price:,}€."
        else:
            return f"opportunità irrinunciabile a {price:,}€."

    def _generate_rooms_description(self, rooms) -> str:
        match rooms:
            case 0: 
                return "zerolocale"
            case 1: 
                return "monolocale con un bagno"
            case 2: 
                return "bilocale con due bagni"
            case 3: 
                return "trilocale con 12 bagni"
            case 4: 
                return "quadrilocale con un bagno"
            case _:
                return "casolare enorme con bagni"
    def _condition_classes(self, data) -> str:
        
        if data['OverallCond'] > 7:
            return 'Ottimo'
        elif data['OverallCond'] > 4:
            return 'Buono'
        else:
            return 'Da ristrutturare'
    def _location_classes(self, location):
        if location == "Gtl":
            return "Centro"
        elif location == "Mod":
            return 'Periferia'
        elif location == "Sev":
            return 'Mare'
    
    def description_assembly(self, house_data: dict) -> str:
        '''
        Funzione che raccoglie tutti i pezzi delle varie descrizioni e 
        li compone in maniera sensata dando in output una roba leggibile.
        Parameters:
        house_data: dizionario contenente tutte le informazioni standard, proveniente da un to_dict() di una riga
        '''
        template = random.choice(self.templates)

        #seleziona altri pezzi dagli attributi
        intro          = random.choice(self.intros)
        condition_desc = random.choice(self.conditions[self._condition_classes(house_data)])
        location_desc  = random.choice(self.location_descriptors[self._location_classes(house_data['LandSlope'])])
        features_desc  = self._generate_features_description(house_data)
        rooms_desc     = self._generate_rooms_description(house_data.get('GarageCars'))
        price_desc     = self._generate_price_description(house_data.get('SalePrice', 0))
    
        #filling template
        description = template.format(
            intro          = intro,
            sqm            = house_data.get('LotArea', 'N/A'),
            location_desc  = location_desc,
            condition_desc = condition_desc,
            rooms_desc     = rooms_desc,
            features_desc  = features_desc,
            price_desc     = price_desc,
            property_type  = rooms_desc
        )

        return description.strip()

class DatasetMerger:
    '''classe che fa un merge di datasets testuali ed immagini basandosi sull'id'''
    def __init__(self, output_path: str, image_folder: str, dataframes, features: list, test: bool=False):
        self.output_path  = output_path
        self.image_folder = Path(image_folder)
        self.dataframes   = dataframes
        self.features     = features
        self.test         = test

    def create_image_dict(self, id_pattern: str = r'(\d+)', ):
        image_dict = {}

        if not self.image_folder.exists():
            print('Cartella non trovata. \nReturn dizionario vuoto')
            return image_dict
        for image_path in self.image_folder.iterdir():
            match_id = re.search(id_pattern, image_path.stem)
            if match_id:
                image_id = match_id.group(1)
                if image_id not in image_dict:
                    image_dict[image_id] = []
                image_dict[image_id].append(str(image_path))
                
        print(f"Trovati {len(image_dict)} ID unici con immagini")
        total_images = sum(len(paths) for paths in image_dict.values())
        print(f"Totale immagini: {total_images}")
        
        return image_dict
    
    def load_text_dataset(self, features):
        """Carica dataset"""
        try:
            if self.dataframes[0].endswith('.csv'):
                
                df_house = dfx.read_clean_dataset(self.dataframes[0], features)
                
                #df_house  = pd.read_csv(self.dataframes[0])
                df_desc  = pd.read_csv(self.dataframes[1])

            elif self.dataframes[0].endswith(('.xlsx', '.xls')):
                df_house = pd.read_excel(self.dataframes[0])
                df_desc = pd.read_excel(self.dataframes[1])
            else:
                raise ValueError("Formato file non supportato. Usa CSV o Excel.")
            
            return df_house, df_desc
            
        except Exception as e:
            print(f"Errore nel caricamento del dataset: {e}")
            return None
    def images_dataframer(self, image_dict):
        pre_dataframe = []
        for house_id, paths in image_dict.items():
            for path in paths:
                pre_dataframe.append({
                    'Id': int(house_id),
                    'image_path': path
                })
        return pd.DataFrame(pre_dataframe)

    def dataset_merging(self):
        dizionario_immagini = self.create_image_dict()
        df_img = self.images_dataframer(dizionario_immagini)
        df_house, df_desc = self.load_text_dataset(self.features)
        text_df = pd.DataFrame.merge(df_house, df_desc, on='Id', how='inner')

        if self.test:
            print('\n\n DATASET HOUSING')
            print(df_house)
            print('\n\n DATASET DESCRIZIONI')
            print(df_desc)
            print('\n\n DATASET Merged')
            print(text_df)
        
        total_df = pd.DataFrame.merge(text_df, df_img, on = 'Id', how = 'inner')
        return total_df