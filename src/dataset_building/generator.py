import random

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
    def _condition_classes(self, condition):
        if condition > 7:
            return 'Ottimo'
        elif condition > 4:
            return 'Buono'
        else
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
        condition_desc = random.choice(self.conditions[self._condition_classes(house_data['OverallCond'])])
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