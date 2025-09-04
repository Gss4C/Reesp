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
            "Splendido appartamento", "Magnifico bilocale", "Elegante trilocale",
            "Luminoso appartamento", "Affascinante dimora", "Esclusiva proprietà", "Mirabolante immobile", "Strepitoso cementoide"
        ]
        self.conditions = {
            'Ottimo': ['in ottime condizioni', 'perfettamente ristrutturato', 'in condizioni impeccabili'],
            'Buono': ['in buone condizioni', 'ben tenuto', 'pronto da abitare'],
            'Da ristrutturare': ['da ristrutturare', 'con potenziale di personalizzazione', 'da rimodernare']
        }
        self.location_descriptors = {
            'Centro Storico': ['nel prestigioso centro storico', 'nel cuore della città', 'in zona centrale'],
            'Periferia': ['in zona residenziale', 'in tranquilla zona periferica', 'in area verde'],
            'Mare': ['a pochi passi dal mare', 'in zona costiera', 'con vista mare']
        }
    def _generate_
    