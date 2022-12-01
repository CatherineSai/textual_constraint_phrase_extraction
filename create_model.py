import spacy
from phrase_extraction import *

# Create and save model with phrase_spans component - only needs to run once
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('merge_noun_chunks')
nlp.add_pipe('merge_entities')
nlp.add_pipe('phrase_spans')
nlp.to_disk('./models/method_a')

# Afterwards, to use the above model with pipeline, simply load the previously saved model
# nlp = spacy.load('models/method_a')