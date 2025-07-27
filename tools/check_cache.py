import os
import json


def check_the_cache(concepts):
    file_path = os.path.join(os.getcwd(), 'cached_vocab.json')

    with open (file_path) as f:
        vocab = json.load(f)

    file_path = os.path.join(os.getcwd(), 'savings', 'concepts_output.json')
    with open(file_path) as f:
        current_concepts = json.load(f)

    new_concept_out = {}
    savings={}
    
    for concept, sentence in current_concepts.items():
        if concept in vocab.keys():
            savings[concept] = vocab[concept]
        else:
            new_concept_out[concept] = sentence
    
    categorized_save_path = os.path.join(os.getcwd(), 'savings', 'categorized_concepts.json')
    new_concept_output_path = os.path.join(os.getcwd(), 'savings', 'new_concepts_output.json')

    with open(categorized_save_path, 'w') as f:
        json.dump(savings, f, indent=4)

    with open(new_concept_output_path, 'w') as f:
        json.dump(new_concept_out, f, indent=4)

    return "Done"

