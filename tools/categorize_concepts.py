import json
from huggingface_hub import InferenceClient
import re
import os
from tqdm import tqdm

def load_json_file(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json_file(file_path, data, indent=4):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=indent)

def update_and_save_json(file_path, new_data):
    """Update existing JSON file with new data or create new file."""
    existing_data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    
    existing_data.update(new_data)
    save_json_file(file_path, existing_data)

def create_categorization_prompt(concepts_report, ontology_data):
    """Create system message and prompt for categorizing concepts."""
    sys_msg = '''You are radiology expert with the following knowledge:
            Chest x-ray review is a key competency for medical students, junior doctors and other allied health professionals. Using P, A, B, C, D, E, F is a helpful and systematic method for chest x-ray review:
            
            P: projection views(PA, AP, lateral views)

            A: airways (intraluminal mass, narrowing, splayed carina)

            B: breathing (lungs, pulmonary vessels, pleural spaces)

            C: circulation (cardiomediastinal contour, great vessels)

            D: diaphragm and below (diaphragmatic paresis, pneumoperitoneum, gaseous distension, splenomegaly, calculi)

            E: external e.g. chest wall (ribs, shoulder girdles, fractures), soft tissues

            F: foreign material (devices, foreign bodies, gossypibomas)

            '''
            
    prompt = f''' You will be given two data: the concept-sourcesentence from reports and the ontology mapping of each concept. based on these data and given protocol category, categorize all the given concepts into PABCDEF.
            Below is the mapping of medical concepts and their SNOMEDCT ontology. The mapping is a list of concepts and follows structure as: {{'medical concept': the extracted medical concept, 'Primary literal': the main pathology or abnormality found, 'Secondary literal': severity or anatomical structure.}} Each primary or secondary literal mapping follow format like: SNOMEDCT ONTOLOGY(DESCRIPTION)-->ANCESTORS. Hence using the given mapping categorize each medical concept into ABCDEF categories.
            For example:
            INPUT:
            [{{ 
                "medical-concept": "cardiomegaly",
                "Primary literal": [
                    "CARDIOMEGALY(Cardiomegaly (disorder))-->[Structural disorder of heart, Heart disease, Disorder of mediastinum, Cardiac finding, Disorder of cardiovascular system, Disorder of thorax, Mediastinal finding, Disorder of body system, Cardiovascular finding, Viscus structure finding, Disease, Clinical finding, Finding of region of thorax, Disorder of thoracic segment of trunk, Disorder of trunk, Finding of upper trunk, Finding of trunk structure]"
                ],
                "Secondary literal": []
            }},
            {{
                "medical-concept": "pleural effusion",
                "Primary literal": [
                    "PLEURAL EFFUSION(Pleural effusion (disorder))-->[Disorder of pleura and pleural cavity, Disorder of thorax, Disorder of lower respiratory system, Finding of region of thorax, Disorder of thoracic segment of trunk, Disorder of respiratory system, Respiratory finding, Disorder of body system, Disorder of trunk, Finding of upper trunk, Disease, Finding of trunk structure, Clinical finding]"
                ],
                "Secondary literal": [
                    "EFFUSION(Effusion (substance))-->[Body fluid, Body material, Liquid substance, Substance categorized by physical state, Material, Body substance, Substance]",
                    "EFFUSION(Effusion (morphologic abnormality))-->[Fluid disturbance, Mechanical abnormality, Morphologically abnormal structure, Morphologically altered structure, Body structure]"
                ]
            }}]
            OUTPUT:
            json {{"cardiomegaly": "C", "Pleural Effusion": "B"}}
            Similarly when concept-source sentence is {concepts_report} and the ontology mapping is:{ontology_data}
            OUTPUT:'''
    
    return sys_msg, prompt

def parse_model_response(output):
    """Extract and parse JSON from model response."""
    pattern = r'\{.*?\}'
    try:
        match = re.search(pattern, output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        return {}
    except Exception as e:
        print(f"Error parsing model response: {e}")
        return {}

def categorize_the_concepts(start):
    """
    Categorize medical concepts into P/A/B/C/D/E/F categories using an LLM.
    
    Args:
        start: A trigger parameter (not functionally used)
        
    Returns:
        dict: Categorized medical concepts
    """
    # Setup paths
    base_dir = os.getcwd()
    segregated_file = os.path.join(base_dir, 'savings', 'segregated.json')
    concepts_output_file = os.path.join(base_dir, 'savings', 'concepts_output.json')
    categorized_file = os.path.join(base_dir, 'savings', 'categorized_concepts.json')
    vocab_file = os.path.join(base_dir, 'cached_vocab.json')
    
    # Load input data
    ontology_data = load_json_file(segregated_file)
    if ontology_data is None:
        print(f"File {segregated_file} does not exist. Please check the path.")
        return {'concepts already in vocabulary': 'skipped'}
    
    concepts_report = load_json_file(concepts_output_file)
    if concepts_report is None:
        print(f"File {concepts_output_file} does not exist. Please check the path.")
        return {'concepts already in vocabulary': 'skipped'}
    
    # Setup inference client
    client = InferenceClient(base_url="http://localhost:8080/v1/")
    
    # Create prompt and get model response
    sys_msg, prompt = create_categorization_prompt(concepts_report, ontology_data)
    
    try:
        response = client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        output = response.choices[0].message.content.strip()
        
        # Parse response
        categorized_concepts = parse_model_response(output)
        
        # Save results
        update_and_save_json(categorized_file, categorized_concepts)
        update_and_save_json(vocab_file, categorized_concepts)
        
        return categorized_concepts
        
    except Exception as e:
        print(f"Error during model inference: {e}")
        return {}





