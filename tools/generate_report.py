from huggingface_hub import InferenceClient
import re
import json
import os
from tqdm import tqdm


def get_prompt(concepts, concepts_report):
    """
        Constructs a prompt for the language model to generate a structured medical report.
        
        Args:
            concepts (dict): Categorized medical concepts (e.g., {"finding": "category"})
            concepts_report (dict): Mapping of concepts to their source sentences from the text report
            
        Returns:
            str: A formatted prompt instructing the LLM how to generate the report
    """
    # Define the P.A.B.C.D.E.F protocol for chest x-ray review
    protocol='''

            Chest x-ray review is a key competency for medical students, junior doctors and other allied health professionals. Using P, A, B, C, D, E, F is a helpful and systematic method for chest x-ray review:

                P: projection views(PA, AP, lateral views or no views specified)
                        
                A: airways (intraluminal mass, narrowing, splayed carina)

                B: breathing (lungs, pulmonary vessels, pleural spaces)

                C: circulation (cardiomediastinal contour, great vessels)

                D: diaphragm and below (diaphragmatic paresis, pneumoperitoneum, gaseous distension, splenomegaly, calculi)

                E: external e.g. chest wall (ribs, shoulder girdles, fractures), soft tissues

                F: foreign material (devices, foreign bodies, gossypibomas)

             '''

    # Construct the full prompt with instructions and input data
    prompt = f'''
                You are given with the categorized medical concepts according to the following protocol: {protocol} and each concept mapped to its source sentence from the text report\n
                Your task is to generate structured medical reports according to the categories mentioned above in the protocol, given the structured concepts.\n
                Your generated report must have a json format like: {{"P": "Mention views information if no views specified then write 'No views'", "A": "Findings of concepts belonging to A", "B": "Findings of concepts belonging to B", "C":"Findings of concepts belonging to C", "D":"Findings of concepts belonging to D", "E":"Findings of concepts belonging to E", "F":"Findings of concepts belonging to F"}}.\n
                If from the given concepts any of the categories are missing have its report as "No findings", for example if there are no concepts from "A" then have the findings as "No Findings".\n
                You must write report in a medical radiologist style, in a descriptive way, DONT JUST AGGREGATE THE CONCEPTS!, DONT BRING IN ANY EXTRA MEDICAL FINDINGS.\n
                So when the concept-source sentence mapping is:{concepts_report} and the concept categories are: {concepts}, the generated structured report is:
            '''
    return prompt


def get_llm_response(prompt):
    """
    Get the response from LLM based on the prompt.
    
    Args:
        prompt (str): The formatted prompt for the language model
        
    Returns:
        str: The LLM's response containing the generated medical report
    """
    # Initialize the local inference client
    client = InferenceClient(base_url="http://localhost:8080/v1/")
    
    # Send the prompt to the language model
    response = client.chat.completions.create(
        model="tgi",
        messages=[{"role": "system", "content": "You are a helpful clinical assistant for writing medical reports."},
                  {"role": "user", "content": prompt}],
        stream=False
    )
    
    # Extract the response content, removing any thinking section
    output = response.choices[0].message.content.split('</think>')[1]
    return output

def generate_the_report(concepts):
    """
    Generate a structured medical report from categorized concepts.
    
    Args:
        concepts: Input parameter to initiate the report generation process
        
    Returns:
        dict: The generated structured report in JSON format
    """
    # Load previously saved categorized concepts and concept-report mappings
    with open(os.path.join(os.getcwd(), 'savings', 'categorized_concepts.json')) as f:
        categorized_concepts = json.load(f)
    with open(os.path.join(os.getcwd(), 'savings', 'concepts_output.json')) as f:
        concepts_report = json.load(f)
    
    # Generate the prompt and get LLM response
    user = get_prompt(categorized_concepts, concepts_report)
    outputs = get_llm_response(user)

    # Extract JSON data from the LLM response
    pattern = r'\{.*?\}'
    match = re.search(pattern, outputs, re.DOTALL)
    json_str = match.group(0) 
    texts = json.loads(json_str)

    # Save the structured report to a file
    file_path = os.path.join(os.getcwd(), 'savings', 'categorized_report.json')
    with open(file_path, 'w') as json_file:
        json.dump(texts, json_file, indent=4)
        
    return texts


