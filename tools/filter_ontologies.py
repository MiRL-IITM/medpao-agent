import json
from huggingface_hub import InferenceClient
import re
import dspy
import requests
import pandas as pd
import os
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

def get_message(concept, mappings):
    """
        Construct system and prompt messages for the language model to categorize medical ontology mappings.
        
        Args:
            concept (str): The medical concept being analyzed
            mappings (list): The SNOMED CT mappings and their ancestors for the concept
            
        Returns:
            tuple: System message and prompt message for the language model
    """
    sys_msg = ("You are a medical expert. There are a few medical concepts mapped to the SNOMED CT ontology, along with their descriptions and ancestors.\n"
            "The conncepts are represented as 'concepts': [MATCHINGS1-->[ANCESTOR1, ANCESTOR2,...], MATCHINGS2-->[ANCESTOR1, ANCESTOR2,...]]\n"
            "In the above, 'concepts' refers to actual concept we have at hand, MATCHINGS1, MATCHINGS2 are the mappings of the concept to SNOMED CT ontology, and ANCESTOR1, ANCESTOR2,... are the ancestors of the concept in the ontology.\n"
            "The ancestors are provided for your understanding of the concept and its mappings, but they should not be altered or interpreted as additional concepts.\n"
            "Please be aware that the concepts may have multiple mappings, and each mapping may have multiple ancestors.\n"
            "Do not change anything in the ontology information provided. That is MATCHINGS1-->[ANCESTOR1, ANCESTOR2,...] should remain same , do not mis interprete ancestors as extra concepts they are just for reading, do not edit them.\n"
            "Your task is to categorize the given SNOMED CT terms into **primary literal** and **secondary literal** based on their relevance to the concept.\n"
            "The **primary literal** is the main term that defines the condition or disorder, while the **secondary literal** provides additional context or location.\n")
    
    prompt=(
            "Provide the output in following json format:\n"
            "{{'medical-concept': The provided medical concept\n 'Primary literal': mapped ontology term\n 'Secondary literal': All remaining mapping}}\n"
            "For example:\n"
            "- **AC JOINT ARTHRITIS** is mapped to:\n"
            " [ARTHRITIS(Arthritis (disorder))-->[ANCESTOR1,...], JOINT(Joint structure (body structure))-->[ANCESTOR1, ANCESTOR2,...]]\n"
            "Hence, **AC JOINT ARTHRITIS**: Primary literal: [ARTHRITIS(Arthritis (disorder))-->[ANCESTOR1,...], Secondary literal: JOINT(Joint structure (body structure))-->[ANCESTOR1, ANCESTOR2,...]\n\n"
            "Similarly, provide the mapping for the following concept:\n"
            f"- **{concept}**:\n"
            f"{mappings}\n"    
        )

    return sys_msg , prompt


def get_llm_response(concept, mappings, error=None):
    """
    Get the response from LLM based on the concept and its mappings.
    
    Args:
        concept (str): Medical concept to categorize
        mappings (list): SNOMED CT mappings for the concept
        error (Exception, optional): Previous error to help with response correction
        
    Returns:
        str: The LLM response with categorization results
    """
    # Initialize the local inference client
    client = InferenceClient(base_url="http://localhost:8080/v1/")
    sys_msg, prompt = get_message(concept, mappings)
    
    # If there was an error in previous processing, add it to the prompt
    if error:
        prompt = f"For your previous output i got this error: {error}, now generate the output in correct format for :\n{prompt}"

    # Call the language model
    response = client.chat.completions.create(
        model="tgi",
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": prompt}],
        stream=False
    )
    
    # Extract the content from the response, removing any <think> tags
    output = response.choices[0].message.content.split('</think>')[1].strip()
    return output

def filter_the_ontology(mappe_ontologies)->list:
    """
    Filter the mapped ontologies into PRIMARY and SECONDARY literals for a concept 
    based on abnormality, severity and anatomy.
    
    Args:
        mappe_ontologies (dict): Dictionary of concepts and their mapped ontologies
        
    Returns:
        list: Filtered ontology categorizations
    """
    
    print("============================================Enterd filtering Ontologies===============================================")
    filtered = []
    
    # Check if the ontology mapping CSV exists
    csv_path = os.path.join(os.getcwd(), 'savings', 'ontologyMapping.csv')
    if not os.path.exists(csv_path):
        return ['concepts already in local files', 'skipped']
    
    else:
        # Load ontology mappings from CSV
        df = pd.read_csv(csv_path)
        concepts = set(df['Concept'].tolist())
        mapped_ontologies = {}

        # Extract concept mappings and ancestors from the DataFrame
        for concept in concepts:
            mapped_ontologies[concept] = []
            concept_df = df[df["Concept"] == concept]
            for idx, row in concept_df.iterrows():
                this_str = str(row['mapped term']) + '->' + str(row['ancestors'][1:-1].replace("'", ""))
                mapped_ontologies[concept].append(this_str)
        
        # Process each concept with the language model
        for concept, match in tqdm(mapped_ontologies.items()):
            # Get initial LLM response
            output = get_llm_response(concept, match)
            pattern = r'\{.*?\}'
            match = re.search(pattern, output, re.DOTALL)
            
            # Try to parse the JSON response, retry if there's an error
            try:
                json_str = match.group(0)
                t = json.loads(json_str)  
            except Exception as e:
                # Request a corrected response if the first one had errors
                output = get_llm_response(concept, match, error=e)
                pattern = r'\{.*?\}'
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    json_str = match.group(0)
                else:
                    json_str = output
            filtered.append(json_str)
        
        # Convert string JSON responses to actual JSON objects
        json_objects = []
        for json_str in filtered:
            try:
                json_objects.append(json.loads(json_str))
            except Exception as e:
                json_objects.append(json_str)

        # Save the results to a JSON file
        file_path = os.path.join(os.getcwd(), 'savings', 'segregated.json')
        with open(file_path, 'w') as json_file:
            json.dump(json_objects, json_file, indent=4)

        print(f"SEGREGATED ONTOLOGIES ARE SAVED TO: {file_path}")

        return filtered

