from huggingface_hub import InferenceClient
import pandas as pd
import json
from tqdm import tqdm
import re
import os


def parsing(text):
    """
    Extract and parse JSON data from the LLM response text.
    
    Args:
        text (str): The raw text response from the language model
        
    Returns:
        dict or str: Parsed JSON data if successful, error message otherwise
    """
    # Replace single quotes with double quotes for valid JSON
    text = text.replace("'", '"')
    
    # Use regex to find JSON content within the response
    match = re.search(r'\{.*\}', text, re.DOTALL)

    if match:
        json_str = match.group()
        try:
            # Attempt to parse the JSON string
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            # Return error message if JSON parsing fails
            return "NULLA"
    else:
        # Return error message if no JSON pattern is found
        return "NULLB"


def get_prompt(medical_final):
    """
    Construct a prompt for the language model with example reports and their
    extracted concepts to guide the model in concept extraction.
    
    Args:
        medical_final (str): The medical report text to extract concepts from
        
    Returns:
        str: A formatted prompt with examples and the target report
    """
    # Example reports used to demonstrate concept extraction to the model
    rep1 = 'No focal consolidation, pleural effusion, or pneumothorax is seen. Heart and mediastinal contours are within normal limits.  There is no evidence for pulmonary edema.'
    rep2 = 'PA and lateral views of the chest provided.   Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax. The cardiomediastinal silhouette is notable for an unfolded  thoracic aorta.  Imaged osseous structures are intact.  No free air below the  right hemidiaphragm is seen.'
    rep3 = "Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.  No new consolidations are identified.  There is no  pleural effusion or pneumothorax.  Cardiac and mediastinal silhouettes are  normal.  No acute fractures are identified."
    rep4 = "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.  No large effusion or pneumothorax.   Cardiomediastinal silhouette is normal.  Bony structures are intact."
    
    # Example concepts mapped to their source sentences for each report
    conc1 = {
        "no focal consolidation" : "No focal consolidation, pleural effusion, or pneumothorax is seen.", 
        "no pleural effusion" : "No focal consolidation, pleural effusion, or pneumothorax is seen.", 
        "no pneumothorax" : "No focal consolidation, pleural effusion, or pneumothorax is seen.", 
        "normal heart contours" : "Heart and mediastinal contours are within normal limits.", 
        "normal mediastinal contours" : "Heart and mediastinal contours are within normal limits.", 
        "no pulmonary edema" : " There is no evidence for pulmonary edema."
    }


    conc2 = {
    "PA and lateral views": "PA and lateral views of the chest provided.",
    "low lung volumes": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "likely no focal consolidation": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "likely no effusion": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "likely no pneumothorax": "Lung volumes are somewhat low  though allowing for this, there is no focal consolidation, effusion, or  pneumothorax.",
    "noteable cardiomediastinal silhouette": "The cardiomediastinal silhouette is notable for an unfolded  thoracic aorta. ", 
    "intact osseous structures": "Imaged osseous structures are intact.",
    "no free air below right hemidiaphragm": " No free air below the  right hemidiaphragm is seen."
    }

    conc3 = {
        "improving left lower lobe opacity":"Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.",
            "likely due to resolving pneumonia": "Previously visualized left lower lobe opacity has improved and is suggestive  of resolving pneumonia.",
            "no new consolidations": "No new consolidations are identified.",
            "no pleural effusion": "There is no  pleural effusion or pneumothorax.",
            "normal cardiac silhouette": "Cardiac and mediastinal silhouettes are  normal.",
            "normal mediastinal silhouette": "Cardiac and mediastinal silhouettes are  normal.",
            "no acute fractures": "No acute fractures are identified."
    }

    conc4 = {
            "PA and lateral views": "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.",
            "bilateral lower lung patchy opacities with left lower lung involvement": "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.",
            "likely due to pneumonia": "PA and lateral views of the chest provided demonstrate bilateral  lower lung patchy opacities most confluent in the left lower lung which is  compatible with pneumonia.",
            "no large effusion": "No large effusion or pneumothorax.",
            "no pneumothorax": "No large effusion or pneumothorax.",
            "normal cardiomediastinal silhouette": "Cardiomediastinal silhouette is normal.",
            "normal bony structures": "Bony structures are intact."

    }

    # Construct the prompt with instruction, examples, and the target report
    instr = "Assume you are an expert radiology practitioner.you have to extract the medical concepts from the given report and also indicate from which sentence the concept was picked.\n"
    prompt1 = f"Extracted concepts and their corresponding sentence from {rep1} are {conc1} ##\n"
    prompt2 = f"Extracted concepts and their corresponding sentence from {rep2} are {conc2} ##\n"
    prompt3 = f"Extracted concepts and their corresponding sentence from {rep3} are {conc3} ##\n"
    prompt4 = f"Extracted concepts and their corresponding sentence from {rep4} are {conc4} ##\n"
  
    prompt = f"Extracted concepts and their corresponding sentence from {medical_final} are "
    
    # Combine all parts into the final prompt
    eval_prompt = instr+prompt1+prompt2+prompt3+prompt4+prompt
    return eval_prompt

def get_llm_response(medical_report):
    """
    Send the medical report to the language model and get a response.
    
    Args:
        medical_report (str): The medical report text to process
        
    Returns:
        str: The LLM's response containing extracted concepts
    """
    # Initialize the local inference client
    client = InferenceClient(base_url="http://localhost:8081",)
    
    # Generate the prompt and send it to the language model
    user = get_prompt(medical_report)
    response = client.text_generation(user, max_new_tokens=300)
    return response

def get_the_concept(medical_report):
    """
    Extract medical concepts from a medical report and save them to a file.
    
    Args:
        medical_report (str): The medical report text to extract concepts from
        
    Returns:
        dict: The extracted concepts mapped to their source sentences
    """
    # Get the response from the language model
    response = get_llm_response(medical_report)
    
    # Parse the response to extract the JSON data
    llm_out = parsing(response)
    
    # Create the savings directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), 'savings'), exist_ok=True)
    
    # Save the extracted concepts to a JSON file
    with open(os.path.join(os.getcwd(), 'savings', 'concepts_output.json'), 'w') as f:
        json.dump(llm_out, f, indent=4)
    
    return llm_out







