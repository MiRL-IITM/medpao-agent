from mcp.server.fastmcp import FastMCP
import os
import pandas as pd
from urllib import parse
from huggingface_hub import InferenceClient
import regex as re
import json

from nltk.stem import WordNetLemmatizer
from tools.get_concept import get_the_concept
from tools.ontology_mapping import OntologyMapper
from tools.filter_ontologies import filter_the_ontology
from tools.categorize_concepts import categorize_the_concepts
from tools.generate_report import generate_the_report
from tools.check_cache import check_the_cache

mcp = FastMCP("BMI Server")

print(f"Starting server {mcp.name}")


@mcp.tool()
def get_concept(medical_report:str)->list:
    """
    Get the medical concepts and their source sentence from the given medical report. Use this only if original medical report is given.
    The tool returns a dictionary consisting of extracted concept and source sentence, also the filename string where the output is saved
    """
    print("============================================Called getConcept func=============================================")
    concepts = get_the_concept(medical_report)
    return concepts, 'savings/concepts_output.json'


@mcp.tool()
def check_cache(concepts)->str:
    """
    Used to check if the extracted concepts are already present in local files. 
    The tool acts upon raw concepts to check whether they are already present in system vocabulary.

    """
    print("=========================================Checking cache===================================================")
    return check_the_cache(concepts)


@mcp.tool()
def ontology_mapping(concepts)->list:
    """
    Get SNOMEDCT and RADLEX ontology mapping for each concept in concepts in the given dictionary keys. This is the first step after getting the concepts.
    The tool returns a dictionary consisting of concept and its mapped ontologies, also the filename string where the output is saved.
    """
    print("===========================================Doing Ontology Mapping===================================================")
    
    with open('apis.json') as f:
        apis = json.load(f)

    # Extract the API key from the apis.json file
    my_api_key = apis.get('API_KEY')
  
    mapper = OntologyMapper(api_key=my_api_key)
    final_mappings = mapper.map_concepts(concepts)

    if final_mappings is None:
        return 'concepts already in vocabulary', 'skipping the step'

    return final_mappings, 'savings/ontologyMapping.csv'


@mcp.tool()
def filter_ontology(mapped_ontologies:str)->list:
    """
    filtering the mapped ontologies into PRIMARY and SECONDARY literal for a concept based on abnormality, severity and anatomy.
    The tool returns a list of filtered ontologies and also the string consisting of saved filename.
    """
    filtered_ontologies = filter_the_ontology(mapped_ontologies)
    return filtered_ontologies, 'savings/segregated.json'
    


@mcp.tool()
def categorize_concepts(filtered_ontologies:str)->list:
    """
    Categorizing the extracted concepts with filtered ontologies into multiple categories according to the protocol.
    The tool returns a dictionary consisting of concept and its category mapping and also the filename string where the output is saved.
    """
    print("=========================================Categorizing the concepts===================================================")
    # Pass filtered_ontologies to the function instead of "start"
    concepts = categorize_the_concepts(filtered_ontologies)
    return concepts, 'savings/categorized_concepts.json'

@mcp.tool()
def generate_report(categorized_concepts:str)->list:
    """
    Generate the final structured report based on the categorized concepts.
    The tool returns a dictionary consisting of the structured report according to protocol and also the filename string where the output is saved.
    """
    print("=========================================Generating the report===================================================")
    report = generate_the_report(categorized_concepts)
    return report, 'savings/categorized_report.json'






if __name__ == "__main__":
    mcp.run(transport="stdio")
