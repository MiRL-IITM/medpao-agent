import requests
from urllib import parse
import pandas as pd
from typing import List, Dict, Any, Optional
import os
from tqdm import tqdm
import json
from nltk.stem import WordNetLemmatizer

class OntologyMapper:
    """
    A class to map medical concepts to specified ontologies using the BioPortal Annotator API.

    It encapsulates the logic for making API requests, parsing responses, fetching
    ancestor terms, and formatting the results into a pandas DataFrame and a dictionary.
    """
    
    BASE_URL = 'http://data.bioontology.org/annotator?'

    def __init__(self, api_key: str, ontologies: List[str] = None, output_file: str = 'ontologyMapping.csv'):
        """
        Initializes the OntologyMapper.

        Args:
            api_key (str): Your BioPortal API key.
            ontologies (List[str], optional): A list of ontologies to search (e.g., ['SNOMEDCT', 'RADLEX']). 
                                              Defaults to ['SNOMEDCT', 'RADLEX'].
            output_file (str, optional): The path to save the output CSV file. 
                                         Defaults to 'ontologyMapping.csv'.
        """
        if not api_key:
            raise ValueError("API key is required.")
            
        self.api_key = api_key
        self.ontologies = ontologies if ontologies is not None else ['SNOMEDCT', 'RADLEX']
        self.output_directory = output_file
        print(f"OntologyMapper initialized to search in: {', '.join(self.ontologies)}")

    def _fetch_ancestors(self, ancestors_url: str) -> List[str]:
        """
        Private helper method to fetch ancestor terms for a given ontology class.
        """
        try:
            response = requests.get(f"{ancestors_url}?apikey={self.api_key}")
            response.raise_for_status()  # Will raise an HTTPError for bad responses (4xx or 5xx)
            ancestors_data = response.json()
            return [ancestor.get('prefLabel', 'N/A') for ancestor in ancestors_data]
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch ancestors from {ancestors_url}: {e}")
            return ['Error fetching ancestors']

    def _parse_match(self, match: Dict[str, Any], original_concept: str, ontology: str) -> Optional[Dict[str, Any]]:
        """
        Private helper method to parse a single match from the API response.
        """
        try:
            annotated_class = match['annotatedClass']
            annotations = match['annotations']
            
            iri = annotated_class.get("@id")
            if not iri:
                return None

            links = annotated_class.get('links', {})
            ancestors_url = links.get('ancestors')
            
            ancestors = self._fetch_ancestors(ancestors_url) if ancestors_url else ['N/A']
            
            return {
                'Concept': original_concept,
                'mapped term': annotations[0].get('text', 'N/A'),
                'ontology': ontology,
                'concept id': iri.split('/')[-1],
                'IRI': iri,
                'ancestors': ancestors
            }
        except (KeyError, IndexError) as e:
            print(f"Error parsing match for concept '{original_concept}': Missing key {e}")
            return None

    def map_concepts(self, concepts: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Maps a list of concepts to their ontology terms.

        This is the main public method that orchestrates the mapping process. It
        saves the detailed results to a CSV and returns a summary dictionary.

        Args:
            concepts (List[str]): A list of concept strings to map.

        Returns:
            Dict[str, List[str]]: A dictionary where keys are the original concepts and
                                  values are a list of formatted mapping strings.
        """
        print("\n===========================================Doing Ontology Mapping===================================================")
        all_results = []

        new_path = os.path.join(os.getcwd(), 'savings', 'new_concepts_output.json')
        path = os.path.join(os.getcwd(), 'savings', 'concepts_output.json')
        if os.path.exists(new_path):
            with open(new_path) as f:
                data = json.load(f)
            concept_list = list(data.keys())
        
        elif os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            concept_list = list(data.keys())
            
        else:
            concept_list = concepts.split(', ')

        if not len(concept_list):
            return None
        
        else:
            lemmatizer = WordNetLemmatizer()
            processed_concepts = {}
            for concept in concept_list:
                words = concept.split()
                lemmatized_words = [lemmatizer.lemmatize(word, pos='n') for word in words]
                processed_concepts[concept] = ' '.join(lemmatized_words)
        

        for original_concept, lemmatized_concept in tqdm(processed_concepts.items()):
            print(f"Processing concept: '{original_concept}'")
            found_match_for_concept = False
            for ontology in self.ontologies:
                params = {
                    'apikey': self.api_key,
                    'ontologies': ontology,
                    'format': 'json',
                    'include': 'prefLabel',
                    'text': lemmatized_concept
                }
                
                try:
                    response = requests.get(self.BASE_URL, params=params)
                    response.raise_for_status() # Raise an exception for bad status codes
                    results_json = response.json()

                    if not results_json:
                        print(f"  - No results found for '{original_concept}' in {ontology}")
                        continue
               
                    for results in results_json:
                    # Process the first and most relevant match found
                        best_match_data = self._parse_match(results, original_concept, ontology)
                        if best_match_data:
                            all_results.append(best_match_data)
                            found_match_for_concept = True
                    break # Stop searching other ontologies if a match is found
                        

                except requests.exceptions.RequestException as e:
                    print(f"API request failed for '{concept}' in {ontology}: {e}")
            
            if not found_match_for_concept:
                 print(f"  - No matches found for '{concept}' in any specified ontology.")
                 
            


        if not all_results:
            print("No ontology mappings were found for any of the concepts.")
            return {concept: [] for concept in concepts}

        # Create and save DataFrame
        df = pd.DataFrame(all_results)
        os.makedirs(os.path.join(os.getcwd(), 'savings'), exist_ok=True)
        df.to_csv(os.path.join(os.getcwd(), 'savings', self.output_directory), index=False)
        print(f"\nONTOLOGY MAPPINGS SAVED TO: {self.output_directory}")

        # Format the final dictionary to be returned
        concepts = list(processed_concepts.keys())
        mapped_dict = {concept: [] for concept in concepts}
        for _, row in df.iterrows():
            concept_key = row['Concept']
    
            mapping_str = f"{str(row['mapped term']).upper()}-->{str(row['ancestors'])}"
            mapped_dict[concept_key].append(mapping_str)

        return mapped_dict
