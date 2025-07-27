# MedPAO: A Protocol-Driven Agent for Structuring Medical Reports

## Overview

MedPAO is an intelligent agent system designed to process and structure medical radiology reports according to the standardized P.A.B.C.D.E.F protocol for chest x-ray interpretation. The system extracts medical concepts from free-text radiology reports, maps them to standard medical ontologies, and organizes them into a structured clinical format.

## Features

- Extracts key medical concepts from free-text reports
- Maps concepts to SNOMED CT and RADLEX medical ontologies
- Categorizes findings according to the P.A.B.C.D.E.F protocol
- Generates structured medical reports with standardized sections


## System Requirements

- Python 3.8 or higher
- Local LLM servers:
  - One server running on port 8080 for reasoning tasks
  - One server running on port 8081 for concept extraction
- Internet access (for BioPortal API calls)

## Installation

1. Cloning the repository and installing dependencies.
```bash
# Clone the repository
git clone <repository-url>
cd medpao-agent

# Install dependencies
pip install mcp-python-sdk pandas requests nltk tqdm huggingface_hub regex
```

2. Register and Get the Bioportal API access(https://bioportal.bioontology.org/). Enter the API key in `apis.json` file as:
```bash
{
    "API_KEY": "your api key"

}
```

3. To ensure local LLMs inference, refer to Huggingface TGI framework according to the available machines (https://huggingface.co/docs/text-generation-inference/en/index)

4. We use the following two models: 
- LLM engine : https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- concept extractor : https://huggingface.co/shrishSVaidya/medllama_finetuned

## Usage

1. Run the LLM engine on port 8080 of the machine via TGI.
```bash
# For example, on one node of Gaudi2 accelerator we run:
volume=/mnt/hf_model    #volume where the model weights are stored
model=deepseek-ai/DeepSeek-R1-Distill-Llama-70B     #model-id
docker run --name deepseek_runner -p 8080:80    --runtime=habana    --cap-add=sys_nice    --ipc=host    -v $volume:/data    -e MAX_TOTAL_TOKENS=4056    -e BATCH_BUCKET_SIZE=256    -e PREFILL_BATCH_BUCKET_SIZE=4    -e RUST_LOG=debug    -e PAD_SEQUENCE_TO_MULTIPLE_OF=64    ghcr.io/huggingface/text-generation-inference:3.2.3-gaudi    --model-id $model    --sharded true --num-shard 6    --max-input-tokens 6048 --max-total-tokens 8056    --max-batch-prefill-tokens 6096 --max-batch-size 2    --max-waiting-tokens 7 --waiting-served-ratio 1.2        #Command to start TGI server
```

2. Run the concept extractor model on port 8081 of the machine via TGI.

```bash
# For example, on one node of Gaudi2 accelerator we run:
volume=/mnt/hf_model    #volume where the model weights are stored
model=model_id      #finetuned concept extraction model
hf_token=...        #Huggingface access token for authorization
docker run --name medllama_runner -p 8081:80    --runtime=habana    --cap-add=sys_nice    --ipc=host    -v $volume:/data    -e MAX_TOTAL_TOKENS=3056    -e BATCH_BUCKET_SIZE=256    -e PREFILL_BATCH_BUCKET_SIZE=4 -e HF_TOKEN=$hf_token   -e RUST_LOG=debug    -e PAD_SEQUENCE_TO_MULTIPLE_OF=64    ghcr.io/huggingface/text-generation-inference:3.2.3-gaudi    --model-id $model    --sharded false    --max-input-tokens 1048 --max-total-tokens 3056    --max-batch-prefill-tokens 2096 --max-batch-size 8    --max-waiting-tokens 7 --waiting-served-ratio 1.2    #Command to start TGI server
```

3. Export the Bioportal API key
```bash
export API_KEY="your-api-key"
```
4. Run the MedPAO agent with prompt:
- If you want to utilize the local caching(for faster inference) then specify it in prompt:
```bash
python mcp_client.py --prompt "the task is to structure the given medical report according to ABCDEF protocol, utilizing the check_cache tool: {findings}"
```

- Similarly, if you dont want to use local caching:
```bash
python mcp_client.py --prompt "the task is to structure the given medical report according to ABCDEF protocol, without using check_cache tool: {findings}"
```

5. The system will process the medical reports defined in the `{findings}` placeholder of the above prompt.


## Project Structure

```
├── cached_vocab.json         # Cache of previously processed concepts
├── apis.json                 # json containing api key
├── mcp_client.py             # PAO agent implementation
├── mcp_server.py             # MCP server with medical processing tools
├── tools/                    # Processing tools
│   ├── categorize_concepts.py
│   ├── check_cache.py
│   ├── filter_ontologies.py
│   ├── generate_report.py
│   ├── get_concept.py
│   └── ontology_mapping.py
├── savings/                  # Output directory
   
```
- The `cached_vocab.json` will have all unique concepts and their corresponding category according to the ABCDEF protocol and it will keep updating as and when the agent is used on multiple samples.

- The `apis.json` file is created to store the Bioportal API, so that the relevant tool would use it.

- The `mcp_client.py` and `mcp_server.py` contain the MedPAO agent implementation as described in the paper.

- The `tools` directory contains implementation of all the tools as described in the paper.

- The `savings` directory contains temporary files, containing the output of each tool execution for every sample.


## Outputs

The system generates several output files for each sample case in the `savings/` directory:

- `concepts_output.json`: Contains the output of `get_concept` tool, i.e., Extracted concepts and their source sentences from the given report
- `new_concepts_output.json`: Contains output of `check_cache` tool, i.e., Concepts not found in local cache ( `cached_vocab.json`)
- `ontologyMapping.csv` : Contains output of `ontology_mapping` tool, i.e.,Concepts alongwith their mapped ontologies and the heirarchy.
- `segregated.json` : Contains the output of `filter_ontologies` tool, i.e., PRIMARY and SECONDARY classification of mapped ontologies for each concept.
- `categorized_concepts.json`: Contains output of `categorize_concepts` tool, i.e., Categorized concepts according to specified protocol.
- `categorized_report.json`: Contains the output of `generate_report` tool, i.e., Final structured report.


