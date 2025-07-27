import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import json
from huggingface_hub import InferenceClient
import regex as re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import shutil
from tqdm import tqdm
import pandas as pd
import argparse

class PlanningMode(Enum):
    STEP_BY_STEP = "step_by_step"  # Plan each action individually
    FULL_PIPELINE = "full_pipeline"  # Plan entire pipeline upfront

class PAOPhase(Enum):
    PLAN = "plan"
    ACTION = "action"
    OBSERVE = "observe"

@dataclass
class ActionStep:
    tool_name: str
    arguments: Dict[str, Any]
    description: str
    step_number: int

@dataclass
class PAOState:
    user_query: str
    planning_mode: PlanningMode
    current_phase: PAOPhase
    
    # Step-by-step mode fields
    current_plan: str = ""
    
    # Full pipeline mode fields
    full_pipeline: List[ActionStep] = field(default_factory=list)
    pipeline_generated: bool = False
    current_step_index: int = 0
    
    # Common fields
    action_taken: Dict[str, Any] = None
    observation: str = ""
    goal_achieved: bool = False
    iteration_count: int = 0
    max_iterations: int = 10
    context_history: List[str] = field(default_factory=list)
    action_results: List[str] = field(default_factory=list)

server_params = StdioServerParameters(command="python", args=["/home/user/agent/mcp_codes/mcp_server.py"])

def llm_client(message: str) -> str:
    """Send a message to the LLM and return the response."""
    client = InferenceClient(base_url="http://localhost:8080/v1/")
    response = client.chat.completions.create(
        model="tgi",
        messages=[
            {"role": "system", "content": "You are an intelligent medical assistant. Follow instructions precisely."},
            {"role": "user", "content": message}
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()



def get_full_pipeline_planning_prompt(state: PAOState, tools) -> str:
    """Generate prompt for full pipeline planning mode."""
    tools_description = "\n".join([f"- {tool.name}: {tool.description}, Input Schema: {tool.inputSchema}" for tool in tools])
    
    return f"""You are in FULL PIPELINE PLANNING mode of a PAO agent.

                USER'S ORIGINAL QUERY: {state.user_query}

                AVAILABLE TOOLS:
                {tools_description}

                TASK: Create a complete pipeline of actions to fully satisfy the user's query. 

                You must respond with a JSON array of action steps in the following format:

                [
                    {{
                        "step_number": 1,
                        "tool_name": "tool-name",
                        "arguments": {{
                            "argument-name": "value"
                        }},
                        "description": "What this step accomplishes"
                    }},
                    {{
                        "step_number": 2,
                        "tool_name": "another-tool",
                        "arguments": {{
                            "argument-name": "value"
                        }},
                        "description": "What this step accomplishes"
                    }}
                ]

                Requirements:
                1. Plan ALL steps needed to complete the user's request
                2. Each step should logically build on the previous ones
                3. Use appropriate tools with correct argument schemas
                4. Provide clear descriptions for each step
                5. Ensure the pipeline will fully satisfy the user's query

                IMPORTANT: Respond ONLY with the JSON array, no additional text.
            """



def get_observation_prompt(state: PAOState, tool_result: str, is_final_step: bool = False) -> str:
    """Generate prompt for observation phase."""
    mode_context = ""
    if state.planning_mode == PlanningMode.FULL_PIPELINE:
        if is_final_step:
            mode_context = f"\nThis was the FINAL step ({state.current_step_index + 1}/{len(state.full_pipeline)}) of your planned pipeline."
        else:
            mode_context = f"\nThis was step {state.current_step_index + 1}/{len(state.full_pipeline)} of your planned pipeline."
    
    return f"""You are in the OBSERVATION phase of a PAO agent.

                USER'S ORIGINAL QUERY: {state.user_query}

                PLANNING MODE: {state.planning_mode.value.upper().replace('_', ' ')}

                ACTION TAKEN: {state.action_taken}

                TOOL RESULT: {tool_result}
                {mode_context}

                TASK: Analyze the tool result and determine:
                1. Was the action successful?
                2. Does the current result (considering all previous results) fully satisfy the user's query?
                3. What should be done next?

                ALL PREVIOUS RESULTS: {state.action_results if state.action_results else 'None'}

                Respond in the following format:
                ANALYSIS: [Your analysis of the tool result]
                GOAL_ACHIEVED: [YES/NO - whether the user's query is fully satisfied with all results so far]
                NEXT_STEPS: [What should be done next, or "COMPLETE" if goal is achieved. If an error occured ask the llm to run the previous tool with modified prompt]
            """

def parse_observation(observation_text: str) -> tuple[str, bool, str]:
    """Parse the observation response to extract components."""
    lines = observation_text.strip().split('\n')
    analysis = ""
    goal_achieved = False
    next_steps = ""
    
    for line in lines:
        if line.startswith("ANALYSIS:"):
            analysis = line.replace("ANALYSIS:", "").strip()
        elif line.startswith("GOAL_ACHIEVED:"):
            goal_str = line.replace("GOAL_ACHIEVED:", "").strip().upper()
            goal_achieved = goal_str == "YES"
        elif line.startswith("NEXT_STEPS:"):
            next_steps = line.replace("NEXT_STEPS:", "").strip()
    
    return analysis, goal_achieved, next_steps

def extract_json_from_response(response: str) -> Optional[Union[Dict, List]]:
    """Extract JSON object or array from LLM response."""
    # Try to find JSON array first
    array_pattern = r'(\[(?:[^\[\]]|(?R))*\])'
    array_match = re.search(array_pattern, response, re.DOTALL)
    
    if array_match:
        json_str = array_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Fallback to JSON object
    object_pattern = r'(\{(?:[^{}]|(?R))*\})'
    object_match = re.search(object_pattern, response, re.DOTALL)
    
    if object_match:
        json_str = object_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return None
    else:
        print("❌ No JSON found in the response.")
        return None

def parse_pipeline_response(response: str) -> List[ActionStep]:
    """Parse the full pipeline planning response."""
    pipeline_data = extract_json_from_response(response)
    
    if not isinstance(pipeline_data, list):
        print("❌ Expected JSON array for pipeline")
        return []
    
    action_steps = []
    for step_data in pipeline_data:
        try:
            action_step = ActionStep(
                tool_name=step_data["tool_name"],
                arguments=step_data["arguments"],
                description=step_data["description"],
                step_number=step_data["step_number"]
            )
            action_steps.append(action_step)
        except KeyError as e:
            print(f"❌ Missing required field in pipeline step: {e}")
            continue
    
    return action_steps


async def execute_full_pipeline_cycle(state: PAOState, session: ClientSession) -> PAOState:
    """Execute PAO cycle in full pipeline mode."""
    tools = await session.list_tools()
    
    # PLAN (only once)
    if not state.pipeline_generated:
        print(f"\n🧠 FULL PIPELINE PLANNING")
        planning_prompt = get_full_pipeline_planning_prompt(state, tools.tools)
        plan_response = llm_client(planning_prompt)
        
        pipeline_steps = parse_pipeline_response(plan_response)
        if not pipeline_steps:
            print("❌ Failed to generate valid pipeline")
            state.observation = "Failed to generate valid pipeline"
            return state
        
        state.full_pipeline = pipeline_steps
        state.pipeline_generated = True
        
        print(f"Generated pipeline with {len(state.full_pipeline)} steps:")
        for step in state.full_pipeline:
            print(f"  {step.step_number}. {step.description} (using {step.tool_name})")
    
    # ACTION (execute current step)
    if state.current_step_index < len(state.full_pipeline):
        current_step = state.full_pipeline[state.current_step_index]
        print(f"\n⚡ EXECUTING STEP {current_step.step_number}: {current_step.description}")
        
        tool_call = {
            "tool": current_step.tool_name,
            "arguments": current_step.arguments
        }
        
        state.action_taken = tool_call
        print(f"Tool Call: {tool_call}")
        
        try:
            get_concept_done = False
            if tool_call["tool"]=='get_concept':
                args = tool_call["arguments"]
                get_concept_done = True
            elif tool_call["tool"]=='ontology_mapping' and not get_concept_done:
                args = tool_call["arguments"]
            else:
                args = {list(tool_call["arguments"].keys())[0]:"start"}

            result = await session.call_tool(tool_call["tool"], arguments=args)
            tool_result = result.content[0].text
            state.action_results.append(tool_result)
            print(f"Result: {tool_result}")
        except Exception as e:
            tool_result = f"Error: {str(e)}"
            state.action_results.append(tool_result)
            print(f"❌ Error: {e}")
        
        # OBSERVE (after each step or at the end)
        is_final_step = (state.current_step_index + 1) >= len(state.full_pipeline)
        if is_final_step:
            print(f"\n👁️ OBSERVATION (Step {current_step.step_number})")
            observation_prompt = get_observation_prompt(state, tool_result, is_final_step)
            observation_response = llm_client(observation_prompt)
            
            analysis, goal_achieved, next_steps = parse_observation(observation_response)
            state.observation = analysis
            state.goal_achieved = goal_achieved
            
            print(f"Analysis: {analysis}")
            print(f"Goal Achieved: {'✅ YES' if goal_achieved else '❌ NO'}")
        else:
            print(f"\n👁️ OBSERVATION (Step {current_step.step_number}) - Skipping for now, will observe after full pipeline")
            analysis = "Observation skipped for intermediate step"
            goal_achieved = False
        
        # Move to next step if goal not achieved and more steps remain
        if not goal_achieved and not is_final_step:
            state.current_step_index += 1
        
        # If goal not achieved after pipeline completion, agent can plan additional steps
        if not goal_achieved and is_final_step:
            print(f"\n🔄 Pipeline completed but goal not achieved. May need additional actions.")
    
    return state

async def run_pao_agent(user_query: str, planning_mode: PlanningMode = PlanningMode.FULL_PIPELINE, max_iterations: int = 10):
    """Run the PAO agent with specified planning mode."""
    print(f"🚀 Starting PAO Agent")
    print(f"Query: {user_query}")
    print(f"Planning Mode: {planning_mode.value.upper().replace('_', ' ')}")
    print(f"Max Iterations: {max_iterations}")
    
    state = PAOState(
        user_query=user_query,
        planning_mode=planning_mode,
        current_phase=PAOPhase.PLAN,
        max_iterations=max_iterations
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            while not state.goal_achieved and state.iteration_count < state.max_iterations:
                print(f"\n{'='*60}")
                print(f"ITERATION {state.iteration_count + 1} - {planning_mode.value.upper().replace('_', ' ')} MODE")
                print(f"{'='*60}")
              
                state = await execute_full_pipeline_cycle(state, session)
                
                state.iteration_count += 1
                
                if state.goal_achieved:
                    print(f"\n🎉 SUCCESS! Goal achieved in {state.iteration_count} iterations.")
                    break
                elif state.iteration_count >= state.max_iterations:
                    print(f"\n⏰ Maximum iterations reached.")
                    break
                elif planning_mode == PlanningMode.FULL_PIPELINE and state.current_step_index >= len(state.full_pipeline):
                    # Pipeline completed, check if we need to continue with step-by-step
                    if not state.goal_achieved:
                        print(f"\n🔄 Pipeline completed but goal not achieved.")
                      
                
            # Final summary
            print(f"\n{'='*60}")
            print(f"FINAL SUMMARY")
            print(f"{'='*60}")
            print(f"User Query: {state.user_query}")
            print(f"Planning Mode: {state.planning_mode.value.upper().replace('_', ' ')}")
            print(f"Total Iterations: {state.iteration_count}")
            print(f"Goal Achieved: {'✅ YES' if state.goal_achieved else '❌ NO'}")
            print(f"Final Observation: {state.observation}")
            print(f"\nAll Results:")
            for i, result in enumerate(state.action_results, 1):
                print(f"  {i}. {result}")
            
            return state

if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Run PAO agent with a prompt')
    parser.add_argument('--prompt', type=str, help='The prompt to run the PAO agent with')
    args = parser.parse_args()
    if args.prompt:
        prompt = args.prompt
    else:
        finding = "PA and lateral views of the chest are submitted. Lungs appear well inflated without evidence of focal airspace consolidation, pleural effusions, pulmonary edema, or pneumothorax. Cardiac and mediastinal contours are within normal limits. No acute bony abnormality is appreciated."
        prompt = f"the task is to structure the given medical report according to ABCDEF protocol: '{report}'"
  
    
    if os.path.exists(os.path.join(os.getcwd(), 'savings')):
        shutil.rmtree(os.path.join(os.getcwd(), 'savings'))
   
    
  
    print("\n" + "="*80)
    print("RUNNING IN FULL PIPELINE MODE")
    print("="*80)
    result2 = asyncio.run(run_pao_agent(prompt, PlanningMode.FULL_PIPELINE, max_iterations=6))
        
        
  

