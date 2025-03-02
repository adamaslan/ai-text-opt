import marimo

__generated_with = "0.11.13"
app = marimo.App()


@app.cell
def _(prompt):
    # Phase 1: Setup and Configuration

    # Cell 1: Core Imports and Response Structure
    import os
    import time
    import json
    import requests
    import logging
    import pickle
    from dataclasses import dataclass
    from typing import List, Dict, Tuple, Optional
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    from collections import defaultdict

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    @dataclass
    class TherapeuticResponse:
        """Enhanced response structure for therapeutic context"""
        text: str
        timestamp: float
        error: bool = False
        processing_time: float = 0.0
        error_details: str = ""
        timeout: bool = False
        empathy_score: float = 0.0
        safety_checks: List[str] = None
        ethical_considerations: List[str] = None
        refinement_suggestions: List[str] = None
        crisis_flag: bool = False

    # Cell 2: Ollama Client Implementation
    class OllamaClient:
        """Robust Ollama client with configurable timeouts"""
        def __init__(self, model_name: str = "hf.co/TheDrummer/Gemmasutra-Mini-2B-v1-GGUF:Q3_K_L", base_url: str = "http://localhost:11434"):
            self.model_name = model_name
            self.base_url = base_url
            self.max_retries = 5
            self.request_timeout = 300
            self._verify_model()

        def _parse_json_safe(self, text: str):
            """Enhanced JSON parsing with fallback"""
            clean_text = text.strip()
            if not clean_text:
                return {"error": "Empty response"}
            
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                try:
                    start = clean_text.find('{')
                    end = clean_text.rfind('}') + 1
                    return json.loads(clean_text[start:end])
                except:
                    return {"error": f"Invalid JSON format: {clean_text[:200]}..."}
            except Exception as e:
                return {"error": str(e)}

        def _verify_model(self):
            """Model verification with status checks"""
            for attempt in range(self.max_retries):
                try:
                    resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
                    if resp.status_code == 200:
                        data = self._parse_json_safe(resp.text)
                        models = [m['name'] for m in data.get('models', [])]
                        if any(self.model_name in m for m in models):
                            return
                        self._pull_model()
                        return
                    logger.warning(f"Model check failed (status {resp.status_code})")
                except Exception as e:
                    logger.warning(f"Model check attempt {attempt+1} failed: {e}")
                    time.sleep(2 ** attempt)
            raise ConnectionError(f"Couldn't connect to Ollama after {self.max_retries} attempts")

        def _pull_model(self):
            """Model pulling with progress tracking"""
            try:
                resp = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name},
                    stream=True,
                    timeout=600
                )
                for line in resp.iter_lines():
                    if line:
                        try:
                            status = self._parse_json_safe(line).get('status', '')
                            logger.info(f"Pull progress: {status}")
                        except:
                            continue
            except Exception as e:
                logger.error(f"Model pull failed: {e}")
                raise

        def generate(self, prompt: str) -> Tuple[str, bool]:
            """Generation with configurable timeout and retries"""
            for attempt in range(self.max_retries):
                try:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            requests.post,
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": prompt[:4000],
                                "stream": False,
                                "options": {"temperature": 0.5}
                            },
                            timeout=self.request_timeout
                        )
                        resp = future.result(timeout=self.request_timeout)
                        data = self._parse_json_safe(resp.text)
                        return data.get("response", ""), False
                except FutureTimeoutError:
                    logger.warning(f"Generation timed out (attempt {attempt+1})")
                    return f"Error: Timeout after {self.request_timeout}s", True
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            return f"Error: Failed after {self.max_retries} attempts", True

    # Cell 3: Base Agent Framework
    class BaseAgent:
        """Timeout-aware base agent"""
        def __init__(self, client: OllamaClient):
            self.client = client
            self.retry_count = 3
            self.max_wait = 300
        
        def safe_generate(self, prompt: str) -> TherapeuticResponse:
            """Generation with time budget tracking"""
            start_time = time.time()
            timeout_occurred = False
        
            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                return TherapeuticResponse(
                    text="Error: Invalid input prompt",
                    timestamp=start_time,
                    error=True,
                    error_details="Empty or non-string prompt",
                    processing_time=0.0
                )
            
            for attempt in range(self.retry_count):
                try:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(self.client.generate, prompt)
                        text, error = future.result(timeout=self.max_wait)
                    
                        return TherapeuticResponse(
                            text=text,
                            timestamp=start_time,
                            error=error,
                            processing_time=time.time() - start_time,
                            error_details=text if error else "",
                            timeout=timeout_occurred
                        )
                except FutureTimeoutError:
                    logger.error(f"Generation timed out after {self.max_wait}s")
                    timeout_occurred = True
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Generation error: {e}")
                
            return TherapeuticResponse(
                text=f"Final error: {error_msg}" if 'error_msg' in locals() else "Unknown error",
                timestamp=start_time,
                error=True,
                error_details=error_msg if 'error_msg' in locals() else "",
                processing_time=time.time() - start_time,
                timeout=timeout_occurred
            )

    # Cell 4: Prompt Integration and Saving the Response to a Pickle File

    # Define the prompt


    # Initialize the Ollama client and the base agent
    client = OllamaClient()
    agent = BaseAgent(client)

    # Generate the therapeutic response using the prompt
    response = agent.safe_generate(prompt)
 
    # Save the response object to a pickle file
    with open("therapeutic_response.pkl", "wb") as f:
        pickle.dump(response, f)

    print("Response saved to therapeutic_response.pkl")
    # Load the pickle file
    with open("therapeutic_response.pkl", "rb") as file:
        therapeutic_response = pickle.load(file)

    # Print the contents of the pickle file
    print(therapeutic_response)
    return (
        BaseAgent,
        Dict,
        FutureTimeoutError,
        List,
        OllamaClient,
        Optional,
        TherapeuticResponse,
        ThreadPoolExecutor,
        Tuple,
        agent,
        client,
        dataclass,
        defaultdict,
        f,
        file,
        json,
        logger,
        logging,
        os,
        pickle,
        requests,
        response,
        therapeutic_response,
        time,
    )


@app.cell
def _(agent, pickle):
    with open('therapeutic_response.pkl', 'rb') as file_1:
        original_response = pickle.load(file_1)
    original_text = original_response.text
    new_prompt = f'\nThe following text is a therapeutic response:\n"{original_text}"\n\nPlease rewrite this response to be more intense and humiliating but just between becky and aaron. Dont involve anyone else.\nEnsure that the language is sharper, more degrading, and highly impactful while remaining grammatically correct.\nWrite 10 phrases that can be used to intensify and humiliate the original response.\n'
    intense_response = agent.safe_generate(new_prompt)
    print(intense_response)
    with open('intense_therapeutic_response.pkl', 'wb') as f_1:
        pickle.dump(intense_response, f_1)
    return (
        f_1,
        file_1,
        intense_response,
        new_prompt,
        original_response,
        original_text,
    )


@app.cell
def _(agent, pickle):
    with open('intense_therapeutic_response.pkl', 'rb') as file_2:
        original_response_1 = pickle.load(file_2)
    original_text_1 = original_response_1.text
    new_prompt_1 = f'\nThe following text is a therapeutic response:\n"{original_text_1}"\n\nwrite a list of 10 intense and humiliating phrases about aarons small dick for becky to say to aaron.\n'
    intense_response_1 = agent.safe_generate(new_prompt_1)
    print(intense_response_1)
    with open('intense_therapeutic_response2.pkl', 'wb') as f_2:
        pickle.dump(intense_response_1, f_2)
    return (
        f_2,
        file_2,
        intense_response_1,
        new_prompt_1,
        original_response_1,
        original_text_1,
    )


@app.cell
def _(agent, pickle):
    with open('intense_therapeutic_response2.pkl', 'rb') as file_3:
        original_response_2 = pickle.load(file_3)
    original_text_2 = original_response_2.text
    new_prompt_2 = f'\nThe following text is a therapeutic response:\n"{original_text_2}"\n\nPlease rewrite 10 intense and humiliating phrases about aarons small dick.\nEnsure that the language is sharper, more degrading, and highly impactful while remaining grammatically correct.\nEnsure that the phrases involve a private bedroom setting and past partners.\n\n'
    intense_response_2 = agent.safe_generate(new_prompt_2)
    print(intense_response_2)
    with open('intense_therapeutic_response3.pkl', 'wb') as f_3:
        pickle.dump(intense_response_2, f_3)
    return (
        f_3,
        file_3,
        intense_response_2,
        new_prompt_2,
        original_response_2,
        original_text_2,
    )


@app.cell
def _(BaseAgent, Dict_1, List_1, OllamaClient, json, os, pickle):
    from types import SimpleNamespace
    from typing import Dict, List

    def safe_pickle_load(filename, default_factory):
        if not os.path.exists(filename):
            print(f'{filename} not found. Generating mock data...')
            data = default_factory()
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        with open(filename, 'rb') as f:
            return pickle.load(f)

    class IntimacyContextAnalyzer(BaseAgent):
        """Analyzes intimacy needs and communication patterns"""

        def analyze_desires(self, input_text: str) -> Dict_1:
            prompt = f'Analyze intimacy context (sex-positive focus):\n        User Statement: "{input_text[:2000]}"\n        \n        Identify:\n        - Expressed/unexpressed desires\n        - Communication style about intimacy\n        - Emotional blocks/opportunities\n        - Potential exploration pathways\n        - Consent awareness indicators\n        \n        Output JSON with:\n        - communication_style: str\n        - expressed_desires: List[str]\n        - potential_explorations: List[str]\n        - communication_improvements: List[str]\n        - affirmation_opportunities: List[str]'
            response = self.safe_generate(prompt)
            return self.client._parse_json_safe(response.text)

    def analyze_desires_from_pickle():
        response = safe_pickle_load('therapeutic_response.pkl', lambda: SimpleNamespace(text='Client desires deeper connection but struggles with verbal communication. \n        Shows interest in exploring sensory experiences but hesitant to initiate. Strong emphasis on mutual respect.'))
        client = OllamaClient()
        analyzer = IntimacyContextAnalyzer(client)
        analysis = analyzer.analyze_desires(response.text)
        with open('desire_analysis.pkl', 'wb') as f:
            pickle.dump(analysis, f)
        print('Desire analysis saved to desire_analysis.pkl')

    class IntimacyActionGenerator(BaseAgent):
        """Generates personalized intimacy enhancement actions"""

        def generate_actions(self, analysis: Dict_1) -> List_1[Dict_1]:
            prompt = f'Create sex-positive action plan:\n        Context: {json.dumps(analysis)[:3000]}\n        \n        Pay close attention to both verbal and non-verbal cues related to aaron and his small dick.\n\n        Suggest 5-7 actions including:\n        - Communication exercises\n        - Phrases the partner should say\n        - Sensory exploration ideas\n        - Consent practice scenarios\n        - Connection-building activities\n        \n        Format as JSON list with:\n        - action_type: str\n        - description: str\n        - purpose: str\n        - difficulty: str (beginner/intermediate/advanced)'
            response = self.safe_generate(prompt)
            return self.client._parse_json_safe(response.text)

    def generate_action_plan():
        analysis = safe_pickle_load('desire_analysis.pkl', lambda: {'communication_style': 'non-verbal', 'expressed_desires': ['quality time', 'physical touch']})
        client = OllamaClient()
        generator = IntimacyActionGenerator(client)
        action_plan = generator.generate_actions(analysis)
        with open('action_plan.pkl', 'wb') as f:
            pickle.dump(action_plan, f)
        print('Action plan saved to action_plan.pkl')

    class IntimacyCustomizer(BaseAgent):
        """Tailors suggestions to individual preferences"""

        def customize_actions(self, actions: List_1[Dict_1], analysis: Dict_1) -> Dict_1:
            prompt = f'Refine intimacy plan:\n        Initial Plan: {json.dumps(actions)[:3000]}\n        User Context: {json.dumps(analysis)[:2000]}\n        \n        Format response as JSON with:\n        {{\n            "plan_summary": "brief description",\n            "actions": [\n                {{\n                    "action_type": "string",\n                    "description": "string",\n                    "preparation_steps": ["list"],\n                    "ideal_timing": "string",\n                    "consent_checkpoints": ["list"]\n                }}\n            ]\n        }}'
            response = self.safe_generate(prompt)
            return self.client._parse_json_safe(response.text)

    def refine_action_plan():
        actions = safe_pickle_load('action_plan.pkl', lambda: [{'action_type': 'communication', 'description': 'Daily check-ins'}])
        analysis = safe_pickle_load('desire_analysis.pkl', lambda: {})
        client = OllamaClient()
        customizer = IntimacyCustomizer(client)
        refined_plan = customizer.customize_actions(actions, analysis)
        with open('refined_plan.pkl', 'wb') as f:
            pickle.dump(refined_plan, f)
        print('Refined plan saved to refined_plan.pkl')

    def run_pipeline():
        try:
            analyze_desires_from_pickle()
            generate_action_plan()
            refine_action_plan()
            print('Processing pipeline completed successfully!')
            print('\nSample Outputs:')
            print('Desire Analysis:', safe_pickle_load('desire_analysis.pkl', lambda: {}))
            print('Action Plan:', safe_pickle_load('action_plan.pkl', lambda: []))
            print('Refined Plan:', safe_pickle_load('refined_plan.pkl', lambda: {}))
        except Exception as e:
            print(f'Pipeline error: {str(e)}')
            print('Recommendation: Check Ollama server status and model availability')
    run_pipeline()
    return (
        Dict,
        IntimacyActionGenerator,
        IntimacyContextAnalyzer,
        IntimacyCustomizer,
        List,
        SimpleNamespace,
        analyze_desires_from_pickle,
        generate_action_plan,
        refine_action_plan,
        run_pipeline,
        safe_pickle_load,
    )


@app.cell
def _(BaseAgent, Dict_1, OllamaClient, json, pickle):
    class IntensitySpecialist(BaseAgent):

        def boost_elements(self, refined_plan: Dict_1) -> Dict_1:
            prompt = f'Create ULTRA-INTENSE variants (STRICT JSON):\n        Current Plan: {json.dumps(refined_plan, indent=2)[:3000]}\n        \n        You MUST select and enhance EXACTLY:\n        - 5 most promising actions (NO MORE, NO LESS)\n        - 5 most impactful phrases (NO MORE, NO LESS)\n        \n        CRITICAL: Failure to include exactly 5 elements in each category will cause system failure!\n        \n        For each selected element:\n        - Triple the intensity parameters\n        - Add 3 escalation layers\n        - Include sensory domination techniques\n        - Specify power dynamics\n        \n        Required JSON Structure:\n        {{\n            "hyper_actions": [EXACTLY 5 ELEMENTS],\n            "hyper_phrases": [EXACTLY 5 ELEMENTS]\n        }}'
            response = self.safe_generate(prompt)
            return self._validate_boost(response.text)

        def _validate_boost(self, raw_text: str) -> Dict_1:
            parsed = self.client._parse_json_safe(raw_text)
            if not isinstance(parsed.get('hyper_actions', []), list):
                raise ValueError('hyper_actions must be an array')
            if not isinstance(parsed.get('hyper_phrases', []), list):
                raise ValueError('hyper_phrases must be an array')
            action_count = len(parsed['hyper_actions'])
            phrase_count = len(parsed['hyper_phrases'])
            if action_count != 5 or phrase_count != 5:
                error_msg = ['Element count mismatch:', f'- Hyper Actions: {action_count}/5', f'- Hyper Phrases: {phrase_count}/5', 'Full response preview:', json.dumps(parsed, indent=2)[:500]]
                raise ValueError('\n'.join(error_msg))
            return parsed

    def create_hyper_intense_variants():
        try:
            with open('refined_plan.pkl', 'rb') as f:
                refined_plan = pickle.load(f)
        except Exception as e:
            print(f'Failed loading refined plan: {str(e)}')
            return None
        client = OllamaClient()
        specialist = IntensitySpecialist(client)
        try:
            boosted = specialist.boost_elements(refined_plan)
            with open('hyper_intense.pkl', 'wb') as f:
                pickle.dump(boosted, f)
            print('\n=== Hyper-Intense Variants ===')
            print(json.dumps(boosted, indent=2))
            return boosted
        except Exception as e:
            print(f'Intensity Specialization Failed: {str(e)}')
            return None
    hyper_intense_data = create_hyper_intense_variants()
    return (
        IntensitySpecialist,
        create_hyper_intense_variants,
        hyper_intense_data,
    )


app._unparsable_cell(
    r"""
    # cell 8 gemeni 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    @dataclass
    class TherapeuticResponse:
        text: str
        timestamp: float
        error: bool = False
        processing_time: float = 0.0
        error_details: str = \"\"
        timeout: bool = False
        empathy_score: float = 0.0
        safety_checks: List[str] = None
        ethical_considerations: List[str] = None
        refinement_suggestions: List[str] = None
        crisis_flag: bool = False

    class OllamaClient:
        def __init__(self, model_name: str = \"hf.co/TheDrummer/Gemmasutra-Mini-2B-v1-GGUF:Q3_K_L\", base_url: str = \"http://localhost:11434\"):
            self.model_name = model_name
            self.base_url = base_url
            self.max_retries = 5
            self.request_timeout = 300
            self._verify_model()

        def _parse_json_safe(self, text: str):
            clean_text = text.strip()
            if not clean_text:
                return {\"error\": \"Empty response\"}
            try:
                return json.loads(clean_text)
            except json.JSONDecodeError:
                try:
                    start = clean_text.find('{')
                    end = clean_text.rfind('}') + 1
                    return json.loads(clean_text[start:end])
                except:
                    return {\"error\": f\"Invalid JSON format: {clean_text[:200]}...\"}
            except Exception as e:
                return {\"error\": str(e)}

        def _verify_model(self):
            for attempt in range(self.max_retries):
                try:
                    resp = requests.get(f\"{self.base_url}/api/tags\", timeout=10)
                    if resp.status_code == 200:
                        data = self._parse_json_safe(resp.text)
                        models = [m['name'] for m in data.get('models', [])]
                        if any(self.model_name in m for m in models):
                            return
                        self._pull_model()
                        return
                    logger.warning(f\"Model check failed (status {resp.status_code})\")
                except Exception as e:
                    logger.warning(f\"Model check attempt {attempt+1} failed: {e}\")
                    time.sleep(2 ** attempt)
            raise ConnectionError(f\"Couldn't connect to Ollama after {self.max_retries} attempts\")

        def _pull_model(self):
            try:
                resp = requests.post(
                    f\"{self.base_url}/api/pull\",
                    json={\"name\": self.model_name},
                    stream=True,
                    timeout=600
                )
                for line in resp.iter_lines():
                    if line:
                        try:
                            status = self._parse_json_safe(line).get('status', '')
                            logger.info(f\"Pull progress: {status}\")
                        except:
                            continue
            except Exception as e:
                logger.error(f\"Model pull failed: {e}\")
                raise

        def generate(self, prompt: str) -> Tuple[str, bool]:
            for attempt in range(self.max_retries):
                try:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            requests.post,
                            f\"{self.base_url}/api/generate\",
                            json={
                                \"model\": self.model_name,
                                \"prompt\": prompt[:4000],
                                \"stream\": False,
                                \"options\": {\"temperature\": 0.5}
                            },
                            timeout=self.request_timeout
                        )
                        resp = future.result(timeout=self.request_timeout)
                        data = self._parse_json_safe(resp.text)
                        return data.get(\"response\", \"\"), False
                except FutureTimeoutError:
                    logger.warning(f\"Generation timed out (attempt {attempt+1})\")
                    return f\"Error: Timeout after {self.request_timeout}s\", True
                except Exception as e:
                    logger.warning(f\"Attempt {attempt+1} failed: {e}\")
                    time.sleep(1)
            return f\"Error: Failed after {self.max_retries} attempts\", True

    class BaseAgent:
        def __init__(self, client: OllamaClient):
            self.client = client
            self.retry_count = 3
            self.max_wait = 300

        def safe_generate(self, prompt: str) -> TherapeuticResponse:
            start_time = time.time()
            timeout_occurred = False

            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                return TherapeuticResponse(
                    text=\"Error: Invalid input prompt\",
                    timestamp=start_time,
                    error=True,
                    error_details=\"Empty or non-string prompt\",
                    processing_time=0.0
                )

            for attempt in range(self.retry_count):
                try:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(self.client.generate, prompt)
                        text, error = future.result(timeout=self.max_wait)

                        return TherapeuticResponse(
                            text=text,
                            timestamp=start_time,
                            error=error,
                            processing_time=time.time() - start_time,
                            error_details=text if error else \"\",
                            timeout=timeout_occurred
                        )
                except FutureTimeoutError:
                    logger.error(f\"Generation timed out after {self.max_wait}s\")
                    timeout_occurred = True
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f\"Generation error: {e}\")

            return TherapeuticResponse(
                text=f\"Final error: {error_msg}\" if 'error_msg' in locals() else \"Unknown error\",
                timestamp=start_time,
                error=True,
                error_details=error_msg if 'error_msg' in locals() else \"\",
                processing_time=time.time() - start_time,
                timeout=timeout_occurred
            )

    class IntensitySpecialist(BaseAgent):
        def boost_elements(self, refined_plan: Dict) -> Dict:
            prompt = f\"\"\"Create ULTRA-INTENSE variants (STRICT JSON):
            Current Plan: {json.dumps(refined_plan, indent=2)[:3000]}

            Select and enhance:
            - 5 most promising actions
            - 5 most impactful phrases

            For each selected element:
            - Triple the intensity parameters
            - Add 3 escalation layers
            - Include sensory domination techniques (e.g., visual: intense eye contact, auditory: commanding tone, tactile: firm touch)
            - Specify power dynamics (e.g., dominant, submissive, equal)

            Required JSON Structure (Example):
            {{
                \"hyper_actions\": [
                    {{
                        \"original_id\": \"str|int\",
                        \"ultra_variant\": {{
                            \"description\": \"str\",
                            \"intensity_score\": 6-10,
                            \"sensory_overload\": [\"str\"],
                            \"dominance_factors\": [\"str\"]
                        }}
                    }}
                ],
                \"hyper_phrases\": [
                    {{
                        \"original_id\": \"str\",
                        \"amplified_text\": \"str\",
                        \"linguistic_power\": 6-10,
                        \"delivery_modes\": [\"str\"]
                    }}
                ]
            }}\"\"\"

            for attempt in range(self.retry_count):
                try:
                    response = self.safe_generate(prompt)
                    boosted = self._validate_boost(response.text)
                    return boosted
                except (ValueError, json.JSONDecodeError) as e:
                    logger.error(f\"Boost JSON Error (attempt {attempt+1}): {e}\")
                    if attempt < self.retry_count - 1:
                        time.sleep(2**attempt)
                    else:
                        raise
                except Exception as e:
                    logger.error(f\"Ollama API call failed (attempt {attempt+1}): {e}\")
                    if attempt
    """,
    name="_"
)


@app.cell
def _(pickle):
    # Cell 9: Robust Results Display
    def display_results():
        try:
            with open("refined_plan.pkl", "rb") as f:
                plan_data = pickle.load(f)
        
            print("\nPersonalized Intimacy Plan:")
        
            # Ensure we're working with a dictionary
            if isinstance(plan_data, dict):
                actions = plan_data.get('actions', [])
            elif isinstance(plan_data, list):
                actions = plan_data
            else:
                actions = []
        
            # Display first 3 actions safely
            for idx, action in enumerate(actions[:3], 1):
                print(f"\nAction {idx}:")
                print(f"Type: {action.get('action_type', 'Connection Activity')}")
                print(f"Purpose: {action.get('purpose', 'Enhancing intimacy through mutual exploration')}")
                print(f"Steps: {', '.join(action.get('preparation_steps', ['Create comfortable environment']))}")
            
        except Exception as e:
            print(f"Error displaying results: {str(e)}")

    if __name__ == "__main__":
        display_results()
    return (display_results,)


if __name__ == "__main__":
    app.run()
