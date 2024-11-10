from lmnr import Laminar as L
from dotenv import load_dotenv
import os


def laminar_main_text(content):
    """Processes the text using the main Laminar pipeline."""
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    laminar_api_key = ""

    L.initialize(project_api_key=laminar_api_key, instruments=set())
    
    result = L.run(
        pipeline='ASR_Main_text',
        inputs={'question': content['text']},
        env={'OPENAI_API_KEY': openai_api_key},
    )
    return result

def laminar_assistant(content):
    """Processes the text and instruction using the assistant Laminar pipeline."""
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    laminar_api_key = ""

    L.initialize(project_api_key=laminar_api_key, instruments=set())
    
    result = L.run(
        pipeline='ASR_Assistant',
        inputs={'text': content['text'], 'instruction': content['instruction']},
        env={'OPENAI_API_KEY': openai_api_key},
    )
    return result
