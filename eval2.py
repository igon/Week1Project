from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
from openai import OpenAI
import json

from dotenv import load_dotenv
load_dotenv()

from langsmith.wrappers import wrap_openai
from langsmith import traceable

client = wrap_openai(OpenAI())

@traceable
def prompt_compliance_evaluator(run: Run, example: Example) -> dict:
    inputs = example.inputs['input']
    outputs = example.outputs['output']

    #Extract system prompt
    system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")

    
    #Extract message history 
    message_history = []
    for msg in inputs:
        if msg['type'] in ['human', 'ai']:
            message_history.append({
                'role': "user" if msg['type'] == 'human' else 'assistant',
                'content': msg['data']['content']
            })

    #Extract latest user message and model output
    latest_message = message_history[-1]['content'] if message_history else ""
    model_output = outputs['data']['content']

    evaluation_prompt = f"""
    System Prompt: {system_prompt}

    Message History:
    {json.dumps(message_history, indent=2)}

    Latest User Message: {latest_message}

    Model Output: {model_output}

    Based on the above information, evaluate the model's output for compliance with the system prompt and context of the conversation. 
    Provide a score from 0 to 10, where 0 is completely non-compliant and 10 is perfectly compliant.
    Also provide a brief explanation for your score.
        
    Respond in the following JSON format:
    {{
        "score": <int>,
        "explanation": "<string>"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the compliance of model outputs to given prompts and conversation context."},
            {"role": "user", "content": evaluation_prompt}
        ],
    temperature=0.2
)

    try: 
        result = json.loads(response.choices[0].message.content)
        return {
            "key": "prompt_compliance",
            "score": result["score"] /10, 
            "reason": result["explanation"]
        }
    except json.JSONDecodeError:
            return {
                "key": "prompt_compliance",
                "score": 0,
                "reason": "Failed to parse evaluator response"
            }
# The name or UIUI of the Langsmith dataset to evaluate on 
data = "Week1Project"

#A Strin to prefix the experiment name with. 
experiment_prefix = "Portuguese-Compliance"

#List of evaluators to score the outputs of target task 
evaluators = [
     prompt_compliance_evaluator
]

#evalute the target task
results = evaluate(
    lambda inputs: inputs, 
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
)




