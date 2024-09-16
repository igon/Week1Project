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

    # Handle inputs
    if isinstance(inputs, str):
        system_prompt = inputs
        message_history = []
        latest_message = ""
    else:
        system_prompt = next((msg['data']['content'] for msg in inputs if msg['type'] == 'system'), "")
        message_history = [
            {'role': "user" if msg['type'] == 'human' else 'assistant', 'content': msg['data']['content']}
            for msg in inputs if msg['type'] in ['human', 'ai']
        ]
        latest_message = message_history[-1]['content'] if message_history else ""

    # Handle outputs
    model_output = outputs if isinstance(outputs, str) else outputs.get('data', {}).get('content', '')

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
        model="gpt-3.5-turbo",
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
experiment_prefix = "Portuguese-Compliance-Eval2"

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


