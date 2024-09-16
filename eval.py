from dotenv import load_dotenv
import openai
from langsmith.wrappers import wrap_openai
from metadata import SPANISH_PORTUGUESE_COGNATES

import requests
from bs4 import BeautifulSoup

load_dotenv()


def get_web_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None
    

#Get the english and portuguese cognates
url = "https://duolingo.fandom.com/wiki/User_blog:HelpfulDuoFan/100_Portuguese-English_cognates:_Nouns"
response = get_web_content(url)
soup = BeautifulSoup(response, 'html.parser')
table = soup.find('table', class_='fandom-table')

#Get the english and portuguese cognates
cognates = []
for row in table.find_all('tr'):
    cols = row.find_all('td')
    if len(cols) >= 2:
        english = cols[0].text.strip()
        portuguese = cols[1].text.strip()
        cognates.append((portuguese, english))
cognates = []

#Append all English cognates 
#append all the portuguese cognates to the text
full_text = ""
for cognate in cognates:
    full_text += f"{cognate[0]} is similar in english to {cognate[1]}\n"

#Append all Spanish cognates 
for cognate in SPANISH_PORTUGUESE_COGNATES:
    full_text += f"{cognate[0]} is similar in spanish to {cognate[1]}\n"

openai_client = wrap_openai(openai.Client())

def answer_translate_question(inputs: dict) -> dict:
    """
    Generates answers to user questions on portuguese based on a provided set of cognates in English and Spanish. 

    Parameters:
    inputs (dict): A dictionay with a single key 'question', representing the user's question as a string. 

    Returns:
    dict: A dictionary with a single key 'output', containing the generated answer as a string. 
    """
    #System prompt 
    system_msg = (
        f"Answer user questions in 2-3 sentences about this context: \n\n\n{full_text}"
    )

    #Pass in website text
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": inputs["input"]},
    ]

    #Call OpenAI
    response = openai_client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
    )

    #Response in output dict
    return {"answer": response.dict()["choices"][0]["message"]["content"]}

from langsmith.evaluation import evaluate, LangChainStringEvaluator

#Evaluators 
qa_evaluator = [LangChainStringEvaluator("cot_qa")]
dataset_name = "Week1Project"

experiment_results = evaluate(
    answer_translate_question,
    data=dataset_name,
    evaluators=qa_evaluator,
    experiment_prefix="Portuguese-Compliance-Eval1",
    metadata={
        "variant": "stuff website context into gpt-4o"
    },
)

print(experiment_results)