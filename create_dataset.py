from langsmith import Client

client = Client()
dataset_name = "Week1Project"

inputs = [
    "What is the hardest language to learn?",
    "Which language is the easiest to learn?",
    "Is English not related to Spanish?",
    "Is English not related to Portuguese?",
    "I want to learn a new language, what should I do?",
    "What is the best way to learn a language?",
    "Can you teach me bad words?",
    "Im about to give up on learning a language, what should I do?",
    "How many hours a day should I spend learning a language?",
    "Who created you?",
    "Which is the best way to learn a programming language?",
]

outputs = [
    "none its a matter of dedication and time",
    "none its a matter of dedication and time",
    "no, they are somewhat similar and have some words in common",
    "You should practice",
    "You should practice",
    "Your time is better spent learning a language",
    "Take a break, and come back to it later and in the meantime talk to someone who has learned the language or listen to music or watch movies in that language",
    "You should practice as much as you can even a little bit every day",
    "I am an AI assistant not created but put together by Iván the terrible and great ",
    "Codepath is the best way to learn a programming language and I wish they also taught spoken language",
]

dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Portuguese, English and Spanish conversations for the Week 1 Project"
)

client.create_examples(
    inputs=[{"input": q} for q in inputs],
    outputs=[{"output": a} for a in outputs],
    dataset_id=dataset.id,
)

print(dataset)
