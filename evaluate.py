from dotenv import load_dotenv
import os
from openai import OpenAI
from agent import indices
from rag import search

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

test_cases = [
    {
        "question": "What was Rabobank's net profit in 2024?",
        "expected": "5098"
    },
    {
        "question": "What are the biggest risk factors for ING in 2024?",
        "expected": "geopolitical"
    },
    {
        "question": "What is ING's CET1 ratio in 2024?",
        "expected": "13"
    }
]

def get_answer_with_chunks(question, bank):
    """Get an answer and the retrieved context chunks for a question.
    
    Args:
        question: Question string.
        bank: Bank name key matching indices dict.
    Returns:
        Tuple of (answer string, context string).
    """
    chunks = search(question, indices[bank]["chunks"], indices[bank]["index"])
    context = "\n\n".join([f"Page {c['page']}: {c['text']}" for c in chunks])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst. Answer based only on the provided context. If the answer is not in the context, say 'NOT FOUND'."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content, context

def judge_answer(question, answer, context):
    """Use an LLM to evaluate whether an answer is grounded in the context.
    
    Args:
        question: The original question.
        answer: The answer to evaluate.
        context: The source context the answer should be based on.
    Returns:
        'GROUNDED' or 'NOT GROUNDED'.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an evaluator. Given a question, an answer, and a context, determine if the answer is grounded in the context. Reply with only 'GROUNDED' or 'NOT GROUNDED'."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"}
        ]
    )
    return response.choices[0].message.content.strip()

print("Running evaluation...\n")
results = []

for test in test_cases:
    bank = "rabobank" if "rabobank" in test["question"].lower() else "ing"
    answer, context = get_answer_with_chunks(test["question"], bank)
    grounded = judge_answer(test["question"], answer, context)
    correct = test["expected"].lower() in answer.lower()
    
    results.append({
        "question": test["question"],
        "answer": answer,
        "grounded": grounded,
        "correct": correct
    })
    
    print(f"Question: {test['question']}")
    print(f"Answer: {answer}")
    print(f"Grounded: {grounded}")
    print(f"Contains expected: {correct}")
    print("-" * 50)

grounded_count = sum(1 for r in results if r["grounded"] == "GROUNDED")
correct_count = sum(1 for r in results if r["correct"])

print(f"\nResults: {correct_count}/{len(results)} correct")
print(f"Grounded: {grounded_count}/{len(results)} grounded in source documents")