import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("youtube-ai-research")

video_id = "s2EYIDY8wSM"

gemini_api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=gemini_api_key,
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant answering questions about a YouTube video.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:

"I don't have enough information from the video to answer that."

Context:

{context}

Question:

{question}
""")

chain = prompt | llm

questions = [
    "What is the main topic of the video?",
    "What does the speaker do in the morning?",
    "What are the three tips for learning English every day?",
    "How does the speaker recommend practicing speaking?",
    "What does the video say about making mistakes while learning English?",
    "What is the capital of France?"
]

print("Evaluation setup successful!")
print("Video ID:", video_id)
print("Number of questions:", len(questions))

print("\nStarting retrieval evaluation...\n")

for i, question in enumerate(questions, start=1):

    results = index.search(
        namespace="youtube",
        query={
            "inputs": {
                "text": question
            },
            "top_k": 4,
            "filter": {
                "video_id": {
                    "$eq": video_id
                }
            }
        }
    )

    print(f"Question {i}: {question}")
    print("-" * 60)

    for j, hit in enumerate(results["result"]["hits"], start=1):

        print(f"\nRetrieved Chunk {j}:")
        print(hit["fields"]["text"][:300])

    print("\n")

print("\nStarting answer evaluation...\n")

for i, question in enumerate(questions, start=1):

    results = index.search(
        namespace="youtube",
        query={
            "inputs": {
                "text": question
            },
            "top_k": 4,
            "filter": {
                "video_id": {
                    "$eq": video_id
                }
            }
        }
    )

    contexts = [
        hit["fields"]["text"]
        for hit in results["result"]["hits"]
    ]

    context = "\n\n".join(contexts)

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print(f"Question {i}: {question}")
    print("-" * 60)
    print("Answer:")
    print(response.text())
    print("\n")