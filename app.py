import youtube_transcript_api as yta
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from google.api_core.exceptions import ResourceExhausted
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import re


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


# -----------------------------
# Pinecone connection
# -----------------------------

pc = Pinecone(api_key=pinecone_api_key)

index = pc.Index("youtube-ai-research")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# -----------------------------
# Gemini model
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=gemini_api_key,
    temperature=0.2
)


# -----------------------------
# Prompt
# -----------------------------

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

summary_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant summarizing a YouTube video.

Create a short, clear, and easy-to-read summary using only the provided transcript.

Use exactly this structure:

### Main Topic
Give 1-2 sentences describing the main topic.

### Key Points
Give 3-5 important points as bullet points.

### Important Takeaways
Give 2-3 short practical takeaways from the video.

Do not add information that is not present in the transcript.

Transcript:
{transcript}
""")

summary_chain = summary_prompt | llm

# -----------------------------
# Extract YouTube video ID
# -----------------------------

def extract_video_id(url):

    pattern = r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})"

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    return None


# -----------------------------
# Get YouTube transcript
# -----------------------------

def get_transcript(video_id):

    ytt_api = yta.YouTubeTranscriptApi()

    transcript_list = ytt_api.fetch(
        video_id,
        languages=["en"]
    )

    return " ".join(
        snippet.text
        for snippet in transcript_list
    )


# -----------------------------
# Upload transcript to Pinecone
# -----------------------------

def upload_transcript(video_id, youtube_url, transcript):

    chunks = splitter.create_documents([transcript])

    records = []

    for i, chunk in enumerate(chunks):

        records.append({
            "_id": f"{video_id}-chunk-{i}",
            "text": chunk.page_content,
            "video_id": video_id,
            "chunk_number": i,
            "source": youtube_url
        })

    index.upsert_records(
        namespace="youtube",
        records=records
    )

    return len(records)


# -----------------------------
# Ask question about video
# -----------------------------

def ask_video(question, video_id):

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

    sources = list({
        hit["fields"].get("source")
        for hit in results["result"]["hits"]
    })

    try:

        response = chain.invoke({
            "context": context,
            "question": question
        })

    except ResourceExhausted:

        return (
            "Gemini API quota has been reached. Please try again later.",
            sources,
            contexts
        )

    return response.text(), sources, contexts


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="YouTube AI Research Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube AI Research Assistant")

if "loaded_video_id" not in st.session_state:
    st.session_state.loaded_video_id = None

if "transcript" not in st.session_state:
    st.session_state.transcript = None

st.write(
    "Ask questions about the YouTube video using its transcript."
)


# -----------------------------
# YouTube URL
# -----------------------------

youtube_url = st.text_input(
    "Enter YouTube URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

youtube_url = youtube_url.strip().strip('"').strip("'")

if youtube_url:
    st.video(youtube_url)


# -----------------------------
# Load Video
# -----------------------------

if st.button("Load Video"):

    if not youtube_url:

        st.warning("Please enter a YouTube URL.")

    else:

        video_id = extract_video_id(youtube_url)

        if not video_id:

            st.error("Please enter a valid YouTube URL.")

        else:

            with st.spinner("Loading YouTube transcript..."):

                transcript = get_transcript(video_id)

                upload_transcript(
                    video_id,
                    youtube_url,
                    transcript
                )
                st.session_state.transcript = transcript
         
            st.session_state.loaded_video_id = video_id

            st.success(
                "Video loaded successfully! You can now ask questions."
            )


# -----------------------------
# Question
# -----------------------------

question = st.text_area(
    "Ask a question",
    placeholder="What are Large Language Models?",
    height=100
)


# -----------------------------
# Ask button
# -----------------------------

if st.button("Ask"):

    if not youtube_url:

        st.warning("Please enter a YouTube URL.")

    elif st.session_state.loaded_video_id is None:

        st.warning("Please load the video first.")

    elif not question:

        st.warning("Please enter a question.")

    else:

        video_id = st.session_state.loaded_video_id

        with st.spinner("Searching the video..."):

            answer, sources, contexts = ask_video(
                question,
                video_id
            )

        st.subheader("Answer")

        st.write(answer)

        with st.expander("View Retrieved Context"):

            for i, chunk in enumerate(contexts, start=1):

                st.write(f"**Retrieved Chunk {i}**")

                st.write(chunk)

        if sources:

            st.subheader("Source")

            for source in sources:

                st.markdown(
                    f"[Watch YouTube Video]({source})"
                )

if st.button("✨ Summarize Video"):

    if st.session_state.loaded_video_id is None:

        st.warning("Please load the video first.")

    else:

        with st.spinner("Generating video summary..."):

            try:

                response = summary_chain.invoke({
                    "transcript": st.session_state.transcript
                })

                st.subheader("📝 Video Summary")

                st.write(response.text())

            except ResourceExhausted:

                st.error(
                    "Gemini API quota has been reached. Please try again later."
                )