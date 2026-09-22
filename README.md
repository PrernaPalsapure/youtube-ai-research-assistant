# 🎥 YouTube AI Research Assistant

An AI-powered YouTube research assistant that uses **Retrieval-Augmented Generation (RAG)** to answer questions and generate summaries from YouTube video transcripts.

## 🚀 Features

* 🔗 Accepts a YouTube video URL
* 📝 Fetches the video's English transcript
* ✂️ Splits the transcript into smaller chunks
* 📌 Stores transcript chunks in Pinecone
* 🔍 Retrieves relevant context using semantic search
* 🤖 Uses Google Gemini to generate answers
* 📝 Generates a concise video summary
* 🛡️ Grounds answers in the provided video context
* 🚫 Indicates when information is not available in the video
* 🎥 Displays the YouTube video in the Streamlit interface

## 🧠 How It Works

```text
YouTube URL
     ↓
Fetch Transcript
     ↓
Split Transcript into Chunks
     ↓
Store Chunks in Pinecone
     ↓
User Question
     ↓
Semantic Search
     ↓
Retrieve Relevant Context
     ↓
Google Gemini
     ↓
Answer Based on Video Context
```

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **YouTube Transcript API**
* **LangChain**
* **Pinecone**
* **Google Gemini**
* **python-dotenv**

## 📂 Project Structure

```text
youtube-ai-research-assistant/
│
├── app.py
├── evaluation.py
├── RAG_using_langchain11.ipynb
├── Screenshot (613).png
├── Youtube_Project (1).mp4
├── LICENSE
└── README.md
```

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/PrernaPalsapure/youtube-ai-research-assistant.git
cd youtube-ai-research-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

Install the required packages used by the project:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Keep your API keys private and **do not upload the `.env` file to GitHub**.

### 5. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 💬 Example Questions

After loading a YouTube video, users can ask questions such as:

* What is the main topic of the video?
* What are the key points discussed?
* What tips does the speaker provide?
* What does the speaker recommend for practicing English?

The application retrieves relevant transcript chunks before generating the answer.

## ✨ Video Summary

The application also provides a **Summarize Video** option that generates:

* **Main Topic**
* **Key Points**
* **Important Takeaways**

The summary is generated from the video's transcript.

## 🧪 Evaluation

The RAG pipeline was evaluated using **6 questions** covering different sections of a YouTube transcript, including an out-of-context question.

The evaluation checked both:

* Retrieval of relevant transcript context
* Generation of answers based on the retrieved context

For the out-of-context question, the system correctly indicated that the information was not available in the video.

The evaluation code is available in:

```text
evaluation.py
```

## 📓 Notebook

`RAG_using_langchain11.ipynb` contains the earlier development and experimentation work related to the RAG pipeline.

## 📸 Screenshot

![YouTube AI Research Assistant](Screenshot%20%28613%29.png)

## 🎥 Demo Video

The project demo is available in:

```text
Youtube_Project (1).mp4
```

## 🔮 Future Improvements

* Timestamp-based source references
* Conversation history
* Support for multiple languages
* Additional document and video sources

## 👩‍💻 Author

**Prerna Palsapure**

AI/ML Engineer | Python | Machine Learning | Deep Learning | Generative AI | LLMs | NLP
