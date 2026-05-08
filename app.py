import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Set page configuration
st.set_page_config(
    # Title and description
    page_title="Mini Rubber Ducky 🦆",
    page_icon="🦆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("🔑 OPENAI_API_KEY is missing. Set it in .env or Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

# Define the model
MODEL = "gpt-4.1"

# Render visible app content
st.title("Mini Rubber Ducky 🦆")
st.write("👋 Welcome to Mini Rubber Ducky baby!")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None


# Function to handle user input and generate response


def load_vector_store():
    """Load the vector store ID from .env, then fall back to Streamlit secrets."""
    try:
        vector_store_id = os.getenv("VECTOR_STORE_ID")

        if not vector_store_id:
            try:
                vector_store_id = st.secrets.get("VECTOR_STORE_ID")
            except Exception:
                pass

        return vector_store_id
    except Exception as e:
        st.error(f"⚠️ Error loading vector store id: {e}")
        return None


# Define the initial message
INITIAL_MESSAGE = """🦆 Hi! I am Mini Rubber Ducky, your friendly assistant.
How can I help you today?
I can help you in many modes
"""

# Define instructions
INSTRUCTIONS = """You are Mini Rubber Ducky, the course assistant for a multimodal RAG assistant course.
Be concise, direct, and practical. Use active voice. No fluff.
Primary objective
- Answer questions about the course content and code using the attached Vector Store
(transcripts, notebooks, scripts).
- Prefer retrieved facts over memory. If the files don&#39;t cover it, say so.
- Focus on multimodal RAG concepts, implementations, and best practices.
Retrieval &amp; citations
- Always use File Search first.
- Ground every substantive answer in retrieved snippets.
- If nothing relevant is found, say: &quot;I don&#39;t see this in the course files.&quot; Then suggest the most
relevant module(s) the learner should review.
- Never include source citations or reference labels in the final answer text.
Answer style
- Keep outputs scannable: short paragraphs, bullet steps, compact runnable code samples
when needed.
- When explaining &quot;how to build X&quot;, outline the pipeline stages (ingest → retrieve → generate →
evaluate) before diving into code.
- Close each reply with a friendly follow-up question the learner might ask next.
- Stay approachable, encouraging, and human.
Boundaries
- Don&#39;t invent references, credentials, metrics, or file names.
- If the topic is outside multimodal RAG/this curriculum, acknowledge the gap and offer a high-
level pointer or ask for clarification.
Context: Course focus
- Multimodal RAG: handling text, images, audio, and video in retrieval-augmented generation
systems
- Vector stores and embeddings for multimodal data
- Integration patterns and architectures
- Practical implementations and code examples
If the learner references a lecture/section by name/number, search for files with that stem and
tailor the answer.
Never invent lecture numbers or titles—they change over time.
If the answer isn&#39;t in the corpus, say so clearly.
"""

# Build the ask_bot function


def ask_bot(user_prompt: str):
    """Send questions to OpenAI and get responses."""
    common_kwargs = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "text": {"verbosity": "medium"},
    }
    if st.session_state.vector_store_id:
        common_kwargs["tools"] = [
            {
                "type": "file_search",
                "vector_store_ids": [st.session_state.vector_store_id],
                "max_num_results": 20,
            }
        ]

    if st.session_state.previous_response_id:
        resp = client.responses.create(
            previous_response_id=st.session_state.previous_response_id,
            input=[{"role": "user", "content": user_prompt}],
            **common_kwargs
        )
    else:
        resp = client.responses.create(
            input=[
                {"role": "user", "content": INITIAL_MESSAGE.strip()},
                {"role": "user", "content": user_prompt},
            ],
            **common_kwargs
        )

    st.session_state.previous_response_id = resp.id
    return resp.output_text


# Build function to reset conversation history


def reset_conversation():
    """Reset the conversation history."""
    st.session_state.messages = [{
        "role": "assistant",
        "content": INITIAL_MESSAGE.strip(),
    }]
    st.session_state.previous_response_id = None
    st.rerun()


def main():
    # Load vector store once per session. If unset, chat still works without file search.
    if st.session_state.vector_store_id is None:
        st.session_state.vector_store_id = load_vector_store()

    # Sidebar with reset button
    with st.sidebar:
        st.header("⚙️ Settings")
        if st.button("🔄 Reset Conversation"):
            reset_conversation()

    # Initialize the messages
    if not st.session_state.messages:
        st.session_state.messages = [{
            "role": "assistant",
            "content": INITIAL_MESSAGE.strip(),
        }]

    # Display the entire conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    prompt = st.chat_input("💬 Ask me anything about multimodal RAG")
    if prompt:
        st.session_state.messages.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the user input
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = ask_bot(prompt)
            st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
