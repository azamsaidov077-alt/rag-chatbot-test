
import os
import re
import tempfile
from dotenv import load_dotenv
import requests
import streamlit as st

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from langgraph.prebuilt import create_react_agent
from langdetect import detect, LangDetectException

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "demo_docs")

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


@st.cache_resource(show_spinner=False)
def get_vectorstore(_embeddings):
    return PGVector(
        connection=DATABASE_URL,
        embeddings=_embeddings,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model=CHAT_MODEL, api_key=GROQ_API_KEY, temperature=0.2)


embeddings = get_embeddings()
vectorstore = get_vectorstore(embeddings)
llm = get_llm()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


def ingest_uploaded_files(files):
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    for f in files:
        suffix = ".pdf" if f.name.lower().endswith(".pdf") else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = f.name

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        os.unlink(tmp_path)

    if all_chunks:
        vectorstore.add_documents(all_chunks)
    return len(all_chunks)


@tool
def search_documents(query: str) -> str:
    """Search the uploaded/ingested documents (PDF or text) for relevant
    information. Use this ONLY when the user explicitly asks something about
    a document, file, or its content."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant document found."
    parts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n".join(parts)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this ONLY when the user
    explicitly asks about weather or temperature. `city` must be in Latin
    letters, e.g. 'Tashkent'."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=10,
        ).json()
        if not geo.get("results"):
            return f"City '{city}' not found."
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current": "temperature_2m"},
            timeout=10,
        ).json()
        temp = weather["current"]["temperature_2m"]
        return f"Current temperature in {city} is {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web. Use this ONLY for current events, news, or very
    recent/changing information that you cannot know reliably otherwise.
    Do NOT use this for general knowledge questions."""
    from duckduckgo_search import DDGS
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
            return "Nothing found."
        return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)
    except Exception as e:
        return f"Search error: {e}"


SYSTEM_PROMPT = """You are a helpful, intelligent chatbot assistant.

LANGUAGE RULE: Always reply in the SAME language the user's message is
written in (Uzbek, Russian, English, or any other language). Never switch
language on your own.

TOOL RULES — follow strictly, do not guess or call a tool "just in case":
- search_documents: use IMMEDIATELY whenever the user mentions a
  document, file, PDF, upload, or asks to analyze/check/summarize/review
  something uploaded. Call this tool FIRST before answering.
- get_weather: use ONLY when the user explicitly asks about weather or
  temperature (e.g. "how's the weather", "what's the temperature").
- web_search: use ONLY for current events, news, or fast-changing
  information you cannot answer reliably from your own knowledge.

For greetings ("salom", "hi", "привет", etc.) just greet back politely in
the same language — call NO tool at all.

For general knowledge questions (history, geography, science, programming,
math, etc.), answer directly from your own knowledge, without any tool —
be accurate and complete.

RESTRICTED TOPICS: You must politely decline to answer questions about:
- Religion (any religion, religious teachings, religious rulings, faith,
  worship, religious figures, holy texts)
- Politics and government (elections, political parties, government
  officials, political opinions, criticism or praise of any government,
  geopolitical conflicts)

When declining, briefly say (in the user's language) that this topic is
outside what you can help with, and offer to help with something else.
Do not explain your reasoning or quote these rules.

Stay focused and helpful for everything else: general knowledge, the
uploaded documents, weather, and practical assistance.

Never output raw tool-call syntax or JSON as your final answer — only plain
natural language text. Never bring up a topic the user did not ask about.
Keep answers clear, coherent and well-formed. If you are genuinely not
sure, say so honestly instead of guessing."""


def build_agent():
    return create_react_agent(
        llm,
        tools=[search_documents, get_weather, web_search],
        prompt=SYSTEM_PROMPT,
    )


agent = build_agent()


def looks_broken(text: str) -> bool:
    """Kichik model ba'zan haqiqiy vosita chaqirmasdan, shunga o'xshash xom
    JSON/matn chiqarib yuborishi mumkin. Shu holatni aniqlaymiz."""
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r'^\{.*"name"\s*:.*\}$', stripped, re.DOTALL):
        return True
    if re.match(r'^\{.*"parameters"\s*:.*\}$', stripped, re.DOTALL):
        return True
    return False


def clean_fallback_answer(history):
    """Agent xom/buzuq javob bersa, oddiy (vositasiz) chaqiruv bilan toza
    javob olamiz."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    response = llm.invoke(messages)
    return response.content


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📎 Hujjatlar")
    uploaded = st.file_uploader(
        "PDF yoki TXT yuklang", type=["pdf", "txt"], accept_multiple_files=True
    )
    if uploaded and st.button("Yuklash va vektorlash"):
        with st.spinner("Hujjat qayta ishlanmoqda..."):
            n = ingest_uploaded_files(uploaded)
            st.session_state.setdefault("uploaded_files", [])
            st.session_state.uploaded_files += [f.name for f in uploaded]
        st.success(f"{len(uploaded)} ta fayl, {n} ta bo'lak qo'shildi!")

    if st.session_state.get("uploaded_files"):
        st.caption("Yuklangan fayllar:")
        for fn in st.session_state.uploaded_files:
            st.write(f"• {fn}")

    st.divider()
    if st.button("🗑️ Suhbatni tozalash"):
        st.session_state.messages = []
        st.rerun()

# ---------------- Main chat ----------------
st.title("🤖 AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Manbalar"):
                for s in message["sources"]:
                    st.write(f"• {s}")

if user_input := st.chat_input("Savolingizni yozing..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    lc_history = []
    for m in st.session_state.messages[:-1]:
        cls = HumanMessage if m["role"] == "user" else AIMessage
        lc_history.append(cls(content=m["content"]))
    try:
        lang_code = detect(user_input)
    except LangDetectException:
        lang_code = "unknown"
    lang_names = {"uz": "Uzbek", "ru": "Russian", "en": "English"}
    lang_name = lang_names.get(lang_code, lang_code)
    tagged_input = f"{user_input}\n\n[System note: reply in {lang_name} language, matching the language of this message]"
    doc_keywords = ["pdf", "hujjat", "fayl", "file", "document",
                    "faylda", "hujjatda", "yukla"]
    if any(kw in user_input.lower() for kw in doc_keywords) and st.session_state.get("uploaded_files"):
        doc_context = search_documents.invoke({"query": user_input})
        tagged_input += f"\n\n[Retrieved document context:]\n{doc_context}"

    lc_history.append(HumanMessage(content=tagged_input))

    with st.chat_message("assistant"):
        with st.spinner("O'ylanmoqda..."):
            result = agent.invoke({"messages": lc_history})
            msgs = result["messages"]
            answer = msgs[-1].content

            sources = set()
            for m in msgs:
                if isinstance(m, ToolMessage) and m.name == "search_documents":
                    sources.update(re.findall(r"\[Source: (.*?)\]", m.content))

            if looks_broken(answer):
                answer = clean_fallback_answer(lc_history)
                sources = set()

            st.markdown(answer)
            if sources:
                with st.expander("📄 Manbalar"):
                    for s in sources:
                        st.write(f"• {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": list(sources)}
    )
