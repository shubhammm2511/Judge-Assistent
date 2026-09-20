import html
import re
import uuid
from pathlib import Path

import streamlit as st

from rag_pipeline import stream_answer
from retriever import retrieve_chunks
from vector_store import build_vector_store, get_collection


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "Agent-as-Judge.pdf"
DATABASE_PATH = PROJECT_ROOT / "chroma_db"


st.set_page_config(
    page_title="Agent-as-a-Judge Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def ensure_vector_store() -> int:
    """Build the local index once when running on a fresh deployment."""
    collection = get_collection(DATABASE_PATH)

    if collection.count() == 0:
        build_vector_store(PDF_PATH, DATABASE_PATH)
        collection = get_collection(DATABASE_PATH)

    return collection.count()


try:
    with st.spinner("Preparing the paper index..."):
        ensure_vector_store()
except Exception as error:
    st.error(f"The paper index could not be prepared: {error}")
    st.stop()


def create_conversation() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "title": "New conversation",
        "messages": [],
    }


if "conversations" not in st.session_state:
    first_conversation = create_conversation()
    st.session_state.conversations = [first_conversation]
    st.session_state.active_chat_id = first_conversation["id"]

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "top_k" not in st.session_state:
    st.session_state.top_k = 8


def active_conversation() -> dict:
    for conversation in st.session_state.conversations:
        if conversation["id"] == st.session_state.active_chat_id:
            return conversation

    st.session_state.active_chat_id = st.session_state.conversations[0]["id"]
    return st.session_state.conversations[0]


def new_chat() -> None:
    conversation = create_conversation()
    st.session_state.conversations.append(conversation)
    st.session_state.active_chat_id = conversation["id"]
    st.rerun()


def select_chat(chat_id: str) -> None:
    st.session_state.active_chat_id = chat_id
    st.rerun()


def answer_to_html(answer: str) -> str:
    """Render the model's limited Markdown safely inside the custom bubble."""
    safe_answer = html.escape(answer)
    safe_answer = re.sub(
        r"\*\*(.+?)\*\*",
        r"<strong>\1</strong>",
        safe_answer,
        flags=re.DOTALL,
    )
    safe_answer = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe_answer)
    return safe_answer.replace("\n", "<br>")


dark = st.session_state.dark_mode

theme_css = """
:root {
    --app-bg: #f7f8fc;
    --sidebar-bg: #ffffff;
    --surface: #ffffff;
    --surface-soft: #f1f3f8;
    --text: #151827;
    --muted: #73788b;
    --border: #e6e8f0;
    --accent: #5865f2;
    --accent-strong: #4653dd;
    --assistant-border: rgba(88, 101, 242, .26);
    --shadow: 0 20px 60px rgba(31, 38, 74, .08);
}
""" if not dark else """
:root {
    --app-bg: #0d1018;
    --sidebar-bg: #111520;
    --surface: #151a26;
    --surface-soft: #1c2230;
    --text: #f2f4fb;
    --muted: #9aa2b7;
    --border: #293044;
    --accent: #7180ff;
    --accent-strong: #8995ff;
    --assistant-border: rgba(113, 128, 255, .42);
    --shadow: 0 20px 60px rgba(0, 0, 0, .24);
}
"""

st.markdown(
    f"""
    <style>
    {theme_css}

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, #root, [class*="css"], .stApp {{
        font-family: 'DM Sans', sans-serif;
    }}

    html, body, #root, .stApp, [data-testid="stAppViewContainer"],
    section.main, section[data-testid="stMain"],
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"] {{
        background: var(--app-bg) !important;
        color: var(--text);
    }}

    section[data-testid="stMain"] > div,
    [data-testid="stAppViewContainer"] > .main {{
        background: var(--app-bg) !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding: 1.25rem .9rem;
    }}

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        color: var(--text) !important;
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    .main .block-container {{
        max-width: 1120px;
        padding: 2rem 2.5rem 7rem;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: .65rem;
        margin: .25rem .35rem 1.5rem;
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
    }}

    .brand-mark {{
        display: grid;
        width: 30px;
        height: 30px;
        place-items: center;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent), #9c75ff);
        color: white;
        box-shadow: 0 8px 20px rgba(88, 101, 242, .28);
    }}

    .sidebar-label {{
        margin: 1.4rem .35rem .55rem;
        color: var(--muted);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}

    .stButton > button {{
        min-height: 2.45rem;
        border: 1px solid var(--border);
        border-radius: 11px;
        background: var(--surface);
        color: var(--text);
        font-weight: 500;
        text-align: left;
        transition: border .18s ease, background .18s ease, transform .18s ease;
    }}

    .stButton > button:hover {{
        border-color: var(--accent);
        background: var(--surface-soft);
        color: var(--text);
        transform: translateY(-1px);
    }}

    .new-chat button {{
        border: 0 !important;
        background: var(--accent) !important;
        color: white !important;
        text-align: center !important;
    }}

    .settings-title {{
        margin: 0 0 .45rem;
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 600;
    }}

    .chat-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2.25rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.1rem;
    }}

    .chat-title {{
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 600;
    }}

    .chat-subtitle {{
        margin-top: .2rem;
        color: var(--muted);
        font-size: .8rem;
    }}

    .empty-state {{
        display: flex;
        min-height: 62vh;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}

    .empty-icon {{
        display: grid;
        width: 66px;
        height: 66px;
        place-items: center;
        margin-bottom: 1.2rem;
        border: 1px solid var(--assistant-border);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(88,101,242,.18), rgba(156,117,255,.14));
        color: var(--accent);
        font-size: 2rem;
        box-shadow: 0 14px 30px rgba(88,101,242,.12);
    }}

    .empty-title {{
        margin: 0;
        color: var(--text);
        font-size: clamp(1.8rem, 4vw, 2.5rem);
        font-weight: 600;
        letter-spacing: -.04em;
    }}

    .empty-copy {{
        max-width: 460px;
        margin-top: .7rem;
        color: var(--muted);
        font-size: .95rem;
        line-height: 1.6;
    }}

    .message-row {{
        display: flex;
        gap: .85rem;
        margin: 1.35rem 0;
        align-items: flex-start;
    }}

    .message-row.user {{
        justify-content: flex-end;
    }}

    .avatar {{
        display: grid;
        flex: 0 0 34px;
        width: 34px;
        height: 34px;
        place-items: center;
        border-radius: 12px;
        font-size: .77rem;
        font-weight: 700;
    }}

    .bot-avatar {{
        order: 0;
        background: linear-gradient(135deg, var(--accent), #9c75ff);
        color: #ffffff;
        box-shadow: 0 7px 18px rgba(88,101,242,.25);
    }}

    .user-avatar {{
        order: 2;
        background: var(--surface-soft);
        color: var(--muted);
    }}

    .bubble {{
        max-width: min(74%, 760px);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: .85rem 1rem;
        color: var(--text);
        font-size: .94rem;
        line-height: 1.65;
        white-space: normal;
        box-shadow: 0 8px 24px rgba(31, 38, 74, .04);
    }}

    .assistant-bubble {{
        background: var(--surface);
        border-left: 2px solid var(--accent);
    }}

    .user-bubble {{
        order: 1;
        background: var(--accent);
        border-color: var(--accent);
        color: #ffffff;
        box-shadow: 0 10px 22px rgba(88,101,242,.2);
    }}

    .typing-bubble {{
        display: flex;
        align-items: center;
        gap: 5px;
        min-width: 64px;
        padding: 1rem;
    }}

    .dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 1.2s infinite ease-in-out;
    }}

    .dot:nth-child(2) {{ animation-delay: .16s; }}
    .dot:nth-child(3) {{ animation-delay: .32s; }}

    @keyframes pulse {{
        0%, 70%, 100% {{ opacity: .3; transform: translateY(0); }}
        35% {{ opacity: 1; transform: translateY(-4px); }}
    }}

    .sources-heading {{
        margin: .55rem 0 .25rem 3.3rem;
        color: var(--muted);
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}

    [data-testid="stChatInput"] {{
        position: fixed;
        z-index: 20;
        bottom: 1.15rem;
        left: calc(50% + 135px);
        width: min(760px, calc(100vw - 430px));
        transform: translateX(-50%);
    }}

    [data-testid="stChatInput"] > div {{
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--surface);
        box-shadow: var(--shadow);
    }}

    [data-testid="stChatInput"] textarea {{
        background: transparent !important;
        color: var(--text);
        font-size: .94rem;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--muted) !important;
        opacity: 1;
    }}

    @media (max-width: 900px) {{
        .main .block-container {{ padding: 1.2rem 1rem 6.5rem; }}
        [data-testid="stChatInput"] {{ left: 50%; width: calc(100vw - 2rem); }}
        .bubble {{ max-width: 84%; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


conversation = active_conversation()

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">✦</span>Judge Assistant</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋  New chat", use_container_width=True):
        new_chat()

    st.markdown('<div class="sidebar-label">Chat history</div>', unsafe_allow_html=True)

    for item in reversed(st.session_state.conversations):
        title = item["title"]
        if title == "New conversation" and item["id"] == conversation["id"]:
            title = "Current conversation"

        if st.button(
            title[:34],
            key=f"history_{item['id']}",
            use_container_width=True,
        ):
            select_chat(item["id"])

    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    st.toggle("Dark mode", key="dark_mode")
    st.slider("Retrieved chunks", 1, 8, key="top_k")


st.markdown(
    f"""
    <div class="chat-header">
        <div>
            <div class="chat-title">Agent-as-a-Judge Assistant</div>
            <div class="chat-subtitle">Grounded answers from the research paper</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_message(message: dict) -> None:
    role = message["role"]
    content = answer_to_html(message["content"])

    if role == "user":
        st.markdown(
            f"""
            <div class="message-row user">
                <div class="bubble user-bubble">{content}</div>
                <div class="avatar user-avatar">YOU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="message-row assistant">
            <div class="avatar bot-avatar">✦</div>
            <div class="bubble assistant-bubble">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if not conversation["messages"]:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <h1 class="empty-title">How can I help with the paper?</h1>
            <div class="empty-copy">
                Ask a question and I’ll retrieve the most relevant passages,
                then answer with page-level evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for message in conversation["messages"]:
        render_message(message)

        if message["role"] == "assistant" and message.get("retrieved_chunks"):
            st.markdown(
                '<div class="sources-heading">Retrieved sources</div>',
                unsafe_allow_html=True,
            )
            for index, chunk in enumerate(message["retrieved_chunks"], start=1):
                metadata = chunk["metadata"]
                label = (
                    f"Chunk {index} · {metadata['source']} · "
                    f"Page {metadata['page']} · "
                    f"Similarity {chunk['similarity']:.4f}"
                )
                with st.expander(label):
                    st.write(chunk["text"])


question = st.chat_input("Message Agent-as-a-Judge Assistant")

if question:
    if conversation["title"] == "New conversation":
        conversation["title"] = question[:34]

    conversation["messages"].append({"role": "user", "content": question})
    render_message({"role": "user", "content": question})

    with st.spinner("Searching the paper..."):
        retrieved_chunks = retrieve_chunks(
            question=question,
            database_path=DATABASE_PATH,
            top_k=st.session_state.top_k,
        )

    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        """
        <div class="message-row assistant">
            <div class="avatar bot-avatar">✦</div>
            <div class="bubble assistant-bubble typing-bubble">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    answer_placeholder = st.empty()
    answer_text = ""

    for token in stream_answer(question, retrieved_chunks):
        answer_text += token
        safe_answer = answer_to_html(answer_text)
        typing_placeholder.empty()
        answer_placeholder.markdown(
            f"""
            <div class="message-row assistant">
                <div class="avatar bot-avatar">✦</div>
                <div class="bubble assistant-bubble">{safe_answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    conversation["messages"].append(
        {
            "role": "assistant",
            "content": answer_text,
            "retrieved_chunks": retrieved_chunks,
        }
    )

    st.markdown('<div class="sources-heading">Retrieved sources</div>', unsafe_allow_html=True)
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk["metadata"]
        label = (
            f"Chunk {index} · {metadata['source']} · "
            f"Page {metadata['page']} · "
            f"Similarity {chunk['similarity']:.4f}"
        )
        with st.expander(label):
            st.write(chunk["text"])
