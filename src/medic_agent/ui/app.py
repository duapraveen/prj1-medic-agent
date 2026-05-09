import streamlit as st

from medic_agent.config.settings import AVAILABLE_MODELS, DEFAULT_MODEL_NAME, DEFAULT_SYSTEM_PROMPT
from medic_agent.llm.client import ask

st.set_page_config(page_title="medic-agent", page_icon="🏥", layout="centered")
st.title("medic-agent")
st.caption("Healthcare AI assistant powered by Claude")

model_name = st.selectbox(
    "Model",
    options=list(AVAILABLE_MODELS.keys()),
    index=list(AVAILABLE_MODELS.keys()).index(DEFAULT_MODEL_NAME),
)
model_id = AVAILABLE_MODELS[model_name]

user_query = st.text_area(
    "Your question",
    placeholder="Ask a clinical, administrative, or patient-facing healthcare question...",
    height=120,
)

if st.button("Ask", type="primary", disabled=not user_query.strip()):
    with st.spinner("Thinking..."):
        try:
            response = ask(
                model_id=model_id,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                user_query=user_query,
            )
            st.session_state["response"] = response
            st.session_state["response_model"] = model_name
        except RuntimeError as e:
            st.error(str(e))

if "response" in st.session_state:
    st.divider()
    st.caption(f"Response · {st.session_state['response_model']}")
    st.markdown(st.session_state["response"])
