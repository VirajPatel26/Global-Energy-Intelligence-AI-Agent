
from pathlib import Path
import json
import re
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from analytics import (
    get_dataset_summary,
    get_country_generation
)

# ==================================================
# 1. PROJECT PATHS AND IMPORTS
# ==================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
DATA_DIR = BASE_DIR / "data"

# Ensure analytics.py is imported from the same folder

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from analytics import (
    get_dataset_summary,
    get_country_generation
)


# ==================================================
# 2. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Global Energy Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 3. GEMINI AI CONFIGURATION
# ==================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"

client = None

try:

    GEMINI_API_KEY = st.secrets.get(
        "GEMINI_API_KEY",
        ""
    )

    if GEMINI_API_KEY:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

except Exception:

    GEMINI_API_KEY = ""
    client = None


# ==================================================
# 4. LOAD DATASETS
# ==================================================

@st.cache_data
def load_data():

    datasets = {

        "areas": pd.read_csv(
            DATA_DIR / "dim_area.csv"
        ),

        "sources": pd.read_csv(
            DATA_DIR / "dim_source.csv"
        ),

        "years": pd.read_csv(
            DATA_DIR / "dim_year.csv"
        ),

        "energy": pd.read_csv(
            DATA_DIR / "fact_energy_indicators.csv"
        ),

        "generation": pd.read_csv(
            DATA_DIR / "fact_generation.csv"
        ),
    }

    return datasets


try:

    data = load_data()

    st.sidebar.success(
        "All datasets loaded successfully"
    )

except Exception as e:

    st.error(
        f"Unable to load datasets: {e}"
    )

    st.stop()


# ==================================================
# 5. SIDEBAR
# ==================================================

st.sidebar.title("🌍 Global Energy Intelligence")

st.sidebar.caption(
    "AI-powered global energy analytics"
)

if client:

    st.sidebar.success(
        "Gemini AI is configured"
    )

else:

    st.sidebar.warning(
        "Gemini API key is not configured"
    )


# ==================================================
# 6. APPLICATION HEADER
# ==================================================

st.title(
    "🌍 GLOBAL ENERGY INTELLIGENCE"
)

st.caption(
    "AI-powered global energy analytics "
    "and intelligence platform"
)

st.divider()


# ==================================================
# 7. GEMINI CONNECTION TEST
# ==================================================

st.subheader(
    "🤖 Gemini AI Connection"
)

st.write(
    "Verify that the AI engine is connected and ready."
)

if st.button(
    "Test Gemini Connection",
    type="primary"
):

    if client is None:

        st.error(
            "Gemini API key is missing. "
            "Check .streamlit/secrets.toml."
        )

    else:

        with st.spinner(
            "Connecting to Gemini AI..."
        ):

            try:

                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=(
                        "Reply with exactly: "
                        "Gemini connection successful."
                    )
                )

                if response.text:

                    st.success(
                        "Gemini API connected successfully!"
                    )

                    st.write(response.text)

                else:

                    st.warning(
                        "Gemini responded without text."
                    )

            except Exception as e:

                st.error(
                    "Gemini connection failed."
                )

                st.exception(e)


st.divider()


# ==================================================
# 8. DATASET OVERVIEW
# ==================================================

st.subheader(
    "Dataset Overview"
)

col1, col2, col3, col4, col5 = st.columns(5)

datasets_info = [

    ("Areas", data["areas"]),

    ("Sources", data["sources"]),

    ("Years", data["years"]),

    ("Energy Indicators", data["energy"]),

    ("Generation", data["generation"]),
]

for col, (name, df) in zip(
    [col1, col2, col3, col4, col5],
    datasets_info
):

    col.metric(
        name,
        f"{len(df):,}"
    )


st.divider()


# ==================================================
# 9. DATA PREVIEW
# ==================================================

with st.expander(
    "View Energy Indicators Data",
    expanded=False
):

    st.dataframe(
        data["energy"].head(10),
        use_container_width=True,
        hide_index=True
    )


with st.expander(
    "View Electricity Generation Data",
    expanded=False
):

    st.dataframe(
        data["generation"].head(10),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 10. AI AGENT HEADER
# ==================================================

st.divider()

st.header(
    "🤖 Global Energy Intelligence AI Agent"
)

st.caption(
    "Ask questions about electricity generation, "
    "energy sources, countries and historical trends."
)


# ==================================================
# 11. CHAT HISTORY
# ==================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# 12. AI CHAT INPUT
# ==================================================

user_question = st.chat_input(
    "Ask something about global energy..."
)


if user_question:

    if client is None:

        st.error(
            "Gemini is not configured. "
            "Please check your API key."
        )

    else:

        # Display user message

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(user_question)


        # Generate assistant response

        with st.chat_message("assistant"):

            with st.spinner(
                "Analysing global energy data..."
            ):

                try:

                    # ----------------------------------
                    # STEP 1: IDENTIFY USER INTENT
                    # ----------------------------------

                    instruction = f"""
You are an intent extraction system
for a global energy analytics application.

Read the user's question and return ONLY valid JSON.

Supported actions:

1. dataset_summary
2. country_generation
3. unsupported

For country_generation, extract:

- country: country name
- year: four-digit year if specified, otherwise null

For dataset_summary, use null for country and year.

Use dataset_summary for questions about:
- Dataset coverage
- Number of areas
- Available years
- Number of records
- Available electricity sources

Use country_generation for questions about:
- Electricity generation by source for a country
- Total generation for a country
- Country generation in a specified year

If the question cannot be answered using these actions,
return unsupported.

Do not calculate or invent statistics.

Required JSON format:

{{
    "action": "country_generation",
    "country": "India",
    "year": 2025
}}

User question:
{user_question}
"""

                    intent_response = (
                        client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=instruction
                        )
                    )

                    if not intent_response.text:

                        raise ValueError(
                            "Gemini did not return an intent."
                        )

                    intent_text = (
                        intent_response.text.strip()
                    )

                    # Remove Markdown code fences

                    intent_text = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        intent_text,
                        flags=re.IGNORECASE
                    ).strip()

                    intent = json.loads(intent_text)

                    action = intent.get("action")


                    # ----------------------------------
                    # STEP 2: EXECUTE PYTHON ANALYTICS
                    # ----------------------------------

                    if action == "dataset_summary":

                        result = get_dataset_summary()


                    elif action == "country_generation":

                        country = intent.get("country")

                        year = intent.get("year")

                        if not country:

                            raise ValueError(
                                "Please specify a country."
                            )

                        result = get_country_generation(
                            country,
                            year
                        )


                    else:

                        result = {
                            "error": (
                                "I can currently answer questions "
                                "about dataset coverage and country "
                                "electricity generation."
                            )
                        }


                    # ----------------------------------
                    # STEP 3: CHECK ANALYTICS RESULT
                    # ----------------------------------

                    if "error" in result:

                        final_answer = result["error"]


                    else:

                        # ----------------------------------
                        # STEP 4: GENERATE AI EXPLANATION
                        # ----------------------------------

                        explanation_prompt = f"""
You are Global Energy Intelligence AI.

Answer the user's question using ONLY the
verified Python analytics result provided below.

Rules:

- Never invent figures or statistics.
- Preserve the units exactly.
- Clearly mention the country and year.
- Format large numbers clearly.
- Explain results in professional, simple language.
- If requested information is absent, say so.
- Do not claim that the result contains information
  that it does not contain.
- Do not add statistics from your general knowledge.

User question:

{user_question}

Verified Python analytics result:

{json.dumps(result, indent=2)}

Provide a clear and useful answer.
"""

                        answer_response = (
                            client.models.generate_content(
                                model=GEMINI_MODEL,
                                contents=explanation_prompt
                            )
                        )

                        if not answer_response.text:

                            raise ValueError(
                                "Gemini could not generate an answer."
                            )

                        final_answer = (
                            answer_response.text.strip()
                        )


                    # ----------------------------------
                    # STEP 5: DISPLAY FINAL ANSWER
                    # ----------------------------------

                    st.markdown(final_answer)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": final_answer
                        }
                    )


                except json.JSONDecodeError:

                    error_message = (
                        "The AI returned an invalid response "
                        "while identifying your question. "
                        "Please try asking it differently."
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )


                except Exception as e:

                    error_message = (
                        f"Unable to process your question: {e}"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )


# ==================================================
# 13. CHAT CONTROLS
# ==================================================

if st.session_state.chat_history:

    if st.button(
        "Clear Chat History"
    ):

        st.session_state.chat_history = []

        st.rerun()


# ==================================================
# 14. POWER BI REPORT INTEGRATION
# ==================================================

st.divider()

st.subheader(
    "📊 Global Energy Intelligence Dashboard"
)

POWERBI_URL = (
    "https://app.powerbi.com/reportEmbed?"
    "reportId=23c5da3d-2066-4301-9bd2-8149dde3dd4c"
    "&autoAuth=true"
    "&ctid=4451b433-6541-406e-842f-843279608c78"
)

components.iframe(
    POWERBI_URL,
    height=750,
    scrolling=True
)


# ==================================================
# 15. FOOTER
# ==================================================

st.divider()

st.caption(
    "GLOBAL ENERGY INTELLIGENCE | "
    "AI-POWERED ENERGY ANALYTICS PLATFORM"
)
