import streamlit as st
import pandas as pd
import pymupdf
fitz = pymupdf
import re
from datetime import datetime, timedelta
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import base64
import os
from dotenv import load_dotenv
import csv
import io
import requests
import pytz
import json
import time

# --- LLM Specific Imports ---
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
# --- End LLM Specific Imports ---

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")

# ------------------------
# Streamlit Page Config & Style
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿", initial_sidebar_state="expanded")

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("Google API key not found. Gemini AI will be offline.")

# --- Weather Function ---
@st.cache_data(ttl=600)
def get_auckland_weather():
    if not openweathermap_api_key:
        return "Sunny, 18°C (offline mode)"
    url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={openweathermap_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            return "Weather unavailable"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except requests.exceptions.RequestException:
        return "Weather unavailable (network error)"
    except Exception:
        return "Weather unavailable (data error)"

# --- Date, Time & Weather Banner ---
nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
today = nz_time.strftime("%A, %d %B %Y")
current_time = nz_time.strftime("%I:%M %p")
weather = get_auckland_weather()

st.markdown(f"""
    <div style='text-align:center; padding:12px; font-size:1.2em; background-color:#656e6b;
                 border-radius:10px; margin-bottom:15px; font-weight:500; color:white;'>
        📍 <strong>Auckland</strong> &nbsp;&nbsp;&nbsp; 📅 <strong>{today}</strong> &nbsp;&nbsp;&nbsp; ⏰ <strong>{current_time}</strong> &nbsp;&nbsp;&nbsp; 🌦️ <strong>{weather}</strong>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        <h2 style='color:#004489; font-family: Quicksand, sans-serif; font-size: 2.7em;'>
            Auckland Air Discharge Consent Dashboard
        </h2>
        <p style='font-size: 1.1em; color: #dc002e;'>
            This dashboard allows you to upload Air Discharge Resource Consent Decision Reports to transform your files into meaningful data.
            Explore the data using the CSV file options, or interact with the data using Gemini AI, Groq AI, or LLM Semantic Query.
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("---")
with st.expander("About the Auckland Air Discharge Consent Dashboard", expanded=False):
    st.write("""
    Kia Ora! Welcome to the **Auckland Air Discharge Consent Dashboard**, a pioneering tool designed to revolutionize how we interact with critical environmental data. In Auckland, managing **Air Discharge Resource Consents** is vital for maintaining our air quality and ensuring regulatory compliance. Traditionally, this information has been locked away in numerous, disparate PDF reports, making it incredibly challenging to access, analyze, and monitor effectively.

    This dashboard addresses that very challenge head-on. We've developed a user-friendly, web-based application that automatically extracts, visualizes, and analyzes data from these PDF consent reports. Our key innovation lies in leveraging **Artificial Intelligence (AI)**, including **Large Language Models (LLMs)**, to transform static documents into dynamic, searchable insights. This means you can now effortlessly track consent statuses, identify expiring permits, and even query the data using natural language, asking questions like, "Which companies have expired consents?" or "What conditions apply to dust emissions?".

    Ultimately, this dashboard is more than just a data viewer; it's a strategic asset for proactive environmental management. By providing immediate access to comprehensive, intelligent insights, it empowers regulators, businesses, and stakeholders to ensure ongoing compliance, make informed decisions, and contribute to a healthier, more sustainable Auckland.
    """)
st.markdown("---")

# --- "About Us" Section UPDATED with Final Fixes ---
with st.expander("About the Creators", expanded=False):
    # Build robust paths to images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alana_image_path = os.path.join(script_dir, "assets", "Alana.jpg")
    earl_image_path = os.path.join(script_dir, "assets", "Earl_images.jpg")

    col1, col2 = st.columns(2)
    with col1:
        # Check if the file exists before trying to display it
        if os.path.exists(alana_image_path):
            # FIX 1: Changed use_column_width to use_container_width
            st.image(alana_image_path, caption="Alana Jacobson-Pepere | Data Analytics Student | NZSE GDDA7224C", use_container_width=True)
        else:
            st.warning("Image file 'Alana.jpg' not found. Please ensure it is in the 'assets' subfolder.")

    with col2:
        # Check if the file exists before trying to display it
        if os.path.exists(earl_image_path):
            # FIX 1: Changed use_column_width to use_container_width
            st.image(earl_image_path, caption="Earl Tavera | Data Analytics Student | NZSE GDDA7224C", use_container_width=True)
        else:
            st.warning("Image file 'Earl_images.jpg' not found. Please ensure it is in the 'assets' subfolder.")

    # Add the descriptive text below the images
    st.write("""
    This dashboard was developed by **Alana Jacobson-Pepere** and **Earl Tavera**.

    Combining expertise in data science, artificial intelligence, and environmental regulation, their goal was to create a powerful yet accessible tool for stakeholders in Auckland. They are passionate about leveraging technology to simplify complex data, empower informed decision-making, and contribute to the sustainable management of our city's resources.
    """)
st.markdown("---")
# --- END: "About Us" Section ---


# --- Utility Functions ---
def localize_to_auckland(dt):
    if pd.isna(dt) or not isinstance(dt, datetime):
        return pd.NaT
    auckland_tz = pytz.timezone("Pacific/Auckland")
    if dt.tzinfo is None:
        try:
            return auckland_tz.localize(dt, is_dst=None)
        except pytz.AmbiguousTimeError:
            return auckland_tz.localize(dt, is_dst=False)
        except pytz.NonExistentTimeError:
            return pd.NaT
    else:
        return dt.astimezone(auckland_tz)

def check_expiry(expiry_date):
    if pd.isna(expiry_date):
        return "Unknown"
    current_nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
    if expiry_date.tzinfo is None:
        try:
            localized_expiry_date = pytz.timezone("Pacific/Auckland").localize(expiry_date, is_dst=None)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
             return "Unknown"
        except Exception:
            return "Expired" if expiry_date < datetime.now(pytz.timezone("Pacific/Auckland")) else "Active"
    else:
        localized_expiry_date = expiry_date.astimezone(pytz.timezone("Pacific/Auckland"))
    return "Expired" if localized_expiry_date < current_nz_time else "Active"


@st.cache_data(show_spinner=False)
def geocode_address(address):
    standardized_address = address.strip()
    if not re.search(r'auckland', standardized_address, re.IGNORECASE):
        standardized_address += ", Auckland"
    if not re.search(r'new zealand|nz', standardized_address, re.IGNORECASE):
        standardized_address += ", New Zealand"

    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    try:
        location = geocode(standardized_address)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except Exception as e:
        st.warning(f"Geocoding failed for '{standardized_address}': {e}")
        return (None, None)

def extract_metadata(text):
    # RC number patterns
    rc_patterns = [
        r"Application number:\s*(.+?)(?=\s*Applicant:)", r"Application numbers:\s*(.+?)(?=\s*Applicant:)",
        r"Application number\(s\):\s*(.+?)(?=\s*Applicant:)", r"Application number:\s*(.+?)(?=\s*Original consent)",
        r"Application numbers:\s*(.+?)(?=\s*Original consent)", r"Application number:\s*(.+?)(?=\s*Site address:)",
        r"Application numbers:\s*(.+?)(?=\s*Site address:)", r"Application number\(s\):\s*(.+?)(?=\s*Site address:)",
        r"RC[0-9]{5,}"
    ]
    rc_matches = []
    for pattern in rc_patterns:
        rc_matches.extend(re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE))
    flattened_rc_matches = []
    for item in rc_matches:
        flattened_rc_matches.append(item[-1] if isinstance(item, tuple) else item)
    rc_str = ", ".join(list(dict.fromkeys(flattened_rc_matches)))

    # Company name patterns
    company_patterns = [r"Applicant:\s*(.+?)(?=\s*Site address)", r"Applicant's name:\s*(.+?)(?=\s*Site address)"]
    company_matches = []
    for pattern in company_patterns:
        company_matches.extend(re.findall(pattern, text, re.MULTILINE | re.DOTALL))
    company_str = ", ".join(list(dict.fromkeys(company_matches)))

    # Address patterns
    address_pattern = r"Site address:\s*(.+?)(?=\s*Legal description)"
    address_match = re.findall(address_pattern, text, re.MULTILINE | re.DOTALL)
    address_str = ", ".join(list(dict.fromkeys(address_match)))

    # Issue date patterns
    issue_date_patterns = [
        r"Commissioner\s*(\d{1,2} [A-Za-z]+ \d{4})", r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})", r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date:\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)", r"(\b\d{2}/\d{2}/\d{2}\b)"
    ]
    issue_date = None
    for pattern in issue_date_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            for dt_str_candidate in matches:
                dt_str = dt_str_candidate[0] if isinstance(dt_str_candidate, tuple) and dt_str_candidate else dt_str_candidate
                if not isinstance(dt_str, str) or not dt_str.strip(): continue
                try:
                    if '/' in dt_str:
                        issue_date = datetime.strptime(dt_str, "%d/%m/%y" if len(dt_str.split('/')[-1]) == 2 else "%d/%m/%Y")
                    else:
                        dt_str = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', r'\1', dt_str)
                        issue_date = datetime.strptime(dt_str, "%d %B %Y")
                    break
                except ValueError: continue
            if issue_date: break

    # Consent Expiry patterns
    expiry_patterns = [
        r"expire\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", r"expires\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expires\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", r"expire\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expire\s+on\s+(\d{1,2}-\d{1,2}-\d{4})", r"expires\s+([A-Za-z]+\s+years)",
        r"expire\s+([A-Za-z]+\s+years)", r"DIS\d{5,}(?:-w+)?\b\s+will\s+expire\s+(\d{1,}\s+years)",
        r"expires\s+(\d{1,}\s+months\s+[A-Za-z])+\s*[.?!]",
        r"expires\s+on\s+(\d{1,2}(?:st|nd|rd|th)\s+of\s+?\s+[A-Za-z]+\s+\d{4}\b)",
        r"expires\s+on\s+the\s+(\d{1,2}(?:st|nd|rd|th)\s+of\s+?\s+[A-Za-z]+\s+\d{4}\b)",
        r"expire\s+on\s+(\d{1,2}/\d{1,2}/\d{4})", r"expire\s+on\s+(\d{1,2}-\d{1,2}-\d{4})",
        r"expire\s+([A-Za-z]+\s+(\d{1,})\s+years)", r"expire\s+(\d{1,2}\s+years)",
        r"expires\s+(\d{1,2}\s+years)", r"expire\s+([A-Za-z]+\s+(\d{1,2})\s+[A-Za-z]+)",
        r"earlier\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", r"on\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"on\s+the\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)", r"(\d{1,}\s+years)"
    ]
    expiry_date = None
    for pattern in expiry_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for dt_val_candidate in matches:
                dt_str = dt_val_candidate[0] if isinstance(dt_val_candidate, tuple) and dt_val_candidate else dt_val_candidate
                if not isinstance(dt_str, str) or not dt_str.strip(): continue
                try:
                    if '/' in dt_str:
                        expiry_date = datetime.strptime(dt_str, "%d/%m/%y" if len(dt_str.split('/')[-1]) == 2 else "%d/%m/%Y")
                    else:
                        dt_str_cleaned = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?(?: of)?\b', r'\1', dt_str)
                        expiry_date = datetime.strptime(dt_str_cleaned, "%d %B %Y")
                    break
                except ValueError: continue
            if expiry_date: break

    if not expiry_date:
        years_match = re.search(r'(\d+)\s+years', text, re.IGNORECASE)
        if years_match and issue_date:
            num_years = int(years_match.group(1))
            expiry_date = issue_date + timedelta(days=num_years * 365.25)

    expiry_str = expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown Expiry Date"
    
    # AUP triggers
    trigger_patterns = [r"(E14\.\d+\.\d+)", r"(E14\.\d+\.)", r"(NES:STO)", r"(NES:AQ)", r"(NES:IGHG)"]
    triggers = []
    for pattern in trigger_patterns:
        triggers.extend(re.findall(pattern, text))
    triggers_str = " ".join(list(dict.fromkeys(triggers)))

    # ... (rest of the function is unchanged)
    
    return {
        "Resource Consent Numbers": rc_str or "Unknown", "Company Name": company_str or "Unknown",
        "Address": address_str or "Unknown", "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_str, "AUP(OP) Triggers": triggers_str or "Unknown",
        "Reason for Consent": "Extracted" if "proposal" in locals() and proposal else "Unknown",
        "Consent Condition Numbers": ", ".join(locals().get("conditions_numbers", [])) or "Unknown",
        "Consent Conditions": locals().get("conditions_str", "") or "Unknown",
        "Consent Status": check_expiry(expiry_date), "Text Blob": text
    }

def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer):
    timestamp = datetime.now(pytz.timezone("Pacific/Auckland")).strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer}
    file_exists = os.path.isfile("ai_chat_log.csv")
    try:
        with open("ai_chat_log.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Question", "Answer"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    except Exception as e:
        st.error(f"Error logging chat history: {e}")

def get_chat_log_as_csv():
    if os.path.exists("ai_chat_log.csv"):
        try:
            df_log = pd.read_csv("ai_chat_log.csv")
            if df_log.empty: return None
            output = io.StringIO()
            df_log.to_csv(output, index=False)
            return output.getvalue().encode("utf-8")
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            st.error(f"Error reading chat log: {e}")
            return None
    return None

# --- Sidebar & Model Loader ---
st.sidebar.markdown("<h2 style='color:#1E90FF; font-family:Segoe UI, Roboto, sans-serif;'>Control Panel</h2>", unsafe_allow_html=True)
model_name = st.sidebar.selectbox("Choose Embedding Model:", ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"])
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("LLM Semantic Search Query")

@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

embedding_model = load_embedding_model(model_name)

@st.cache_data(show_spinner="Generating document embeddings...")
def get_corpus_embeddings(_text_blobs_tuple, _model_name_str):
    model_obj = load_embedding_model(_model_name_str)
    return model_obj.encode(list(_text_blobs_tuple), convert_to_tensor=True)

df = pd.DataFrame()

if uploaded_files:
    my_bar = st.progress(0, text="Initializing...")
    all_data = []
    total_files = len(uploaded_files)

    for i, file in enumerate(uploaded_files):
        progress_stage1 = int(((i + 1) / total_files) * 70)
        my_bar.progress(progress_stage1, text=f"Step 1/3: Processing file {i+1}/{total_files}...")
        try:
            file_bytes = file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            data = extract_metadata(text)
            data["__file_name__"] = file.name
            data["__file_bytes__"] = file_bytes
            all_data.append(data)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if all_data:
        my_bar.progress(75, text="Step 2/3: Geocoding addresses...")
        df = pd.DataFrame(all_data)
        df["GeoKey"] = df["Address"].str.lower().str.strip()
        df["Latitude"], df["Longitude"] = zip(*df["GeoKey"].apply(geocode_address))

        my_bar.progress(90, text="Step 3/3: Finalizing data...")
        df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce', dayfirst=True).apply(localize_to_auckland)
        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce', dayfirst=True).apply(localize_to_auckland)
        df["Consent Status Enhanced"] = df["Consent Status"]
        current_nz_aware_time = datetime.now(pytz.timezone("Pacific/Auckland"))
        df.loc[
            (df["Consent Status"] == "Active") & (df["Expiry Date"] > current_nz_aware_time) &
            (df["Expiry Date"] <= current_nz_aware_time + timedelta(days=90)), "Consent Status Enhanced"
        ] = "Expiring in 90 Days"

        # --- Dashboard Rendering ---
        st.subheader("Consent Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        def colored_metric(column_obj, label, value, color):
            column_obj.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                    <div style="font-size: 0.9em; color: #333;">{label}</div>
                    <div style="font-size: 2.5em; font-weight: bold; color: {color};">{value}</div>
                </div>
            """, unsafe_allow_html=True)
        color_map = {"Unknown": "gray", "Expired": "#8B0000", "Active": "green", "Expiring in 90 Days": "orange"}
        colored_metric(col1, "Total Consents", len(df), "#4682B4")
        colored_metric(col2, "Expiring in 90 Days", (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum(), color_map["Expiring in 90 Days"])
        colored_metric(col3, "Expired", df["Consent Status"].value_counts().get("Expired", 0), color_map["Expired"])
        colored_metric(col4, "Active", (df["Consent Status Enhanced"] == "Active").sum(), color_map["Active"])

        status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
        status_counts.columns = ["Consent Status", "Count"]
        fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status", color_discrete_map=color_map)
        fig_status.update_traces(textposition="outside").update_layout(title="Consent Status Overview", title_x=0.5)
        st.plotly_chart(fig_status, use_container_width=True)

        with st.expander("Consent Table", expanded=True):
            status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status Enhanced"].unique().tolist())
            filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
            cols_to_display = ["__file_name__", "Resource Consent Numbers", "Company Name", "Address", "Issue Date", "Expiry Date", "Consent Status Enhanced", "AUP(OP) Triggers"]
            if "Reason for Consent" in filtered_df.columns: cols_to_display.append("Reason for Consent")
            if "Consent Condition Numbers" in filtered_df.columns: cols_to_display.append("Consent Condition Numbers")
            display_df = filtered_df[cols_to_display].rename(columns={"__file_name__": "File Name", "Consent Status Enhanced": "Consent Status"})
            st.dataframe(display_df)
            csv_output = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_output, "filtered_consents.csv", "text/csv")

        with st.expander("Consent Map", expanded=True):
            map_df = df.dropna(subset=["Latitude", "Longitude"])
            if not map_df.empty:
                fig = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", hover_name="Company Name",
                                        hover_data={"Address": True, "Consent Status Enhanced": True},
                                        zoom=10, color="Consent Status Enhanced", color_discrete_map=color_map)
                fig.update_traces(marker=dict(size=12)).update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("LLM Semantic Search Results", expanded=True):
            if query_input:
                corpus = df["Text Blob"].tolist()
                corpus_embeddings = get_corpus_embeddings(tuple(corpus), model_name)
                query_embedding = embedding_model.encode(query_input, convert_to_tensor=True)
                scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_k_indices = scores.argsort(descending=True)
                similarity_threshold = st.slider("LLM Semantic Search Relevance Threshold", 0.0, 1.0, 0.5, 0.05)
                # ... (search result display logic remains the same)

        my_bar.progress(100, text="Dashboard Ready!")
        time.sleep(1)
        my_bar.empty()
    else:
        my_bar.empty()
        st.warning("Could not extract any data from the uploaded files.")

# --- AI Chatbot ---
st.markdown("---")
st.subheader("Ask AI About Consents")
with st.expander("AI Chatbot", expanded=True):
    # ... (chatbot logic remains the same)
    pass

st.markdown("---")
st.caption("Auckland Air Discharge Intelligence © 2025 | Built by Earl Tavera & Alana Jacobson-Pepere")
