import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
import io
import json
import time
import requests
import os

# Read Gemini API key from Streamlit secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
def get_ai_recommendation(prompt: str) -> str:
    """
    Calls the Gemini API to get a concise air purifier recommendation.
    """
    if not GEMINI_API_KEY:
        return "API key not set. Please configure your Gemini API key."
    try:
        apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
        headers = { 'Content-Type': 'application/json' }
        # Add instruction for a short response
        concise_prompt = prompt + " Respond in 5-10 short sentences only. and last line should be a summary highlighting the key points., summary should be first and then the detailed explanation , use bullet points for detailed explanation, both summary and detailed explanation should be in markdown format."
        chatHistory = [{"role": "user", "parts": [{"text": concise_prompt}]}]
        payload = {"contents": chatHistory}
        retries = 0
        while retries < 3:
            response = requests.post(apiUrl, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    return "Sorry, the recommendation could not be generated. Please try again."
            retries += 1
            time.sleep(2 ** retries)
        return "Sorry, I couldn't generate a recommendation at this time. Please try again later."
    except Exception as e:
        return f"Sorry, an error occurred while generating the recommendation: {e}"

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

st.title("🇮🇳 AQI Analysis and Air Purifier Recommendation Dashboard")

# --- Data Loading and Cleaning (Caching for performance) ---
@st.cache_data
def load_data():
    """
    Loads and cleans the AQI data by downloading it from a public URL.
    Caches the result to avoid reloading on every rerun.
    """
    # Replace this URL with the raw file URL from your Hugging Face dataset
    DATA_URL = "https://huggingface.co/datasets/manaspateltech/AQI/resolve/main/aqi.csv"
    
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status() # Raise an error for bad status codes
        
        # Use io.StringIO to read the text content as a file
        df = pd.read_csv(io.StringIO(response.text))
        
        # Rename columns in code to match dataset
        df.columns = [col.lower() for col in df.columns]

        # Drop the 'note' and 'unit' columns as they are not needed for the analysis
        df = df.drop(columns=['note', 'unit'])
        
        # Convert 'date' to datetime objects
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        
        # Ensure there are no duplicates
        df.drop_duplicates(inplace=True)
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading data from URL: {e}")
        return pd.DataFrame() # Return an empty DataFrame on error

df = load_data()

page = st.radio("Select Page", ["India Overview", "Statewise AQI", "Areawise AQI"], index=0)

if page == "India Overview":
    # --- Homepage: Overall India Facts ---
    st.header("Overall India Air Quality Index (AQI)")
    st.metric("Average AQI(2022-2025)", f"{df['aqi_value'].mean():.2f}")
    
    # Plot the monthly data of AQI values
    monthly_aqi = df.resample('M', on='date')['aqi_value'].mean().reset_index()
    fig = px.line(monthly_aqi, x='date', y='aqi_value', title="Monthly Average AQI in India")
    st.plotly_chart(fig, use_container_width=True)

    # Top 10 most polluted states
    state_aqi = df.groupby('state')['aqi_value'].mean().reset_index()
    state_aqi = state_aqi.sort_values(by='aqi_value', ascending=False)
    top_10_states = state_aqi.head(10)
    fig = px.bar(top_10_states, x='state', y='aqi_value', title="Top 10 Most Polluted States in India")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 most polluted areas
    area_aqi = df.groupby('area')['aqi_value'].mean().reset_index()
    area_aqi = area_aqi.sort_values(by='aqi_value', ascending=False)
    top_10_areas = area_aqi.head(10)
    fig = px.bar(top_10_areas, x='area', y='aqi_value', title="Top 10 Most Polluted Areas in India")
    st.plotly_chart(fig, use_container_width=True)
    
elif page == "Statewise AQI":
    # --- Statewise AQI Data Page ---
    st.header("Statewise AQI Data")
    
    # --- State Selection ---
    states = sorted(df['state'].dropna().unique())
    state = st.selectbox("Select a State", states, key="state_select")

    state_df = df[df['state'] == state]
    
    if not state_df.empty:
        # --- AQI Over Time Plot ---
        st.subheader(f"AQI Over Time in {state}")
        state_df_monthly = state_df.set_index('date').resample('M')['aqi_value'].mean().reset_index()
        fig = px.line(state_df_monthly, x='date', y='aqi_value', title=f"AQI Trend in {state}")
        st.plotly_chart(fig, use_container_width=True)

        # --- Prominent Pollutants ---
        st.subheader("Prominent Pollutants")

        state_pollutant_df = state_df.dropna(subset=['prominent_pollutants'])
        state_pollutant_df = state_pollutant_df.assign(
            prominent_pollutants=state_pollutant_df['prominent_pollutants'].str.split(',')
        ).explode('prominent_pollutants')
        state_pollutant_df['prominent_pollutants'] = state_pollutant_df['prominent_pollutants'].str.strip()
        pollutants = state_pollutant_df['prominent_pollutants'].value_counts()

        if not pollutants.empty:
            pollutants_df = pollutants.reset_index()
            pollutants_df.columns = ['Pollutant', 'Count']

            chart = alt.Chart(pollutants_df).mark_bar().encode(
                x=alt.X('Count:Q', title="Count"),
                y=alt.Y('Pollutant:N', sort='-x', title="Pollutant")
            )
            st.altair_chart(chart, use_container_width=True)
            
            # --- AI-Based Recommendation Section ---
            st.subheader("AI-Powered Air Purifier Recommendation")
            avg_aqi_state = state_df['aqi_value'].mean()
            top_pollutants_list = pollutants.head().index.tolist()
            top_pollutants_str = ', '.join(top_pollutants_list)
            
            prompt = (
                f"Based on the average AQI of {avg_aqi_state:.2f} and the most prominent pollutants in {state} which are {top_pollutants_str}, "
                "provide a detailed air purifier recommendation. "
                "Specify the necessary filter types (e.g., HEPA, Activated Carbon) and explain why each is needed. "
                "Suggest a suitable CADR (Clean Air Delivery Rate) and a rationale for your choice."
            )

            if st.button(f"Get Recommendation for {state}"):
                with st.spinner('Generating your personalized recommendation...'):
                    recommendation = get_ai_recommendation(prompt)
                    st.markdown(recommendation)
        else:
            st.write("No pollutant data available for this state.")
    else:
        st.write("No data available for the selected state.")
    
elif page == "Areawise AQI":
    # --- Areawise AQI Data Page ---
    st.header("Areawise AQI Data")
    
    # --- Area Selection ---
    areas = sorted(df['area'].dropna().unique())
    area = st.selectbox("Select an Area", areas, key="area_select")

    area_df = df[df['area'] == area]
    avg_aqi_area = area_df['aqi_value'].mean()
    
    if not area_df.empty:
        # --- AQI Over Time Plot ---
        st.subheader(f"AQI Over Time in {area}")
        area_df_monthly = area_df.set_index('date').resample('M')['aqi_value'].mean().reset_index()
        fig = px.line(area_df_monthly, x='date', y='aqi_value', title=f"AQI Trend in {area}")
        st.plotly_chart(fig, use_container_width=True)

        # --- Prominent Pollutants ---
        st.subheader("Prominent Pollutants")

        area_pollutant_df = area_df.dropna(subset=['prominent_pollutants'])
        area_pollutant_df = area_pollutant_df.assign(
            prominent_pollutants=area_pollutant_df['prominent_pollutants'].str.split(',')
        ).explode('prominent_pollutants')
        area_pollutant_df['prominent_pollutants'] = area_pollutant_df['prominent_pollutants'].str.strip()
        pollutants = area_pollutant_df['prominent_pollutants'].value_counts()
        
        if not pollutants.empty:
            pollutants_df = pollutants.reset_index()
            pollutants_df.columns = ['Pollutant', 'Count']

            chart = alt.Chart(pollutants_df).mark_bar().encode(
                x=alt.X('Count:Q', title="Count"),
                y=alt.Y('Pollutant:N', sort='-x', title="Pollutant")
            )
            st.altair_chart(chart, use_container_width=True)

            # --- AI-Based Recommendation Section ---
            st.subheader("AI-Powered Air Purifier Recommendation")
            
            # Create a prompt for the AI model
            top_pollutants_list = pollutants.head().index.tolist()
            top_pollutants_str = ', '.join(top_pollutants_list)

            prompt = (
                f"Based on the average AQI of {avg_aqi_area:.2f} and the most prominent pollutants in {area} which are {top_pollutants_str}, "
                "provide a detailed air purifier recommendation. "
                "Specify the necessary filter types (e.g., HEPA, Activated Carbon) and explain why each is needed. "
                "Suggest a suitable CADR (Clean Air Delivery Rate) and a rationale for your choice."
            )
            
            if st.button(f"Get Recommendation for {area}"):
                with st.spinner('Generating your personalized recommendation...'):
                    recommendation = get_ai_recommendation(prompt)
                    st.markdown(recommendation)

        else:
            st.write("No pollutant data available for this area.")
    else:
        st.write("No data available for the selected area.")
       
