import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from app.core.bias_engine import analyze_bias
from app.core.explain import explain_bias
from app.utils.validators import validate_dataframe, validate_target_column, validate_sensitive_column
from app.services.report_service import export_report_to_txt
from app.services.llm_service import generate_gemini_explanation

st.set_page_config(page_title="FairSight AI", layout="wide")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

def main():
    st.title("FairSight AI")
    st.subheader("AI System for Detecting and Explaining Bias")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(df.head())

            columns = df.columns.tolist()

            col1, col2 = st.columns(2)
            with col1:
                target_col = st.selectbox("Target Column (Outcome):", columns)
            with col2:
                sensitive_col = st.selectbox("Sensitive Attribute (e.g., Gender, Race):", columns)

            if st.button("Analyze Bias"):
                # Clear previous results if new analysis is triggered
                if 'results' in st.session_state:
                    del st.session_state['results']
                if 'explanation' in st.session_state:
                    del st.session_state['explanation']
                if 'gemini_explanation' in st.session_state:
                    del st.session_state['gemini_explanation']

                if target_col == sensitive_col:
                    st.error("Target column and Sensitive Attribute must be different.")
                    return

                # Validations
                valid_df, df_msg = validate_dataframe(df)
                if not valid_df:
                    st.error(df_msg)
                    return

                valid_target, target_msg = validate_target_column(df, target_col)
                if not valid_target:
                    st.warning(target_msg)
                    return
                    
                valid_sensitive, sensitive_msg = validate_sensitive_column(df, sensitive_col)
                if not valid_sensitive:
                    st.warning(sensitive_msg)
                    return

                with st.spinner("Analyzing bias..."):
                    results = analyze_bias(df, target_col, sensitive_col)
                    explanation = explain_bias(results, sensitive_col, target_col)
                    
                    st.session_state['results'] = results
                    st.session_state['explanation'] = explanation
                    st.session_state['target_col'] = target_col
                    st.session_state['sensitive_col'] = sensitive_col
                    st.session_state['columns'] = columns

            if 'results' in st.session_state:
                results = st.session_state['results']
                explanation = st.session_state['explanation']
                t_col = st.session_state['target_col']
                s_col = st.session_state['sensitive_col']
                
                st.markdown("---")
                st.header("Analysis Results")

                risk = results.get("risk_level", "Unknown")
                if risk == "High Risk Bias":
                    st.error(f"**Status: {risk}**")
                elif risk == "Moderate Bias":
                    st.warning(f"**Status: {risk}**")
                else:
                    st.success(f"**Status: {risk}**")

                col_details, col_chart = st.columns([1, 1])

                with col_details:
                    st.markdown(explanation.get("summary", ""), unsafe_allow_html=True)
                    st.markdown(explanation.get("metrics", ""), unsafe_allow_html=True)
                    
                    st.markdown("### Why This Matters")
                    st.markdown(explanation.get("why_it_matters", ""))
                    
                    st.markdown("### Suggested Actions")
                    st.markdown(explanation.get("actions", ""), unsafe_allow_html=True)

                    st.markdown("### AI Explanation (Powered by Gemini)")
                    if 'gemini_explanation' not in st.session_state:
                        with st.spinner("Generating AI explanation..."):
                            gemini_exp = generate_gemini_explanation(results["group_means"], results["bias_score"])
                            st.session_state['gemini_explanation'] = gemini_exp
                    
                    gemini_text = st.session_state['gemini_explanation']
                    if "unavailable" in gemini_text.lower():
                        st.warning(gemini_text)
                    else:
                        st.info(gemini_text)

                with col_chart:
                    st.markdown("### Group-wise Comparisons")
                    
                    group_means = results["group_means"]
                    groups = [str(g) for g in group_means.keys()]
                    means = list(group_means.values())
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    if len(means) > 0:
                        max_val = max(means)
                        min_val = min(means)
                        colors = []
                        for m in means:
                            if m == max_val:
                                colors.append('#1f77b4') # Blue
                            elif m == min_val:
                                colors.append('#d62728') # Red
                            else:
                                colors.append('#7f7f7f') # Gray
                    else:
                        colors = '#4C72B0'
                        
                    bars = ax.bar(groups, means, color=colors)
                    
                    for bar in bars:
                        yval = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=9)
                        
                    ax.set_xlabel(s_col)
                    ax.set_ylabel(f'Mean {t_col}')
                    ax.tick_params(axis='x', rotation=45)
                    fig.tight_layout()
                    st.pyplot(fig)

                # Export Report
                st.markdown("---")
                with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', suffix='.txt') as tmp:
                    export_report_to_txt(
                        tmp.name, 
                        results, 
                        explanation, 
                        t_col, 
                        s_col, 
                        st.session_state['columns']
                    )
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as file:
                    st.download_button(
                        label="Export Report",
                        data=file,
                        file_name="bias_report.txt",
                        mime="text/plain"
                    )
                os.unlink(tmp_path)

        except Exception as e:
            st.error(f"Error Loading File: {str(e)}")

if __name__ == "__main__":
    main()
