import os
from google import genai

def generate_gemini_explanation(group_means: dict, bias_score: float) -> str:
    """
    Generates an AI explanation of bias using Google Gemini.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    try:
        if project_id and project_id.strip() != "":
            client = genai.Client(vertexai=True, project=project_id, location="us-central1")
        else:
            client = genai.Client(vertexai=True, location="us-central1")
        
        prompt = f"""
        Analyze the following bias detection results for a dataset:
        - Group Means: {group_means}
        - Bias Score: {bias_score:.3f}

        Please provide a concise analysis (max ~120 words) with the following elements:
        1. Clearly identify which group is advantaged and which is disadvantaged based on the means.
        2. Explain what the bias score indicates in this context.
        3. Describe the potential real-world impact of this bias.
        4. Provide 2-3 actionable suggestions for mitigating this bias.
        
        Output only the requested explanation, avoiding any markdown formatting if possible.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        # Instead of crashing, return a fallback message or log the error
        return "AI explanation unavailable. Showing system explanation only."
