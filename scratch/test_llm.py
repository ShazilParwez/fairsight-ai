import os
from google import genai

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "fairsight-ai")
print("Project:", project_id)
try:
    client = genai.Client(vertexai=True, project=project_id, location="us-central1")
    prompt = "hello"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    print("Response:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
