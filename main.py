from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

@app.post("/analyze-ingredients/")
async def analyze_ingredients(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()

        # Load Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Force JSON-only response
        prompt = """
        You are a food ingredient detection system.
        Identify all ingredients present in the uploaded image.
        Return the result in pure JSON format only, like this:
        {
          "ingredients": ["Tomato", "Onion", "Carrot"],
          "shelf_life_days": {
              "Tomato": 5,
              "Onion": 30,
              "Carrot": 14
          }
        }
        """

        # Call Gemini API
        response = model.generate_content(
            [prompt, {"mime_type": file.content_type, "data": contents}],
            generation_config={"response_mime_type": "application/json"}
        )

        return JSONResponse(content=response.text)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
