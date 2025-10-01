from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

# Load API key from environment (.env for local, Render ENV VAR in production)
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# --- Homepage Route ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the EasyBites Ingredient Scanner API"}

# --- Ingredient Analysis Route ---
@app.post("/analyze-ingredients/")
async def analyze_ingredients(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()

        # Load Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro")

        # Force JSON-only response for specific ingredients
        prompt = """
        You are a food ingredient detection system.
        Identify only the following ingredients in the uploaded image:
        Chicken meat, Lady's Finger (Okra), Tomato, Potato, Carrots, Eggplant,
        String Beans (Batong), Onion, Chili Pepper, Vegetable Pear / Chayote (Sayote).

        Return the result in pure JSON format only, like this:
        {
          "ingredients_detected": ["Tomato", "Onion", "Carrot"],
          "shelf_life_days": {
              "Chicken meat": 2,
              "Lady's Finger (Okra)": 5,
              "Tomato": 5,
              "Potato": 30,
              "Carrots": 14,
              "Eggplant": 7,
              "String Beans (Batong)": 5,
              "Onion": 30,
              "Chili Pepper": 14,
              "Vegetable Pear / Chayote (Sayote)": 10
          }
        }
        """

        # Call Gemini API
        response = model.generate_content(
            [prompt, {"mime_type": file.content_type, "data": contents}],
            generation_config={"response_mime_type": "application/json"}
        )

        # Parse Gemini response safely
        return JSONResponse(content=json.loads(response.text))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
