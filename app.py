from dotenv import load_dotenv
load_dotenv()

"""
FitOS — Flask Backend (Fixed for TensorFlow / Keras)
Fixes:
  ✓ Removed ONNX dependency — uses your existing Keras .h5 model
  ✓ Imports model.py correctly (your original file, now fixed)
  ✓ Anthropic SDK for AI coach
  ✓ All routes: /api/predict, /api/plan, /api/coach, /api/progress
  ✓ Render/Railway ready
"""

import os
import json
import numpy as np
from datetime import datetime, date
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Import your model.py ──────────────────────────────────
# Make sure model.py is in the SAME folder as app.py
try:
    from model import predict_food
    MODEL_AVAILABLE = True
except Exception as e:
    print(f"Model not loaded: {e}")
    MODEL_AVAILABLE = False
    def predict_food(image):
        return "unknown", 0.0

# ── Try Groq SDK (for AI coach) ─────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("anthropic package not installed — coach will use rule-based fallback")

app = Flask(__name__)
CORS(app, origins=os.getenv("ALLOWED_ORIGINS", "*"))

# ── Config ────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PORT              = int(os.getenv("PORT", 5000))

# ── Load nutrition data ───────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
nutrition_path = os.path.join(BASE_DIR, "nutrition.json")

NUTRITION_DATA = {}
if os.path.exists(nutrition_path):
    with open(nutrition_path) as f:
        NUTRITION_DATA = json.load(f)
    print(f"Nutrition data loaded: {len(NUTRITION_DATA)} foods")
else:
    print("WARNING: nutrition.json not found. Nutrition values will be 0.")

# ── In-memory progress store ──────────────────────────────
user_progress = {}

def get_user(user_id: str) -> dict:
    if user_id not in user_progress:
        user_progress[user_id] = {
            "weights": [],
            "streak": 0,
            "last_checkin": None,
        }
    return user_progress[user_id]

# ── Calorie calculator (Mifflin-St Jeor) ─────────────────
ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2, "light": 1.375, "moderate": 1.55,
    "active": 1.725, "very_active": 1.9,
}

def calculate_tdee(weight_kg, height_cm, age, sex, activity):
    if sex == "male":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return bmr * ACTIVITY_MULTIPLIERS.get(activity, 1.375)

def get_healthier_option(food: str) -> str:
    food_lower = food.lower().replace(" ", "_")
    swaps = {
        "biryani":       "Try veg biryani with brown rice (~280 kcal)",
        "dosa":          "Try idli + sambar — lighter at ~80 kcal each",
        "pizza":         "Try whole wheat thin-crust pizza (~200 kcal/slice)",
        "samosa":        "Try baked samosa (~120 kcal vs 250 kcal fried)",
        "vada":          "Try steamed idli instead (~60 kcal vs 150 kcal)",
        "chole_bhature": "Try rajma chawal — more protein, less oil",
        "burger":        "Try a grilled chicken wrap (~350 kcal vs 550 kcal)",
        "french_fries":  "Try roasted sweet potato (3x the nutrients)",
        "jalebi":        "Try a small portion of fruit chaat instead",
        "noodles":       "Try whole wheat noodles with extra vegetables",
    }
    return swaps.get(food_lower, f"Enjoy {food} in moderation and pair with vegetables.")


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":    "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "nutrition_foods": len(NUTRITION_DATA),
    })


# ── 1. Food Prediction ────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file  = request.files["image"]
        image = Image.open(file.stream)

        food, confidence = predict_food(image)

        # Look up nutrition (try exact match, then lowercase, then stripped)
        nutrition = (
            NUTRITION_DATA.get(food) or
            NUTRITION_DATA.get(food.lower()) or
            NUTRITION_DATA.get(food.lower().replace("_", " ")) or
            {}
        )

        return jsonify({
            "food":       food,
            "confidence": round(confidence * 100, 1),
            "calories":   nutrition.get("calories", 0),
            "carbs":      nutrition.get("carbs", 0),
            "protein":    nutrition.get("protein", 0),
            "fat":        nutrition.get("fat", 0),
            "fiber":      nutrition.get("fiber", 0),
            "suggestion": get_healthier_option(food),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 2. Plan Generation ────────────────────────────────────
@app.route("/api/plan", methods=["POST"])
def generate_plan():
    try:
        data = request.json
        weight   = float(data["weight"])
        height   = float(data["height"])
        age      = int(data.get("age", 25))
        sex      = data.get("sex", "male")
        goal     = data["goal"]
        target   = float(data["targetWeight"])
        duration = int(data["duration"])
        activity = data.get("activity", "light")
        diet_pref = data.get("diet", "non-veg")

        tdee     = calculate_tdee(weight, height, age, sex, activity)
        calories = tdee
        if goal == "lose":
            calories -= 500
        elif goal == "gain":
            calories += 300

        bmi     = round(weight / ((height / 100) ** 2), 1)
        protein = round(weight * 1.8)
        weekly  = round(abs(weight - target) / duration, 2)

        diet_plans = {
            "vegetarian": {
                "breakfast":     "Oats + banana + nuts (350 kcal)",
                "mid_morning":   "Greek yogurt + fruits",
                "lunch":         "Brown rice + dal + sabzi + curd (550 kcal)",
                "evening_snack": "Roasted chana + green tea",
                "dinner":        "Chapati + paneer curry + salad (450 kcal)",
                "post_workout":  "Protein shake + banana",
            },
            "vegan": {
                "breakfast":     "Smoothie bowl — banana, oats, almond milk, seeds",
                "mid_morning":   "Apple + almond butter",
                "lunch":         "Quinoa + rajma + stir-fried veggies",
                "evening_snack": "Roasted chickpeas",
                "dinner":        "Tofu stir-fry + brown rice + salad",
                "post_workout":  "Soy protein shake + dates",
            },
            "non-veg": {
                "breakfast":     "Egg whites (3) + whole wheat toast + fruits",
                "mid_morning":   "Greek yogurt + nuts",
                "lunch":         "Brown rice + chicken curry + salad (600 kcal)",
                "evening_snack": "Boiled eggs (2) + green tea",
                "dinner":        "Grilled fish / chicken + chapati + sabzi",
                "post_workout":  "Whey protein shake + banana",
            },
            "eggetarian": {
                "breakfast":     "Veggie omelette (3 eggs) + brown bread",
                "mid_morning":   "Fruit bowl + nuts",
                "lunch":         "Brown rice + dal + egg curry + salad",
                "evening_snack": "Boiled egg + sprouts chaat",
                "dinner":        "Chapati + paneer / egg bhurji + sabzi",
                "post_workout":  "Egg white shake + banana",
            },
        }

        workout_plans = {
            "lose": [
                {"day": "Monday",    "focus": "Cardio + Core",     "exercises": ["30 min brisk walk/run", "Plank 3×60s", "Crunches 3×20", "Mountain climbers 3×15"]},
                {"day": "Tuesday",   "focus": "Full Body Strength", "exercises": ["Squats 3×15", "Push-ups 3×12", "Dumbbell rows 3×12", "Lunges 3×12"]},
                {"day": "Wednesday", "focus": "HIIT",               "exercises": ["20 min HIIT (30s on/30s off)", "Burpees 3×10", "Jump squats 3×15", "High knees 3×20"]},
                {"day": "Thursday",  "focus": "Upper Body",         "exercises": ["Shoulder press 3×12", "Bicep curls 3×12", "Tricep dips 3×15", "Push-ups 3×15"]},
                {"day": "Friday",    "focus": "Lower Body + Yoga",  "exercises": ["Deadlifts 3×10", "Glute bridges 3×20", "Calf raises 3×20", "20 min yoga flow"]},
                {"day": "Saturday",  "focus": "Active Cardio",      "exercises": ["45 min cycling or swimming", "Core circuit 3 rounds"]},
                {"day": "Sunday",    "focus": "Rest & Recovery",    "exercises": ["Light walk 20 min", "Full body stretching", "Foam rolling"]},
            ],
            "gain": [
                {"day": "Monday",    "focus": "Chest + Triceps",   "exercises": ["Bench press 4×8", "Incline dumbbell press 3×10", "Cable flyes 3×12", "Skull crushers 3×12"]},
                {"day": "Tuesday",   "focus": "Back + Biceps",     "exercises": ["Deadlift 4×6", "Pull-ups 4×max", "Barbell rows 3×10", "Hammer curls 3×12"]},
                {"day": "Wednesday", "focus": "Legs",              "exercises": ["Squats 4×8", "Leg press 3×12", "Romanian deadlift 3×10", "Calf raises 4×20"]},
                {"day": "Thursday",  "focus": "Shoulders",         "exercises": ["Military press 4×8", "Lateral raises 3×15", "Front raises 3×12", "Face pulls 3×15"]},
                {"day": "Friday",    "focus": "Arms + Core",       "exercises": ["EZ bar curls 3×12", "Tricep pushdown 3×12", "Planks 3×60s", "Ab wheel 3×15"]},
                {"day": "Saturday",  "focus": "Full Body Power",   "exercises": ["Power cleans 3×5", "Box jumps 3×8", "Farmer walks 3×30m"]},
                {"day": "Sunday",    "focus": "Rest & Recovery",   "exercises": ["Mobility work", "Foam rolling", "Light yoga"]},
            ],
            "maintain": [
                {"day": "Monday",    "focus": "Cardio",            "exercises": ["30 min jog", "Core work 3 rounds"]},
                {"day": "Tuesday",   "focus": "Strength",          "exercises": ["Full body compound lifts", "3×10 each"]},
                {"day": "Wednesday", "focus": "Yoga / Flexibility","exercises": ["45 min yoga flow", "Breathing exercises"]},
                {"day": "Thursday",  "focus": "Cardio + Core",     "exercises": ["Cycling 30 min", "Plank variations", "Crunches"]},
                {"day": "Friday",    "focus": "Strength",          "exercises": ["Upper + lower split", "3×12 each"]},
                {"day": "Saturday",  "focus": "Active Recreation", "exercises": ["Sport / swimming / hiking"]},
                {"day": "Sunday",    "focus": "Rest",              "exercises": ["Stretching", "Light walk"]},
            ],
        }

        return jsonify({
            "calories":   round(calories),
            "tdee":       round(tdee),
            "bmi":        bmi,
            "protein":    protein,
            "weeklyGoal": f"{'Lose' if goal=='lose' else 'Gain' if goal=='gain' else 'Maintain'} {weekly} kg/week",
            "diet":       diet_plans.get(diet_pref, diet_plans["non-veg"]),
            "workout":    workout_plans.get(goal, workout_plans["maintain"]),
            "tips": [
                "Drink 2.5–3L of water daily. Most hunger is actually thirst.",
                "Sleep 7–9 hours — growth hormone peaks during deep sleep.",
                f"Track your food for the first 2 weeks to calibrate your {round(calories)} kcal target.",
                "Progressive overload: add 2.5kg or 1 rep every week to keep growing.",
            ],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 3. AI Coach ───────────────────────────────────────────
@app.route("/api/coach", methods=["POST"])
def coach():
    try:
        data    = request.json
        message = data.get("message", "")
        history = data.get("history", [])

        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI fitness coach and nutritionist specializing in "
                    "Indian lifestyle and cuisine. You are warm, motivating, and practical. "
                    "Keep responses under 3 sentences. Suggest Indian-context foods where relevant."
                )
            }
        ]
        for m in (history or [])[-10:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=300,
        )
        reply = response.choices[0].message.content
        reply = reply.replace("**", "").replace("##", "").replace("# ", "")
        return jsonify({"reply": reply})

    except Exception as e:
        print(f"COACH ERROR: {e}")
        return jsonify({"reply": _rule_based_coach(data.get("message", "")), "error": str(e)})


def _rule_based_coach(msg: str) -> str:
    msg = msg.lower()
    if any(w in msg for w in ["tired", "exhausted", "low energy"]):
        return "Rest is part of the process. Have a banana + peanut butter, sleep 8 hours, and come back stronger. 💪"
    if any(w in msg for w in ["plateau", "stuck", "not losing"]):
        return "Plateaus mean your body adapted — increase workout intensity or reduce calories by 100 kcal. 🔥"
    if any(w in msg for w in ["motivation", "give up", "quit"]):
        return "Every champion felt like quitting. The difference is they didn't. You're one workout away from feeling great. 🏆"
    if any(w in msg for w in ["cheat", "ate bad", "junk"]):
        return "One meal doesn't break progress. Get back on track with your next meal — no guilt needed. ✅"
    if any(w in msg for w in ["before workout", "pre workout", "pre-workout"]):
        return "Eat 45–60 min before: banana + 2 eggs or oats + peanut butter. Carbs for fuel, protein for muscle. 🍌"
    return "Stay consistent — results come to those who show up every day, even imperfectly. 💯"


# ── 4. Progress Tracking ──────────────────────────────────
@app.route("/api/progress/log", methods=["POST"])
def log_progress():
    try:
        data    = request.json
        user_id = data.get("userId", "default")
        weight  = float(data["weight"])
        user    = get_user(user_id)
        today   = str(date.today())

        user["weights"].append({"date": today, "weight": weight})

        yesterday = str(date.fromordinal(date.today().toordinal() - 1))
        last = user.get("last_checkin")
        if last == yesterday:
            user["streak"] += 1
        elif last != today:
            user["streak"] = 1
        user["last_checkin"] = today

        return jsonify({
            "message": "Logged",
            "streak":  user["streak"],
            "weights": user["weights"][-30:],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/progress/<user_id>", methods=["GET"])
def get_progress(user_id):
    user    = get_user(user_id)
    weights = user["weights"]
    change  = (weights[-1]["weight"] - weights[0]["weight"]) if len(weights) >= 2 else 0
    return jsonify({
        "streak":      user["streak"],
        "weights":     weights[-30:],
        "totalChange": round(change, 1),
        "entries":     len(weights),
    })


if __name__ == "__main__":
    app.run(
        host  = "0.0.0.0",
        port  = PORT,
        debug = os.getenv("DEBUG", "false").lower() == "true",
    )