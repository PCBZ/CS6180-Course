import streamlit as st
import pandas as pd
import spacy
import os
import gdown
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import hnswlib

# Load models
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading spaCy model 'en_core_web_lg'...")
    from spacy.cli import download
    download("en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):  
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()

ingredient_list = load_ingredient_data()

# Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(ingredient_vectors), ef_construction=200, M=16)
    index.add_items(ingredient_vectors)
    index.set_ef(50)
    return index

annoy_index = build_annoy_index()

#  Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

def direct_search_alternatives(ingredient):

    input_vec = nlp(ingredient.lower()).vector
    
    similarities = []

    for i, vec in enumerate(ingredient_vectors):
        if filtered_ingredient_list[i].lower() == ingredient.lower():
            continue  # Skip the same ingredient
        sim = cosine_similarity(input_vec, vec)
        similarities.append((filtered_ingredient_list[i], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return [ingredient_name for ingredient_name, _ in similarities[:3]]

#  Annoy Search (Fixed for Correct Cosine Similarity)
def annoy_search_alternatives(ingredient):
    input_vec = nlp(ingredient.lower()).vector.reshape(1, -1).astype(np.float32)

    indices, _ = annoy_index.knn_query(input_vec, k=4)
    alternatives = [
        filtered_ingredient_list[i] for i in indices[0]
        if filtered_ingredient_list[i].lower().strip() != ingredient.lower().strip()
    ]

    return alternatives[:3]  # Return top 3 alternatives

#  Generate Recipe
def generate_recipe(ingredients, cuisine, strategy="Greedy", temperature=1.0, topK=5, topP=0.95, beamNumber=1, prompt_control="Structured"):
    input_text = (
        f"Ingredients: {', '.join(ingredients.split(', '))}\n"
        f"Cuisine: {cuisine}\n"
        f"Let's create a dish inspired by {cuisine} cuisine with these ingredients. Here are the preparation and cooking instructions:"
    )

    control = ""
    if prompt_control == "Structured":
        control = """
            Generate a recipe with the following format:\n
            Title: [recipe name]\n
            Ingredients: [list each ingredient with amount]\n
            Steps: [step-by-step instructions]
        """
    elif prompt_control == "Concise":
        control = "Generate a concise recipe with just the essential steps and ingredients."
    elif prompt_control == "Creative":
        control = "Generate a creative and unique recipe that uses the ingredients in an unexpected way."
    else:
        control = ""
    
    input_text += "\n" + control

    with torch.no_grad():

        if strategy == "Greedy":
            outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                                    max_length=250, num_return_sequences=1,
                                    repetition_penalty=1.2)
        elif strategy == "Beam Search":
            outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                                    max_length=250, num_return_sequences=1,
                                    repetition_penalty=1.2, num_beams=beamNumber)
        else:
            outputs = model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], 
                                max_length=250, num_return_sequences=1,
                                repetition_penalty=1.2, do_sample=True, top_k=topK, top_p=topP, temperature=temperature)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()

def estimate_nutrition(recipe_text):
    # Placeholder for nutrition estimation logic
    
    prompt = f"""Food: chicken salad with rice
Nutrition:
- Calories: 420 kcal
- Protein: 30g
- Fat: 15g
- Carbs: 35g
- Fiber: 3g

Food: pasta with tomato sauce
Nutrition:
- Calories: 380 kcal
- Protein: 15g
- Fat: 12g
- Carbs: 52g
- Fiber: 3g

Food: {recipe_text[:80]}
Nutrition:
"""

    with torch.no_grad():
        outputs = model.generate(tokenizer(prompt, return_tensors="pt")["input_ids"],
                                num_return_sequences=1,
                                max_new_tokens=40,
                                repetition_penalty=1.2)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")
ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])

strategy = st.radio("Select Recipe Generation Strategy:", ["Greedy", "Beam Search", "Sampling"])

prompt_control = st.selectbox("Prompt Control:", ["Structured", "Concise", "Creative"])

temperature = st.select_slider("Temperature:", options=[0.5, 1.0, 2.0], value=1.0)

if strategy == "Sampling":
    topK = st.select_slider("Top-K:", options=[5, 50], value=50)
    topP = st.select_slider("Top-P:", options=[0.7, 0.95], value=0.95)
    beamNumber = 1
elif strategy == "Beam Search":
    beamNumber = st.select_slider("Beam Number:", options=[1, 5], value=5)
    topK = 5
    topP = 0.95
else:
    topK = 5
    topP = 0.95
    beamNumber = 1

if st.button("Generate Recipe", use_container_width=True) and ingredients:
    st.session_state["recipe"] = generate_recipe(ingredients, cuisine, strategy=strategy, temperature=temperature, topK=topK, topP=topP, beamNumber=beamNumber, prompt_control=prompt_control)

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    st.text_area("Recipe:", st.session_state["recipe"], height=200)

    st.download_button(label="📂 Save Recipe", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")

    #  Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives
        }
        start_time = time.time()
        alternatives = search_methods[search_method](ingredient_to_replace)
        print(f"Search completed in {time.time() - start_time:.2f} seconds.")
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")

if "recipe" in st.session_state:
    st.markdown("---")
    st.markdown("## 🧮 Estimated Nutritional Information")
    if st.button("📊 Estimate Nutrition", use_container_width=True):
        st.session_state["nutrition"] = estimate_nutrition(st.session_state["recipe"])
    
    if "nutrition" in st.session_state:
        st.text_area("Nutrition per Serving:", st.session_state["nutrition"], height=500)
