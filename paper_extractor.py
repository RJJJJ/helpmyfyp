import os
import PyPDF2
import google.generativeai as genai
import pandas as pd
import json
import time
from tqdm import tqdm

# ================= SETTINGS =================
# Place all downloaded research PDF files into this folder
PDF_FOLDER = './papers/' 
OUTPUT_EXCEL = 'medical_knowledge_base.xlsx'

# API KEY Input
API_KEY = "AIzaSyASRhpjYC5F6_kb35c0Tpph_sfCeEOL3R8" 

# ================= CORE LOGIC =================

def extract_text_from_pdf(pdf_path):
    """Read text content from PDF"""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        # To save tokens, we usually only read the first 10 pages (typically contains abstract, results, discussion)
        # Gemini 1.5 Flash has a large context window, full text is also possible if needed.
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def analyze_paper_with_gemini(text, filename):
    """Analyze paper with AI to extract structured data"""
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview')

    # Ask AI to output specific JSON format for consistency
    prompt = f"""
    You are a medical research assistant. Analyze the following academic paper text.
    Extract quantitative data regarding Nailfold Capillaroscopy.
    
    Target Data points:
    1. Normal Capillary Density (loops/mm)
    2. Normal Apical Width/Diameter (um)
    3. Definitions of Abnormalities (e.g., Giant loop size, Ectasia size)
    4. Disease correlations (e.g., "Giant loops are associated with 80% risk of Scleroderma")
    
    Output strictly in JSON format with this structure (return a list of objects):
    [
        {{
            "Category": "Normal Density" or "Dimension" or "Disease Risk" or "Definition",
            "Parameter": "e.g., Mean Density",
            "Value": "e.g., 9",
            "Unit": "loops/mm",
            "Range": "e.g., 7-12",
            "Context": "e.g., Healthy adults",
            "Source_Text": "Quote the sentence from text",
            "Author_Year": "Extract Author and Year from text if possible"
        }}
    ]
    
    If no relevant data is found, return an empty list [].
    
    Paper Filename: {filename}
    Paper Text Content (truncated):
    {text[:50000]} 
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean response string to ensure valid JSON
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        # Append filename to each record
        for item in data:
            item['Filename'] = filename
        return data
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

def main():
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"Please create folder '{PDF_FOLDER}' and add PDF files!")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in folder.")
        return

    all_extracted_data = []

    print(f"🔍 Found {len(pdf_files)} papers, starting data mining...")
    
    for pdf_file in tqdm(pdf_files):
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        
        # 1. Extract Text
        text = extract_text_from_pdf(pdf_path)
        if not text: continue
        
        # 2. AI Analysis
        extracted_info = analyze_paper_with_gemini(text, pdf_file)
        
        if extracted_info:
            all_extracted_data.extend(extracted_info)
        
        # Avoid API Rate Limit
        time.sleep(2)

    # 3. Export to Excel
    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data)
        # Adjust column order
        cols = ['Category', 'Parameter', 'Value', 'Range', 'Unit', 'Context', 'Disease_Risk', 'Author_Year', 'Filename', 'Source_Text']
        # Ensure all columns exist
        for col in cols:
            if col not in df.columns:
                df[col] = ""
                
        df.to_excel(OUTPUT_EXCEL, index=False)
        print(f"\n✅ Success! Data exported to: {OUTPUT_EXCEL}")
        print(f"Extracted {len(all_extracted_data)} key data points in total.")
    else:
        print("❌ No data extracted.")

if __name__ == "__main__":
    main()