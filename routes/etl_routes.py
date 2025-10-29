from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
import tempfile
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from services.etl_service import ETLPipeline
import json
from fpdf import FPDF
import glob

PAGE_IMAGE_DIR = "out/converted_images"
PARSED_SECTIONS_DIR = "out/parsed_sections"
MODEL_PATH = "models/yolov12s-doclaynet.pt"

os.environ["GOOGLE_API_KEY"] = "AIzaSyDryzLjl84tQcdIqdXA_RwI2-KXGQMh4M0"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

router = APIRouter()
etl_pipeline = ETLPipeline(MODEL_PATH, PAGE_IMAGE_DIR, PARSED_SECTIONS_DIR)

@router.get("/images/pdf")
def get_images_pdf(file_name: str):
    """
    Returns a PDF file created from all boxed_layout images for the given file name.
    file_name: The base name of the file (without extension) as uploaded/processed.
    """
    # Find all boxed_layout.png images for this file in parsed_sections
    base_name = os.path.splitext(file_name)[0]
    section_dirs = sorted(glob.glob(os.path.join(PARSED_SECTIONS_DIR, f"{base_name}_page_*")))
    boxed_images = []
    for section_dir in section_dirs:
        boxed_img_path = os.path.join(section_dir, "boxed_layout.png")
        if os.path.exists(boxed_img_path):
            boxed_images.append(boxed_img_path)
    if not boxed_images:
        raise HTTPException(status_code=404, detail="No boxed_layout images found for this file.")
    pdf_path = os.path.join(PARSED_SECTIONS_DIR, f"{base_name}_boxed_layouts.pdf")
    pdf = FPDF(unit="pt", format="A4")
    for img_path in boxed_images:
        pdf.add_page()
        pdf.image(img_path, x=0, y=0, w=pdf.w, h=pdf.h)
    pdf.output(pdf_path)
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"{base_name}_boxed_layouts.pdf")

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    start = time.time()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
    finally:
        file.file.close()
    initial_json_output = []
    try:
        image_paths = etl_pipeline.convert_document_to_images(temp_file_path, file.filename)
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            base_image_name = os.path.splitext(os.path.basename(image_path))[0]
            page_specific_output_dir = os.path.join(PARSED_SECTIONS_DIR, base_image_name)
            parsed_content = etl_pipeline.parse_image_layout(image_path, page_specific_output_dir)
            initial_json_output.append({
                "page no": page_num,
                "content": parsed_content
            })

        final_json_output = etl_pipeline.MDocAgent(initial_json_output)

        

        end = time.time()
        print(f"Total processing time: {end - start:.2f} seconds")
        return JSONResponse(content=final_json_output)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/chat")
async def ChatBot(input):
    
    gemini = genai.GenerativeModel("gemini-2.0-flash", safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    })
    
    # message, documentContext, conversationHistory
    input = json.loads(input)
    document_context = input["documentContext"]
    conversation_history = input["conversationHistory"]
    message = input["message"]

    try:

        prompt = f"""You are a "Legal Analyst Assistant." Your persona is that of an expert, precise, and efficient professional assistant. Your primary goal is to help users—whether they are legal professionals or laypeople—instantly find, summarize, and understand any detail within a legal document.
You will be given all necessary information in the [TASK PAYLOAD] section below. This payload contains:
    1. [DOCUMENT_JSON]: The complete, pre-analyzed data from the legal document. This is your one and only source of truth.
    2. [CHAT_HISTORY]: A list of the previous questions and answers in this conversation.
    3. [USER_QUESTION]: The user's new question.

CRITICAL GUARDRAILS (MUST FOLLOW)
    1. NEVER FABRICATE, OPINE, OR PREDICT: Your most important rule is to stick 100% to the provided [DOCUMENT_JSON].
        -DO NOT: Invent information, give your own opinions, suggest actions ("you should..."), or predict legal outcomes ("this clause is unenforceable...").
        -DO: You can and should report the analytical findings from the JSON, such as the analysis section (e.g., "The pre-analysis of this clause notes its fairness as 'Landlord-Favorable'"). You are surfacing the existing analysis, not creating your own.
    2. BE A PRECISE DATA RETRIEVER: Do not just chat. Your main job is to fetch and present the exact information the user asks for.
    3. ADAPT TO THE USER'S EXPERTISE:
        -If the user asks a simple, direct question (e.g., "What does this mean for me?"), prioritize the layman_implication.
        -If the user asks a professional or technical question (e.g., "Summarize the indemnification clause"), prioritize the summary and the analysis list.
    4. IF IT'S NOT IN THE JSON, DON'T ANSWER: If the answer is not in the [DOCUMENT_JSON], you MUST state: "I'm sorry, that specific detail is not available in the analyzed document data."

HOW TO ANSWER (YOUR LOGIC)
    1. First, review the [CHAT_HISTORY] for context.
    2. Next, analyze the [USER_QUESTION] to understand the user's intent and likely expertise.
    3. Then, find the answer only within the [DOCUMENT_JSON]:
        -For "Summarize document": Use document_analysis.abstractive_summary and document_analysis.themes.
        -For "Who/What/Where/How Much": Query key_data_points by entity_type and report the label and attributes.
        -For "What does clause X say?" (Professional): Provide the summary and list all items from the analysis list.
        -For "What does clause X mean?" (Layperson): Provide the layman_implication and add context from the analysis list.
        -For "When/Deadlines": Query key_deadlines and list the event, date, and explanation.
        -For "What does 'Lessor' mean?": Query legal_terminology and provide the explanation.
    4. Finally, formulate a direct, factual answer based on the data you found.
    
[TASK PAYLOAD]:
[DOCUMENT_JSON]:
{json.dumps(document_context)}
[CHAT_HISTORY]:
{json.dumps(conversation_history)}
[USER_QUESTION]:
{message}

"""
        response = gemini.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print(f"An error occurred in chatbot: {e}")
