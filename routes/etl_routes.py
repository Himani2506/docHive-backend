from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
import tempfile
import time
from services.etl_service import ETLPipeline

from fpdf import FPDF
import glob

PAGE_IMAGE_DIR = "out/converted_images"
PARSED_SECTIONS_DIR = "out/parsed_sections"
MODEL_PATH = "models/yolov12s-doclaynet.pt"

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
    final_json_output = []
    try:
        image_paths = etl_pipeline.convert_document_to_images(temp_file_path, file.filename)
        for i, image_path in enumerate(image_paths):
            page_num = i + 1
            base_image_name = os.path.splitext(os.path.basename(image_path))[0]
            page_specific_output_dir = os.path.join(PARSED_SECTIONS_DIR, base_image_name)
            parsed_content = etl_pipeline.parse_image_layout(image_path, page_specific_output_dir)
            final_json_output.append({
                "page no": page_num,
                "content": parsed_content
            })
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
