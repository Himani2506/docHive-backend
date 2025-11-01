from typing import List
import os
import cv2
import json
import tempfile
import subprocess
import fitz
import shutil
from PIL import Image
import google.generativeai as genai
import pytesseract
from ultralytics import YOLO
from fastapi import HTTPException
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field


class ETLPipeline:

    VISUAL_LABELS = ['Picture', 'Table', 'Formula']

    def __init__(self, model_path: str, page_image_dir: str, parsed_sections_dir: str):
        self.model = YOLO(model_path)
        self.page_image_dir = page_image_dir
        self.parsed_sections_dir = parsed_sections_dir
        # self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        self.gemini = genai.GenerativeModel("gemini-2.0-flash")
        os.makedirs(self.page_image_dir, exist_ok=True)
        os.makedirs(self.parsed_sections_dir, exist_ok=True)
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDNWii5DoKOjnBdci9BOc-92pb0HtyyDpM"
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    def _render_pdf_to_all_images(self, pdf_path: str, base_filename: str, dpi: int) -> List[str]:
        image_paths = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    zoom_factor = dpi / 72.0
                    matrix = fitz.Matrix(zoom_factor, zoom_factor)
                    pixmap = page.get_pixmap(matrix=matrix)
                    output_image_path = os.path.join(self.page_image_dir, f"{base_filename}_page_{page_num + 1}.jpg")
                    pixmap.save(output_image_path)
                    image_paths.append(output_image_path)
            return image_paths
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error rendering PDF: {e}")

    def convert_document_to_images(self, input_path: str, original_filename: str, dpi: int = 300) -> List[str]:
        file_extension = os.path.splitext(input_path)[1].lower()
        base_filename = os.path.splitext(original_filename)[0]
        image_paths = []
        if file_extension == '.pdf':
            image_paths = self._render_pdf_to_all_images(input_path, base_filename, dpi)
        elif file_extension in ['.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls']:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    subprocess.run([
                        "libreoffice", "--headless", "--convert-to", "pdf:writer_pdf_Export",
                        "--outdir", temp_dir, input_path,
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    intermediate_pdf = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_path))[0] + ".pdf")
                    if not os.path.exists(intermediate_pdf):
                        raise FileNotFoundError("LibreOffice failed to create the intermediate PDF.")
                    image_paths = self._render_pdf_to_all_images(intermediate_pdf, base_filename, dpi)
            except FileNotFoundError:
                raise HTTPException(status_code=500, detail="LibreOffice not found. Please ensure it's installed.")
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"LibreOffice conversion failed: {e.stderr.decode()}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
        return image_paths

    def parse_image_layout(self, source_image_path: str, output_dir: str) -> List[dict]:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        page_content = []
        try:
            source_img = cv2.imread(source_image_path)
            if source_img is None:
                return []
            results = self.model(source_img)
            result = results[0]
            class_names = result.names
            class_counts = {}
            detections = []
            # For drawing bounding boxes
            boxed_img = source_img.copy()
            for box in result.boxes:
                detections.append({'box': box, 'y1': int(box.xyxy[0][1])})
            sorted_detections = sorted(detections, key=lambda d: d['y1'])
            for item in sorted_detections:
                box = item['box']
                class_id = int(box.cls[0])
                label = class_names[class_id]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                # Draw bounding box on boxed_img
                color = (255, 0, 255)  # Green for all boxes, can be customized
                cv2.rectangle(boxed_img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(boxed_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cropped_image = source_img[y1:y2, x1:x2]
                content_data = ""
                if label in self.VISUAL_LABELS:
                    count = class_counts.get(label, 0)
                    filename = f"{label}_{count}.png"
                    save_path = os.path.join(output_dir, filename)
                    class_counts[label] = count + 1
                    cv2.imwrite(save_path, cropped_image)
                    content_data = save_path
                else:
                    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                    text = pytesseract.image_to_string(pil_image, lang='eng')
                    content_data = text.strip()
                page_content.append({
                    "tag": label,
                    "content": content_data
                })
            # Save the image with bounding boxes
            boxed_img_path = os.path.join(output_dir, "boxed_layout.png")
            cv2.imwrite(boxed_img_path, boxed_img)
        except Exception as e:
            pass
        return page_content
    
    def MDocAgent(self, input_json):

        prompt = """
You are an expert-level legal analyst and data extraction engine. Your task is to meticulously read a raw, page-based JSON document, deeply understand its contents, and then convert it into a single, structured, analytical JSON output.
You must follow these steps:
    1. Read the entire [INPUT DOCUMENT] to understand its context, purpose, and all its details.
    2. Populate the [TARGET SCHEMA] provided below. The structure of your output must follow this schema exactly.
    3. Carefully study the [FEW-SHOT EXAMPLES] to see how different types of legal documents are transformed into the same final schema.

[TARGET SCHEMA] - Your output MUST conform to this exact json schema :
{
  "document_analysis": {
    "predicted_document_type": "",
    "abstractive_summary": "",
    "keywords": [],
    "themes": []
  },
  "key_data_points": [
    {
      "entity_id": "",
      "entity_type": "",
      "label": "",
      "attributes": [
        { "key": "", "value": "" }
      ]
    }
  ],
  "key_clauses": [
    {
      "clause_type": "",
      "summary": "",
      "layman_implication": "",
      "affected_entity_ids": [],
      "analysis": [
        {
          "type": "",
          "value": "",
          "details": ""
        }
      ]
    }
  ],
  "key_deadlines": [
    {
      "date": "",
      "event": "",
      "explanation": ""
    }
  ],
  "legal_terminology": [
    {
      "term": "",
      "explanation": ""
    }
  ]
}

[CORE INSTRUCTIONS]

1. document_analysis:
-predicted_document_type: Classify the document (e.g., "Commercial Lease", "Last Will", "NDA").
-abstractive_summary: Write a 2-3 sentence high-level summary.
-keywords: List key literal terms and names (e.g., "Landlord Corp.", "5-year term").
-themes: List key conceptual topics (e.g., "Risk Allocation", "Financial Obligation", "Confidentiality").

2. key_data_points ("The Facts"):
-Extract all key "nouns": Parties, Locations, Amounts, and Dates.
-Create a unique entity_id for each (e.g., "ent_001").
-entity_type must be a general category (e.g., "Party", "Location", "Amount", "Date").
-attributes must be a list of {"key": "...", "value": "..."} pairs describing that entity.

3. key_clauses ("The Rules"):
-Extract the main obligations, risks, permissions, and directives (e.g., "Indemnification", "Bequest").
-Write a clear summary and a layman_implication (what it means for a normal person).
-In affected_entity_ids, list the entity_ids from key_data_points that this clause affects. This is how you link "Rules" to "Facts".
-analysis must be a list of {"type": "...", "value": "...", "details": "..."} pairs. The type can change based on the document (e.g., "Risk Level", "Fairness", "Clarity", "Condition").

4. key_deadlines: Extract all time-based events, dates, or time periods.

5. legal_terminology: Identify general legal jargon (e.g., "Indemnify," "Testator") and provide a simple explanation.


[FEW-SHOT EXAMPLES]

--- EXAMPLE 1: COMMERCIAL LEASE ---

[INPUT 1](JSON):

[
  {
    "page no": 1,
    "content": [
      {"tag": "heading", "content": "Commercial Lease Agreement"},
      {"tag": "content", "content": "This agreement, made 2026-01-01, is between Apex Properties ('Lessor') and Beta Innovations Inc. ('Lessee')."},
      {"tag": "content", "content": "Lessee shall pay a monthly rent of $5,000, due on the first day of each month."},
      {"tag": "section", "content": "4. Indemnification"},
      {"tag": "content", "content": "The Lessee agrees to indemnify and hold harmless the Lessor from any and all claims arising from the Lessee's use of the property."}
    ]
  }
]

[OUTPUT 1](JSON)

{
  "document_analysis": {
    "predicted_document_type": "Commercial Lease Agreement",
    "abstractive_summary": "A commercial lease agreement effective Jan 1, 2026, between Apex Properties (Lessor) and Beta Innovations Inc. (Lessee). The rent is $5,000 monthly.",
    "keywords": ["Lease Agreement", "Apex Properties", "Beta Innovations Inc.", "Indemnification", "$5,000"],
    "themes": ["Financial Obligation", "Risk Allocation", "Liability"]
  },
  "key_data_points": [
    {
      "entity_id": "ent_001",
      "entity_type": "Party",
      "label": "Apex Properties",
      "attributes": [{"key": "role", "value": "Lessor"}]
    },
    {
      "entity_id": "ent_002",
      "entity_type": "Party",
      "label": "Beta Innovations Inc.",
      "attributes": [{"key": "role", "value": "Lessee"}]
    },
    {
      "entity_id": "ent_003",
      "entity_type": "Date",
      "label": "2026-01-01",
      "attributes": [{"key": "label", "value": "Effective Date"}]
    },
    {
      "entity_id": "ent_004",
      "entity_type": "Amount",
      "label": "5000",
      "attributes": [
        {"key": "label", "value": "Monthly Rent"},
        {"key": "currency", "value": "USD"},
        {"key": "frequency", "value": "Monthly"}
      ]
    }
  ],
  "key_clauses": [
    {
      "clause_type": "Indemnification",
      "summary": "The Lessee (Beta Innovations Inc.) must protect the Lessor (Apex Properties) from all legal claims related to their use of the property.",
      "layman_implication": "If someone sues the landlord because of something you did, you have to pay all the legal costs and damages, not the landlord.",
      "affected_entity_ids": ["ent_001", "ent_002"],
      "analysis": [
        {
          "type": "Risk Level",
          "value": "High (for Tenant)",
          "details": "This is a one-sided clause that transfers all risk."
        },
        {
          "type": "Fairness",
          "value": "Lessor-Favorable",
          "details": "Common in commercial leases but harsh for the tenant."
        }
      ]
    }
  ],
  "key_deadlines": [
    {
      "date": "first day of each month",
      "event": "Rent Payment Due",
      "explanation": "This is the day rent must be paid to avoid being late."
    }
  ],
  "legal_terminology": [
    {
      "term": "Lessor",
      "explanation": "The formal, legal word for the 'Landlord'."
    },
    {
      "term": "Lessee",
      "explanation": "The formal, legal word for the 'Tenant'."
    },
    {
      "term": "Indemnify",
      "explanation": "A legal promise to pay for another person's losses or legal costs."
    }
  ]
}


--- EXAMPLE 2: LAST WILL & TESTAMENT ---

[INPUT 2](JSON)

[
  {
    "page no": 1,
    "content": [
      {"tag": "heading", "content": "Last Will and Testament of Jane Doe"},
      {"tag": "content", "content": "I, Jane Doe, the Testator, being of sound mind, do hereby declare this my last will."},
      {"tag": "section", "content": "1. Executor"},
      {"tag": "content", "content": "I appoint my son, John Doe, as the Executor of my will. If he is unable to serve, I appoint Maria Garcia."},
      {"tag": "section", "content": "2. Specific Bequest"},
      {"tag": "content", "content": "I give my 1965 Ford Mustang to my son, John Doe."}
    ]
  }
]

[OUTPUT 2](JSON)

{
  "document_analysis": {
    "predicted_document_type": "Last Will and Testament",
    "abstractive_summary": "The last will of Jane Doe, appointing John Doe as Executor and making a specific bequest of a 1965 Ford Mustang to him.",
    "keywords": ["Last Will", "Jane Doe", "John Doe", "Executor", "Bequest", "Maria Garcia"],
    "themes": ["Estate Distribution", "Fiduciary Appointment", "Asset Transfer"]
  },
  "key_data_points": [
    {
      "entity_id": "ent_w01",
      "entity_type": "Party",
      "label": "Jane Doe",
      "attributes": [{"key": "role", "value": "Testator"}]
    },
    {
      "entity_id": "ent_w02",
      "entity_type": "Party",
      "label": "John Doe",
      "attributes": [
        {"key": "role", "value": "Executor"},
        {"key": "role", "value": "Beneficiary"}
      ]
    },
    {
      "entity_id": "ent_w03",
      "entity_type": "Party",
      "label": "Maria Garcia",
      "attributes": [{"key": "role", "value": "Alternate Executor"}]
    },
    {
      "entity_id": "ent_w04",
      "entity_type": "Asset",
      "label": "1965 Ford Mustang",
      "attributes": [{"key": "label", "value": "Specific Gift"}]
    }
  ],
  "key_clauses": [
    {
      "clause_type": "Fiduciary Appointment (Executor)",
      "summary": "Appoints John Doe as the Executor to manage the estate. Maria Garcia is the backup.",
      "layman_implication": "John Doe is in charge of carrying out the will's instructions. If he can't, Maria Garcia takes over.",
      "affected_entity_ids": ["ent_w01", "ent_w02", "ent_w03"],
      "analysis": [
        {
          "type": "Clarity",
          "value": "High",
          "details": "The primary and alternate appointments are clear."
        }
      ]
    },
    {
      "clause_type": "Specific Bequest",
      "summary": "Grants the 1965 Ford Mustang specifically to John Doe.",
      "layman_implication": "John Doe gets the car. This gift is handled before all other general assets are divided.",
      "affected_entity_ids": ["ent_w02", "ent_w04"],
      "analysis": [
        {
          "type": "Condition",
          "value": "Outright",
          "details": "The gift is given with no strings attached."
        }
      ]
    }
  ],
  "key_deadlines": [],
  "legal_terminology": [
    {
      "term": "Testator",
      "explanation": "The legal term for the person who has made the will."
    },
    {
      "term": "Executor",
      "explanation": "The person appointed to 'execute' or carry out the instructions in the will."
    },
    {
      "term": "Bequest",
      "explanation": "A gift of personal property or assets made in a will."
    }
  ]
}

[YOUR TASK]
Now, process the following [INPUT DOCUMENT] and generate the complete analytical JSON output.

[INPUT DOCUMENT]:
""" + str(input_json)
        
        print(1)
        try:
          response = self.gemini.generate_content(prompt)
        except Exception as e:
            print(e)
        print(2)
        response = response.text.strip()[7:-3]
        json_output = json.loads(response)

        return json_output
