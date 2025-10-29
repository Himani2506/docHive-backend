from fastapi import APIRouter, HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
import json
from datetime import date

router = APIRouter(prefix="/gemma", tags=["gemma"])

PROMPT_TEMPLATE = """
You are an intelligent document processor.

Input: a parsed JSON array of document pages (with tags like \"Text\", \"Section-header\", \"Picture\").

Task: Generate a new JSON in the schema below.  
⚠️ Important: Output **only valid JSON**. No explanations. No extra text. No markdown.

Schema:
{{
	\"metadata\": {{
		\"file_name\": \"string\",
		\"file_type\": \"json\",
		\"page_count\": 1,
		\"extraction_date\": \"YYYY-MM-DD\"
	}},
	\"document_info\": {{
		\"title\": \"string\",
		\"document_type\": \"string\",
		\"date_issued\": null,
		\"reference_number\": null
	}},
	\"content\": {{
		\"summary\": \"short summary\",
		\"keywords\": [\"term1\", \"term2\"],
		\"sections_detected\": [],
		\"actions_or_recommendations\": <actions_recommended_in_file>,
		\"attachments_or_figures\": <figures_or_attachments_in_file>
	}},
	\"entities\": {{
		\"models_or_methods\": [],
		\"people\": "people involved",
		\"organizations\": "department mentioned",
		\"locations\": "extracted locations",
		\"standards_or_policies\": "policies listed">
	}},
	\"notes_on_confidence\": {{
		\"summary_confidence\": \"high\",
		\"entities_confidence\": \"medium\",
		\"remarks\": \"short diagnostic\"
	}},
	\"source_file_citation\": \"original_filename\"
}}

---

INPUT CONTENT:
{content}

---
Output only valid JSON.
"""

@router.post("/generate")
async def generate_structured_json(request: Request):
	try:
		input_json = await request.json()
		input_json = input_json[0] if isinstance(input_json, list) else input_json
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

	try:
		prompt = PROMPT_TEMPLATE.format(content=json.dumps(input_json, indent=2))
		model = lms.llm("google/gemma-3-1b:gemma-3-1b-it-qat")
		result = model.respond(prompt)
		result = str(result)
		result = result.replace("```json", "").replace("```", "").strip()
		# model.unload()
		# Try to parse the result as JSON to ensure valid output
		try:
			response_json = json.loads(result)
			return JSONResponse(content=response_json)
		except Exception:
			# If not valid JSON, return as plain text
			return PlainTextResponse(content=json.dumps({"error": "Model did not return valid JSON", "raw": result}))
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
