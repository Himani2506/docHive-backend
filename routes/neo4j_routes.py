from fastapi import APIRouter
from services.neo4j_service import Neo4jService

URI = "neo4j+s://a0cc0423.databases.neo4j.io"
AUTH = ("a0cc0423", "DzxfJ08-afo4oevlwh3HRyIvn7uDS0_w_QV5zh72AW4")
DB_NAME = "a0cc0423"

router = APIRouter()
neo4j_service = Neo4jService(URI, AUTH, DB_NAME)

@router.get("/search/{keyword}")
def search(keyword: str):
    documents = neo4j_service.get_documents_by_keyword(keyword)
    return documents
