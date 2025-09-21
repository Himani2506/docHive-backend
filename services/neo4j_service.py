from neo4j import GraphDatabase
from typing import List

class Neo4jService:
    def __init__(self, uri: str, auth: tuple, db_name: str):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.db_name = db_name

    def get_documents_by_keyword(self, keyword: str) -> List[str]:
        with self.driver.session(database=self.db_name) as session:
            result = session.run("""
                MATCH (d:Document)-[:HAS_KEYWORD]->(k:Keyword {name: $keyword})
                RETURN d.title AS document_title
            """, keyword=keyword)
            docs = [record["document_title"] for record in result]
            return docs
