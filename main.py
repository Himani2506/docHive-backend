from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.etl_routes import router as etl_router
from routes.neo4j_routes import router as neo4j_router
from routes.gemma_routes import router as gemma_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this list for specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(etl_router)
app.include_router(neo4j_router)
app.include_router(gemma_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Document Management API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)