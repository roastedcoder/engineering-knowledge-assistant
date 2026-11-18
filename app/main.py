from fastapi import FastAPI
app = FastAPI(
  title="Engineering Knowledge Assistant",
  description="A local RAG system with citations and evaluation",
  version="0.1.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "engineering-knowledge-assistant"
    }