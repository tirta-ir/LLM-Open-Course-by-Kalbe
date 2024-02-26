# Import necessary modules
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from fastapi import FastAPI
from langserve import add_routes

# Initialize a HuggingFaceHub model with the specified repository ID and model parameters
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0})

# Create a new FastAPI application
test = FastAPI(
    title       = "Demo app for KDU LLM",
    version     = "1.5",
    description = "Demo for LangServe",
)

# Add routes to the FastAPI application for the base model
add_routes(
    test,
    llm,
    path="/base",
)

# Create a prompt template
prompt = PromptTemplate.from_template("Explain {topic} in a paragraph")

# Add routes to the FastAPI application for the prompted model
add_routes(
    test,
    prompt | llm,
    path = "/demo",
)

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(test, host = "localhost", port = 8000)
