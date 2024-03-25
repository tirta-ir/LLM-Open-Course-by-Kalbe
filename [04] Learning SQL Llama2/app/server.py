from fastapi import FastAPI                         # Importing FastAPI framework
from fastapi.responses import RedirectResponse      # Importing RedirectResponse for redirection
from langserve import add_routes                    # Importing function to add routes
from sql_llama2 import chain as sql_llama2_chain    # Importing the SQL llama2 chain

# Creating a FastAPI application instance
app = FastAPI()

# Defining a route for GET requests to root URL
@app.get("/")
async def redirect_root_to_docs():      # Defining an asynchronous function for redirection
    return RedirectResponse("/docs")    # Redirecting root URL to /docs


# Edit this to add the chain you want to add
add_routes(app, sql_llama2_chain, path="/sql-llama2")  # Adding routes for SQL llama2 chain

if __name__ == "__main__":      # Checking if the script is being run directly
    import uvicorn              # Importing uvicorn for running the application

    uvicorn.run(app, host="localhost", port=8000)  # Running the FastAPI application
