# Importing necessary modules and classes
from pathlib import Path
from langchain.utilities import SQLDatabase
from langchain_community.llms import Replicate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough

# Replication ID for the llama-2-13b model
# Make sure to set REPLICATE_API_TOKEN in your environment
replicate_id = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"

# Initializing Replicate with the specified model and model parameters
llm = Replicate(
    model=replicate_id,
    model_kwargs={"temperature": 0.01, "max_length": 500, "top_p": 1},
)

# Setting up the path for the SQLite database file
db_path = Path(__file__).parent / "kalbe_roster.db"
rel = db_path.relative_to(Path.cwd())
db_string = f"sqlite:///{rel}"

# Creating an SQLDatabase instance from the database URI
db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=0)

# Function to retrieve the schema of the database table
def get_schema(_):
    return db.get_table_info()

# Function to execute SQL queries on the database
def run_query(query):
    return db.run(query)

# Template for prompting user to provide a SQL query based on schema and question
template_query = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""

# Creating a ChatPromptTemplate for the SQL query prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        ("human", template_query),
    ]
)

# Defining a SQL response pipeline
sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# Template for the response to include schema, question, query, and SQL response
template_response = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

# Creating a ChatPromptTemplate for the response prompt
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template_response),
    ]
)

# Defining input types for the prompt
class InputType(BaseModel):
    question: str

# Constructing the pipeline for the entire process
chain = (
    RunnablePassthrough.assign(query=sql_response).with_types(input_type=InputType)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)