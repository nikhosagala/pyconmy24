from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langserve import add_routes
from pydantic import BaseModel, Field

app = FastAPI()
load_dotenv()

SYSTEM_PROMPT = """You're an expert in event industry, tasked with answering question about event.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
You should remove all html from your response"""

DEFAULT_PROMPT = "Generate a comprehensive and informative answer of 80 words or less for the given question."
OPEN_AI_MODEL = "gpt-4o"

generate_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template=SYSTEM_PROMPT),
        SystemMessagePromptTemplate.from_template(template="{prompt}"),
        HumanMessagePromptTemplate.from_template(template="{text}")
    ]
)

# temperature docs
# https://platform.openai.com/docs/guides/text-generation/how-should-i-set-the-temperature-parameter
chat_open_ai_model = ChatOpenAI(temperature=0.6, model_name=OPEN_AI_MODEL).configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="Temperature",
        description="Temperature used for the LLM"
    )
).configurable_fields(
    model_name=ConfigurableField(
        id="model_name",
        name="Model Name",
        description="Model name used for the LLM"
    )
)

pycon_chain = generate_prompt | chat_open_ai_model | StrOutputParser()


class GenerateInput(BaseModel):
    """Input for generate API"""

    text: str = Field(
        ...,
        description="Input or text from user"
    )
    prompt: str = Field(
        default=DEFAULT_PROMPT,
        description="Prompt from client, if not pass will use DEFAULT_PROMPT"
    )


allowed_endpoints = ("input_schema", "output_schema", "invoke", "batch", "stream", "stream_log", "stream_events")
add_routes(app, pycon_chain, path="/generate", input_type=GenerateInput, enabled_endpoints=allowed_endpoints)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
