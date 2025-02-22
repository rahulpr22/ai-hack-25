from fastapi import FastAPI, Request
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import google.generativeai as genai
from google.generativeai import types
import requests
import os
from dotenv import load_dotenv
import asyncio
import time
load_dotenv()

# Configure API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)

async def vectorize_query(query: str):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return embeddings.embed_query(query)

async def get_context(query: str):
    t1 = time.time()
    query_vector = await vectorize_query(query)
    results = pinecone_index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    context = "\n".join([res.metadata.get("text", "") for res in results.matches])
    t2 = time.time()
    print(f"Time taken to get context: {t2 - t1} seconds", flush=True)
    return context

class GeminiTask():
    async def should_search(self, query: str, context: str):
        t1 = time.time()
        model = genai.GenerativeModel("gemini-pro")
        prompt = f"""
        Given the query: "{query}"
        And the retrieved context: "{context}"
        Determine whether a web search is necessary. Respond with "YES" or "NO".
        """
        response = model.generate_content(prompt)
        t2 = time.time()
        print(f"Time taken to determine if search is needed: {t2 - t1} seconds", flush=True)
        return "YES" in response.text.upper()

    async def respond(self, query: str, context: str, search_results: str = ""):
        t1 = time.time()
        model = genai.GenerativeModel("gemini-pro")
        enriched_input = f"""
        Query: "{query}"
        Context from database: "{context}"
        Web search results: "{search_results}"
        Provide a very concise response but a minimum of 30 words.
        """
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=100,
            temperature=0.7
        )
        response = model.generate_content(enriched_input, generation_config=generation_config)
        t2 = time.time()
        print(f"Time taken to generate response: {t2 - t1} seconds", flush=True)
        return response.text

class PerplexitySearchTask():
    async def search(self, query: str):
        t1 = time.time()
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}"
        }
        data = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can search the web for information. Be precise and concise."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],

        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            try:
                json_response = response.json()
                t2 = time.time()
                print(f"Time taken to get search results: {t2 - t1} seconds", flush=True)
                return json_response.get("choices", [{}])[0].get("message", {}).get("content", "No results found")
            except requests.exceptions.JSONDecodeError:
                print(f"Failed to decode JSON. Response content: {response.text}", flush=True)
                return "Error: Unable to get search results"
                
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}", flush=True)
            return "Error: Failed to get search results"

class Agent():
    def __init__(self):
        self._gemini_agent = GeminiTask()
        self._search_agent = PerplexitySearchTask()

    async def chat(self, query: str):
        context = await get_context(query)
        
        # Create tasks from coroutines
        should_search_task = asyncio.create_task(self._gemini_agent.should_search(query, context))
        initial_response_task = asyncio.create_task(self._gemini_agent.respond(query, context, ""))
        
        needs_search, initial_response = await asyncio.gather(should_search_task, initial_response_task)
        
        if needs_search:
            print("Searching the web for more information...", flush=True)
            search_results = await self._search_agent.search(query)  # Use await here
            return await self._gemini_agent.respond(query, context, search_results)
        
        return initial_response

# FastAPI setup
app = FastAPI()
agent = Agent()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")
    response = await agent.chat(query)
    return response

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent:app", host="0.0.0.0", port=10000, reload=True)
