import requests
from ..config.config import get_settings

settings = get_settings()

def create_conversational_ai(
    first_message: str,
    system_prompt: str,
    agent_name: str = "my_agent",
    llm_model: str = "claude-3-5-sonnet",
    temperature: float = 0.7,
    limit_token_usage: float = -1,
    tool_url: str = "https://hack-sales.onrender.com/chat",
    tool_name: str = "sales_agent",
    tool_description: str = "This is the tool you can use for getting a good and accurate answer to any queries specific to the product in discussion",
    tool_method: str = "POST",
    ):

    try:
        agent_id = get_agent_id(agent_name)
        return agent_id
    except Exception as e:
        print(e)
        
    payload = {
        "name": tool_name,
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": system_prompt,
                    "llm": llm_model,
                    "temperature": temperature,
                    "max_tokens": limit_token_usage,
                    "tools": [
                        {
                            "type": "webhook",
                            "name": tool_name,
                            "description": tool_description,
                            "api_schema": {
                                "url": tool_url,
                                "method": tool_method,
                                "request_body_schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "This is the user's prompt query",
                                        }
                                    },
                                    "required": ["query"],
                                }
                            }
                        },
                        {
                            "type": "system",
                            "name": "end_call",
                            "description": ""
                        }
                    ]

                },
                "first_message": first_message,
                "dynamic_variables": {
                    "dynamic_variable_placeholders": {
                        "user_name": "karthik"
                    }
                },

            }
        }
    }
    url = "https://api.elevenlabs.io/v1/convai/agents/create"
    api_key = settings.ELEVEN_LABS_API_KEY
    headers = {
        "Authorization": f"xi-api-key: {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()
        return output["agent_id"]
    else:
        raise Exception(f"Failed to create agent: {response.status_code} {response.text}")
    
def get_agent_id(agent_name: str):
    url = f"https://api.elevenlabs.io/v1/convai/agents"
    api_key = settings.ELEVEN_LABS_API_KEY
    headers = {
        "xi-api-key": f"{api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        output = response.json()
        for agent in output["agents"]:
            if agent["name"] == agent_name:
                return agent["agent_id"]
    else:
        raise Exception(f"Failed to get agent id: {response.status_code} {response.text}")
    
    raise Exception(f"Agent not found: {agent_name}")
