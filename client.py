import requests
from langserve import RemoteRunnable

user_input = "Learn how Amazon Q for Developers help in your overall Software Development Lifecycle. Besides generating code, you can now engage in conversational coding, debug and optimise your code with Amazon Q, and scan through your code for any security vulnerabilities."
server_url = "http://localhost:8000/generate"

# using requests
input_dict = {"input": {"text": user_input}}
response = requests.post(f"{server_url}/invoke", json=input_dict)
print(f"from requests: {response.text}")

# using langserve client
langserve_client = RemoteRunnable(server_url)
input_dict = {"text": user_input}
response = langserve_client.invoke(input=input_dict)

print(f"from langserve_client: {response}")

