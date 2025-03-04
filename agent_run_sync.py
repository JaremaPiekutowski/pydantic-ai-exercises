'''
This example shows how to run an agent synchronously.
'''

from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o")

# Run the agent synchronously using the `run_sync` method
result_sync = agent.run_sync(user_prompt="Who is Gemma in `Severance` series?")

print(result_sync.data)
