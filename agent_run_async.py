'''
This example shows how to run an agent asynchronously.
'''
import asyncio

from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o")


# Define an async function to run the agent
async def main():
    result = await agent.run("Who is Helena Eagan in `Severance` series?")
    print(result.data)


if __name__ == "__main__":
    # Use asyncio.run to run the async function
    asyncio.run(main())
