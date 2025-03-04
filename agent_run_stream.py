'''
Stream the response as an async iterable.
'''
import asyncio

from pydantic_ai import Agent

agent = Agent(
    model="openai:gpt-4o",
    system_prompt="Be concise and answer only in one sentence."
    )


async def main():
    async with agent.run_stream(user_prompt="Who is Ricken in `Severance` series?") as response:
        print(await response.get_data())


if __name__ == "__main__":
    # Use asyncio.run to run the async function
    asyncio.run(main())
