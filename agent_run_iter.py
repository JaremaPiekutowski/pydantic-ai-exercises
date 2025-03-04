'''
Run the agent as an async iterable over the nodes of the agent's underlying Graph.
'''
import asyncio
from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o")


async def main():
    nodes = []
    with agent.iter("Who is Ricken in `Severance` series?") as agent_run:
        async for node in agent_run:
            nodes.append(node)

    print(nodes)


if __name__ == "__main__":
    # Use asyncio.run to run the async function
    asyncio.run(main())


# OUTPUT: NODE LIST
#
# [
#     ModelRequestNode(
#         request=ModelRequest(
#             parts=[
#                 UserPromptPart(
#                     content='Who is Ricken in `Severance` series?',
#                     timestamp=datetime.datetime(2025, 2, 28, 21, 6, 1, 963750, tzinfo=datetime.timezone.utc),
#                     part_kind='user-prompt'
#                 )
#             ],
#             kind='request'
#         )
#     ),
#     HandleResponseNode(
#         model_response=ModelResponse(
#             parts=[
#                 TextPart(
#                     content=(
#                         'In the series "Severance," Ricken is a character who is known for being a self-help author.'
#                         'He is married to Devon, who is the sister of the main character, Mark.'
#                         'Ricken is characterized by his philosophical and often pretentious approach to life, '
#                         'often offering unsolicited advice and insights, which can sometimes be humorous '
#                         'or inadvertently insightful.'
#                     ),
#                     part_kind='text'
#                 )
#             ],
#             model_name='gpt-4o-2024-08-06',
#             timestamp=datetime.datetime(2025, 2, 28, 21, 6, 2, tzinfo=datetime.timezone.utc),
#             kind='response'
#         )
#     ),
#     End(
#         data=FinalResult(
#             data=(
#                 'In the series "Severance," Ricken is a character who is known for being a self-help author. '
#                 'He is married to Devon, who is the sister of the main character, Mark. '
#                 'Ricken is characterized by his philosophical and often pretentious approach to life, '
#                 'often offering unsolicited advice and insights, which can sometimes be humorous '
#                 'or inadvertently insightful.'
#             ),
#             tool_name=None
#         )
#     )
# ]