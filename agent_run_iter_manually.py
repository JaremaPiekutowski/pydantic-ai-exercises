'''
As in agent_run_iter, but drive the iteration manually by passing the node you want
to run next to the AgentRun.next(...) method.
This allows to inspect or modify the node before it executes or skip nodes
based on your own logic, and to catch errors in next() more easily.
'''

import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent(model="openai:gpt-4o")


async def main():
    async with agent.iter("Who is Ricken in `Severance` series?") as agent_run:
        # Get the first node
        node = agent_run.next_node
        # Add it to the list of nodes
        all_nodes = [node]

        # Drive the iteration manually
        while not isinstance(node, End):
            node = await agent_run.next(node)
            all_nodes.append(node)

    print(all_nodes)


if __name__ == "__main__":
    # Use asyncio.run to run the async function
    asyncio.run(main())


# OUTPUT: NODE LIST
# [
#     UserPromptNode(
#         user_prompt='Who is Ricken in `Severance` series?',
#         system_prompts=(),
#         system_prompt_functions=[],
#         system_prompt_dynamic_functions={}
#         ),
#     ModelRequestNode(
#         request=ModelRequest(
#             parts=[
#                 UserPromptPart(
#                     content='Who is Ricken in `Severance` series?',
#                     timestamp=datetime.datetime(
#                         2025, 3, 4, 19, 49, 9, 604303, tzinfo=datetime.timezone.utc
#                         ),
#                     part_kind='user-prompt'
#                     )
#                 ],
#             kind='request'
#             )
#         ),
#     CallToolsNode(
#         model_response=ModelResponse(
#             parts=[
#                 TextPart(
#                     content=(
#                         'In the series "Severance," Ricken Hale is a character '
#                         'who is known for being the free-spirited and philosophical '
#                         'brother-in-law of the main character, Mark Scout. Ricken is '
#                         'married to Mark\'s sister, Devon, and he is portrayed as someone '
#                         'who is interested in alternative ideas and self-help philosophies.'
#                         'Ricken\'s writing and ideas play an interesting role in the series, '
#                         'contrasting the structured and controlled environment of the company, '
#                         'Lumon Industries, where Mark works. Ricken is portrayed by actor '
#                         'Michael Chernus.'
#                         ),
#                     part_kind='text'
#                     )
#                 ],
#             model_name='gpt-4o-2024-08-06',
#             timestamp=datetime.datetime(
#                 2025, 3, 4, 19, 49, 12,
#                 tzinfo=datetime.timezone.utc
#                 ),
#             kind='response')),
#     End(
#         data=FinalResult(
#             data=(
#                 'In the series "Severance," Ricken Hale is a character who is known '
#                 'for being the free-spirited and philosophical brother-in-law of the '
#                 'main character, Mark Scout. Ricken is married to Mark\'s sister, '
#                 'Devon, and he is portrayed as someone who is interested in '
#                 'alternative ideas and self-help philosophies. Ricken\'s writing '
#                 'and ideas play an interesting role in the series, contrasting the '
#                 'structured and controlled environment of the company, Lumon Industries, '
#                 'where Mark works. Ricken is portrayed by actor Michael Chernus.'
#             ),
#             tool_name=None,
#             tool_call_id=None
#       )
#    )
# ]
