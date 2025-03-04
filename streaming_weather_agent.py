'''
Shows how the request and response of an agent are streamed.
'''

import asyncio
from dataclasses import dataclass
from datetime import date

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPartDelta
)
from pydantic_ai.tools import RunContext


# Create a dataclass that contains methods for getting the forecasts.
@dataclass
class WeatherService:
    """A service that provides weather information"""

    async def get_forecast(self, location: str, forecast_date: date) -> str:
        """Get the forecast for a given location and date"""
        # In a normal code an API call would be made here
        return (
            f"The forecast for {location} on {forecast_date} is sunny "
            "with a temperature of 29 degrees celsius."
        )

    async def get_historic_weather(self, location: str, forecast_date: date) -> str:
        """Get the historic weather for a given location and date"""
        # In a normal code an API call would be made here
        return (
            f"The historic weather for {location} on {forecast_date} was sunny "
            "with a temperature of 18 degrees celsius."
        )


# Create the agent with the WeatherService class as dependency
# and str as the return type.
weather_agent = Agent[WeatherService, str](
    model="openai:gpt-4o",
    deps_type=WeatherService,
    result_type=str,
    system_prompt="You are a helpful assistant who provides weather information.",
)


# Create a tool that gets either the forecast or the historic weather.
@weather_agent.tool
async def weather_forecast(
    ctx: RunContext[WeatherService],
    location: str,
    forecast_date: date,
) -> str:
    if forecast_date == date.today():
        return await ctx.deps.get_forecast(location, forecast_date)
    else:
        return await ctx.deps.get_historic_weather(location, forecast_date)

# Create a list to store the messages.
output_messages: list[str] = []


# Create a main function that runs the agent.
async def main():
    user_prompt = "What is the weather in Szczecin on Tuesday?"

    async with weather_agent.iter(
        user_prompt,
        deps=WeatherService(),
    ) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # A user prompt node -> the user has provided input
                output_messages.append(f"=== UserPromptNode:{node.user_prompt} ===")
            elif Agent.is_model_request_node(node):
                # A model request node -> we can stream tokens from the model's request
                output_messages.append("=== ModelRequestNode: streaming partial request tokens ===")
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            output_messages.append(
                                f"[Request] Starting part {event.index}: {event.part!r}"
                            )
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                output_messages.append(
                                    f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                                )
                            elif isinstance(event.delta, ToolCallPartDelta):
                                output_messages.append(
                                    f"[Request] Part {event.index} args_delta={event.delta.args_delta}"
                                )
                        elif isinstance(event, FinalResultEvent):
                            output_messages.append(
                                f"[Result] The model produced a final result (tool_name={event.tool_name})"
                            )
            elif Agent.is_call_tools_node(node):
                # A handle-response node -> the model returned some data, potentially calls a tool
                output_messages.append("=== CallToolsNode: streaming partial response and tool usage ===")
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            output_messages.append(
                                f"[Tools] The LLM calls tool={event.part.tool_name} "
                                f"with args={event.part.args} "
                                f"(tool_call_id={event.part.tool_call_id!r})"
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            output_messages.append(
                                f"[Tools] Tool call {event.tool_call_id!r} returned => "
                                f"{event.result.content}"
                            )
            elif Agent.is_end_node(node):
                assert run.result.data == node.data.data
                # Once an end node is reached, the run is complete.
                output_messages.append(
                    f"=== EndNode: The final result is: {run.result.data} ==="
                )

if __name__ == "__main__":
    asyncio.run(main())

    print("\n\nOutput messages:")
    print("\n".join(output_messages))
