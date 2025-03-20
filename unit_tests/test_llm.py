# import pytest
# from dotenv import load_dotenv
#
# from gwenflow import ChatAzureOpenAI
# from gwenflow.tools import WebsiteReaderTool, DuckDuckGoNewsTool
#
# # Load environment variables
# load_dotenv(override=True)
#
# @pytest.fixture
# def chat_model():
#     """Fixture to initialize the ChatAzureOpenAI model."""
#     return ChatAzureOpenAI(model="gpt-4o-mini", tools=[DuckDuckGoNewsTool(), WebsiteReaderTool()])
#
# @pytest.mark.vcr()
# def test_response_stream(chat_model):
#     """Test if the response stream returns valid data."""
#     messages = [
#         {
#             "role": "user",
#             "content": "Get some recent news about Argentina."
#         }
#     ]
#
#     stream = chat_model.response_stream(messages=messages)
#     response_content = ""
#     has_valid_chunk = False  # Flag to check at least one valid chunk
#
#     for chunk in stream:
#         if chunk.content:  # Only check non-empty content
#             assert isinstance(chunk.content, str), "Chunk content should be a string"
#             response_content += chunk.content
#             has_valid_chunk = True
#
#     assert has_valid_chunk, "At least one non-empty chunk should exist"