from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser

# Provide pure JSON without markdown code block delimiters
message = AIMessage(content='{"foo": "bar"}')
output_parser = JsonOutputParser()

# Parse the message
parsed_output = output_parser.invoke(message)
print(parsed_output)
