import context
import json
from gwenflow.prompts import PromptTemplate, PipelinePromptTemplate


prompt1 = PromptTemplate.from_template("Test de prompt système")
print(prompt1)

prompt2 = PromptTemplate.from_template("Test de prompt avec une variable: {query} et {date}")
print(prompt2)
print(prompt2.format(query="test", date="Apr 2023"))
print(prompt2.input_variables)

prompt3 = PromptTemplate.from_template("contexte: {context}")
prompt4 = PromptTemplate.from_template("Encore un contexte: {context}")

pipeline = PipelinePromptTemplate(prompts=[prompt1, prompt2, prompt3, prompt4])
print(pipeline.input_variables)
print(json.dumps(pipeline.format(query="test", date="Apr 2023", context="Le Monde", test="dummy"), indent=2))
