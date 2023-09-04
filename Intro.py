import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

from langchain import PromptTemplate, HuggingFaceHub, LLMChain

flan_t5 = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":1e-10})
template = """Question:{question} 

Answer: """
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=flan_t5)

question = "Which NFL team won the Super Bowl in the 2010 season?"

print(llm_chain.run(question))
print("Done")