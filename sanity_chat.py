from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_template(
    "Write one-line explanation (<=12 words) about {topic}."
)

# smaller model + smaller context for low RAM
llm = ChatOllama(model="llama3.2:1b", temperature=0.4, num_ctx=1024)
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    topic = input("Topic: ").strip()
    print("\n" + chain.invoke({"topic": topic}))
