from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


llm = ChatOpenAI(temperature=0.9)

def main():
    # read txt from elon_musk.txt
    elon_musk_bio = ""
    with open("elon_musk_info.txt", "r") as f:
        elon_musk_bio = f.read()

    summary_template = """
    Given the information {bio}, provide:
        1. a concise summary of the key points about Elon Musk.
        2. 2 interesting facts about Elon Musk.
    """

    prompt = PromptTemplate(
        input_variables=["bio"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0)
    print(llm.invoke(
        prompt.format(bio=elon_musk_bio)
    ).content)

if __name__ == "__main__":
    main()