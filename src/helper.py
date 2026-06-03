from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from src.prompt import *
import os
from dotenv import load_dotenv


# Open AI authentication
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"


def file_processing(file_path):

    #Load data from the pdf
    loader = PyPDFLoader(file_path)
    data = loader.load()

    """
    print(f"Successfully loaded {len(data)} pages from the PDF.\n")

    # Print the text from the first page (index 0) to make sure it worked
    print("--- Page 1 Content ---")
    print(data[0].page_content)
    """

    question_gen = ""
    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = TokenTextSplitter(
        model_name="gpt-4o",
        chunk_size=1000,
        chunk_overlap=200
    )

    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)

    """
    # 1. Verify that chunks were actually created
    print("\n--- Text Splitting Verification ---")
    print(f"Total number of chunks created: {len(chunk_ques_gen)}")

    # 2. Safety check: make sure it isn't empty
    if len(chunk_ques_gen) > 0:
        print(f"Character length of first chunk: {len(chunk_ques_gen[0])}")
        
        print("\n--- Preview of First Chunk ---")
        # Prints the first 300 characters of the first chunk to your screen
        print(chunk_ques_gen[0][:300] + "...") 
    else:
        print("WARNING: No chunks were created. Check your input text!")
    """
    
    document_ques_gen = [Document(page_content = t) for t in chunk_ques_gen]
    #print(type(document_ques_gen[0]))

    splitter_ans_gen = TokenTextSplitter(
        model_name="gpt-4o",
        chunk_size=1000,
        chunk_overlap=200
    )

    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)
    #print(len(document_ques_gen))
    
    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatOpenAI(
    model = "llama-3.3-70b-versatile",
    temperature = 0.3
)


    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])


    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables = ["existing_answer", "text"],
        template = refine_template
    )


    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline,
                                        chain_type = "refine",
                                        verbose = True,
                                        question_prompt = PROMPT_QUESTIONS,
                                        refine_prompt = REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)


    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatOpenAI(temperature=0.1, model="llama-3.3-70b-versatile")

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith("?") or element.endswith(".")]

    answer_generation_chain = RetrievalQA.from_chain_type(
                            llm = llm_answer_gen,
                            chain_type = "stuff",
                            retriever = vector_store.as_retriever()
    )

    return answer_generation_chain, filtered_ques_list

