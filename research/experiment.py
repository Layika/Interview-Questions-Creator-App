import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

os.environ["OPENAI_API_KEY"] = GROQ_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"

from langchain_community.document_loaders import PyPDFLoader

file_path = "../data/SDGs.pdf" #go up one folder, Since the script code is running inside the research folder, you tell Python to look outside its current folder first
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


from langchain_text_splitters import TokenTextSplitter

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

# Convert the chunks into document object because split_text() was used which outputted raw strings
from langchain_core.documents import Document

document_ques_gen = [Document(page_content = t) for t in chunk_ques_gen]
#print(type(document_ques_gen[0]))

splitter_ans_gen = TokenTextSplitter(
    model_name="gpt-4o",
    chunk_size=1000,
    chunk_overlap=200
)

document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)
#print(len(document_ques_gen))

from langchain.chat_models import ChatOpenAI
llm_ques_gen_pipeline = ChatOpenAI(
    model = "llama-3.3-70b-versatile",
    temperature = 0.3
)

prompt_template = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their coding and exam tests.
You do this by asking the questions about the text below:

-------------------
{text}
-------------------

Create questions that will prepare coders or programmers for their tests.
Make sure not to lose any important information.

QUESTIONS:
"""

from langchain_core.prompts import PromptTemplate

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

refine_template = ("""
You are an expert at creating practice questions based on coding materials and documents.
Your goal is to help a coder or programmer prepare or a coding test.
We have recieved some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.

---------------
{text}
---------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:                
""")