import os
from pyprojroot import here
import pandas as pd
import chromadb
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
print(load_dotenv())

# Load the LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_base= os.getenv("OPENAI_API_BASE"),
    openai_api_key= os.getenv("OPENAI_API_KEY"),
    # tream=True,
    temperature=0)


chroma_client = chromadb.PersistentClient(path=str(here("data/chroma")))

# 列出所有集合的名称
existing_collections = chroma_client.list_collections()

collection_name = "titanic_small"

# 获取所有集合
existing_collections = chroma_client.list_collections()

# 提取集合名称
existing_collection_names = [collection.name for collection in existing_collections]

if collection_name in existing_collection_names:
    # 如果集合存在，获取它
    collection = chroma_client.get_collection(name=collection_name)
    print(f"Retrieved existing collection: {collection_name}")
else:
    # 如果集合不存在，创建它
    collection = chroma_client.create_collection(name=collection_name)
    print(f"Created new collection: {collection_name}")


file_dir = here("data/for_upload/titanic_small.csv")
df = pd.read_csv(file_dir, nrows=5)

from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

# 设置 OpenAI API 密钥
import os
# os.environ["OPENAI_API_KEY"] = openai_api_key  # 如果你已经设置了环境变量，则不需要这行

# 创建 OpenAIEmbeddings 实例
OpenAIEmbeddings = OpenAIEmbeddings()

docs = []
metadatas = []
ids = []
embeddings = []
for index, row in df.iterrows():
    output_str = ""
    # Treat each row as a separate chunk
    for col in df.columns:
        output_str += f"{col}: {row[col]},\n"
    response = OpenAIEmbeddings.embed_documents(output_str)[0]
    embeddings.append(response)
    docs.append(output_str)
    metadatas.append({"source": "titanic_small"})
    ids.append(f"id{index}")

collection.update(
    documents=docs,
    metadatas=metadatas,
    embeddings=embeddings,
    ids=ids
)

print("Number of vectors in vectordb:", collection.count())

query_texts = "what's the average age of survivors"
response = OpenAIEmbeddings.embed_documents(query_texts)[0]
query_embeddings = response

results = vectordb.query(
    query_embeddings = query_embeddings,
    n_results=1 #top_k
)


from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    # tream=True,
    temperature=0)

system_role = "You will recieve the user's question along with the search results of that question over a database. Give the user the proper answer."
prompt = f"User's question: {query_texts} \n\n Search results:\n {results}"

messages = [
    {"role": "system", "content":system_role},
    {"role": "user", "content": prompt}
]

# Call the model with the messages
response = chat(messages)

# Print the response
print(response.content)