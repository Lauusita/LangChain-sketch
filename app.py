import openai
import os
# PDF'S IMPORTATION
from langchain.document_loaders import PyPDFLoader
# TXT SPLITTER
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
# EMBEDDINGS - VECTORS
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
# TIKTOKEN
import tiktoken
# LLM 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

openai.api_key = os.getenv('OPENAI_API_KEY')

pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV')) 
# print(pinecone.list_indexes())
os.path.dirname('MOTOR CREDITO -  CRECER - TARIFA - 14-7-2023.pdf')

if __name__ == "__main__":
    # print('Hello langchain')

    load = PyPDFLoader('../REGLAS AFFINITY/MOTOR CREDITO -  CRECER - TARIFA - 14-7-2023.pdf') # load the PDF by MY directory, i should put dir but im lazy :p
    document1 = PyPDFLoader.load(self=load) # the self=load is redundant, it just returns the function 
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=200) # calls the function to split the whole document in chunks of 1.4k
    
    texts = text_splitter.split_documents(documents=document1) 

    print(len(texts)) # print the amount of chunks that the splitter generates
    
    # print(texts[2].page_content)   

    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key) 

    vector = Pinecone.from_documents(texts, embeddings, index_name= "seguros-affinity") # Pinecone is a vector database in which i will save the amount of chunks there, taking as a second argument the OpenAI Embedding model and as a third argument de index name, that is the name of the Pinecone index that i have created 

    print('se ha conectado :)') # hola :))


    def search(vector, pregunta): # Just for convenience, i made a new fuction that will take as arguments the vector and the question.
        llm = ChatOpenAI(temperature=0, model="gpt-4") # i establish as larga language model the OpenAI model, with temperature 0 and with gpt-4 model.
        retriever = vector.as_retriever(search_type="similarity") # vector as retriever basically extracts the info of each vector (that is the information of the document but splitted on vectors ) of vector (variable) and search the information by similarity

        chain = RetrievalQA.from_chain_type(llm = llm, chain_type="stuff", retriever= retriever) # i create a chain that will return the search using the RetrievalQA as a langchain.chain function, that search from chain type and requires the large language model, the chain-type "stuff", that basically takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM, and the retriever equals the information extracted

        answer = chain.run(pregunta) # runs the pregunta argument as the chain that was previously initializated. 
        return answer

    important = "Muéstrame los datos en un archivo JSON y la prima total calculada."

    pregunta= f"la marca del vehículo es ford, modelo fiesta, año 2020, tipo de vehículo automovil, valor asegurado de RD$1,200,000, sin adición de servicios adicionales, póliza del motor es deducible. Ten en cuenta que la tarifa base se suma con el valor de la tarifa del valor asegurado. {important}"

    def token(): # this is unnecesary, but i wanted to see how many tokens the prompt and the answer generates, usting tiktoken. 
        encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
        for page in texts:
            promptTokens = sum([len(encoding.encode(page.page_content))])
            answerTokens = sum([len(encoding.encode(pregunta))])
            totalTokens = promptTokens + answerTokens
        print(f'prompt tokens: {promptTokens}')
        print(f'answer tokens: {answerTokens}')
        print(f'total tokens: {totalTokens}')
        print(f'USD $: {totalTokens / 1000*0.0001}')

    response =search(vector, pregunta)
    print(response)
    print(token())

    






# peras = """
# maría fue a la tienda y compró dos amnzadas y tres peras

# """ 

    # text_splitter = CharacterTextSplitter(chunk_size=1000, )
    


    # pears = """ me podrías decir cuántas peras {peras} compró maría? """ # se define el prompt del user la cual es una pregunta

    # question = PromptTemplate(input_variables=["pera"], template=pears) # 

    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    # callingPrompt = LLMChain(llm=llm, prompt=question)

    # print(callingPrompt.run(peras = peras))
