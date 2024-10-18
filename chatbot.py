import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import hub
import pprint

load_dotenv()

class ChatBot:
    def __init__(self):
        # Initialize Pinecone, Google Generative AI, and load environment variables
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.index_name = os.environ.get("INDEX_NAME")

        # Setup Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)

    def get_response(self, message):
        """
        Receives a message (query) and returns a response based on embeddings and similarity search in Pinecone.
        If no documents are found, it queries the Google Gemini API directly for information.
        """
        try:
            # Step 1: Create embedding for the incoming query message

            custom_embeddings = HuggingFaceEmbeddings()


            # Step 3: Perform similarity search in Pinecone
            vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=custom_embeddings)
            result = vectorstore.similarity_search(message)
            #print(result)

            # Step 4: Convert Pinecone search results to LangChain Document objects
            documents = [
                Document(page_content=doc.page_content, metadata=doc.metadata) for doc in result if doc.page_content
            ]

            # Check if we have found any relevant documents
            if not documents:
                # If no documents were found, query the LLM directly for more information
                return self.query_llm(message)

            template = """
            You are a fortune teller. These Human will ask you a questions about their life. 
            Use following piece of context to answer the question. 
            If you don't know the answer, just say you don't know. 
            Keep the answer within 2 sentences and concise.

            Context: {context}
            Question: {question}
            Answer: 

            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )


            # Define the repo ID and connect to Mixtral model on Huggingface
            repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            llm = HuggingFaceHub(
                repo_id=repo_id,
                model_kwargs={"temperature": 0.8, "top_k": 50},
                huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
            )


        except Exception as e:
            return f"Error: {str(e)}"

    def get_response2(self, message):

        custom_embeddings = HuggingFaceEmbeddings()

        template = """
                You are an expert on tao face yoga. Give me a precise answer.

                Context: {context}
                Question: {question}
                Answer: 

                """
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )


        index = os.environ.get("INDEX_NAME")
        vectorstore = PineconeVectorStore(index_name=index, embedding=custom_embeddings)

        print("result" + str(vectorstore))

        # Define the repo ID and connect to Mixtral model on Huggingface
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        #repo_id = "Qwen/Qwen2-72B-Instruct"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

        # Create the rag_chain with the context and question
        rag_chain = (
                prompt
                | llm
                | StrOutputParser()
        )

        # Create a dictionary with the context and question
        inputs = {
            "context": vectorstore.as_retriever(),
            "question": prompt
        }


        # Execute the chain with the inputs
        response = rag_chain.invoke(inputs)  # Correctly pass inputs to the chain
        # If the answer is in a formatted string, you can slice it:
        if "Answer:" in response:
            answer = response.split("Answer:")[1].strip()  # This gets the part after "Answer:"
        else:
            answer = response  # If no special format, return the whole response

        return response


    def get_response3(self, message):


        index = os.environ.get("INDEX_NAME")
        model_id = "Qwen/Qwen2-72B-Instruct"
        #tokenizer = AutoTokenizer.from_pretrained(model_id)
        #model = AutoModelForCausalLM.from_pretrained(model_id)

        # Set up the pipeline for text generation
        #pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_token=100)


        #hf = HuggingFacePipeline(pipeline = pipe)
        #llm = hf
        #prompt = hub.pull("rlm/rag-prompt")
        #vectoreStore = PineconeVectorStore.from_documents(
        #    embedding_function=HuggingFaceEmbeddings(),  # Pass your custom embedding model
        #    index_name=index
        #)

        #qa_chain = RetrievalQA.from_chain_type(
        #   llm,
        #    retriever=vectoreStore.as_retriever(),
        #chain_type_kwargs={"prompt": prompt}
        #)

        #question = message
        #result = qa_chain({"query": question})
        #pp = pprint.PrettyPrinter(indent=4)


        #return pp.pprint(result["result"])
        return "hola"

    def get_response_inference(message):
        # Use Hugging Face API for inference
        model_id = "Qwen/Qwen2.5-72B-Instruct"

        # Initialize the LLM using Hugging Face Hub
        llm = HuggingFaceHub(
            repo_id=model_id,
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
            model_kwargs={"temperature": 0.7, "top_k": 50}
        )

        # Define a prompt template that expects both a question and context
        prompt = PromptTemplate(
            template="You are an expert on face yoga. Use the following context to answer the question.\n\n"
                     "Context: {context}\n"
                     "Question: {question}\n\n"
                     "Answer concisely:",
            input_variables=["context", "question"]
        )

        # Setup your vector store as usual (if using Pinecone)
        index = os.environ.get("INDEX_NAME")
        custom_embeddings = HuggingFaceEmbeddings()
        vectorstore = PineconeVectorStore(index_name=index, embedding=custom_embeddings)

        # Retrieval-based QA with the proper prompt template
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

        # Ask the question and retrieve context
        result = qa_chain({"query": message})

        # Extract the result and clean up any unnecessary "Human" or "Assistant" prefixes
        answer = result["result"]

        # Extract the text after "Answer concisely:"
        marker = "Answer concisely:"
        if marker in answer:
            concise_answer = answer.split(marker)[1].strip()  # Get everything after "Answer concisely:"
        else:
            concise_answer = answer  # If "Answer concisely:" is not found, return the full answer

        # Return the concise answer
        return concise_answer


chatbot = ChatBot()
#print(chatbot.get_response_inference("you are an expert content writer. Your task is to create an 800-word article based on the following context. Please ensure the article has a clear introduction, body, and conclusion. Use the key points mentioned in the context to guide the structure, and feel free to elaborate on ideas to create a coherent narrative.can you create an article of 800 words with the content and context provided?"))
#print(ChatBot.get_response_inference("where is savina atai from and where does face yoga come from?"))




