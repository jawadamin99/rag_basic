import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import CHUNK_SIZE, RETRIEVAL_TOP_K, STORAGE_DIR, DATA_DIR


class HistoryChatBotRag():
    def __init__(self, model_name, embedding_model, prompt_template):
        print("--- 1- CONNECTION BUILDING --- ")
        self.llm = Ollama(
            model=model_name, temperature=0.7,
            # define [port
            #base_url="http://localhost:11434"
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            #base_url="http://localhost:11434"
            # define [port
        )

        print("--- 2- CREATING VECTORSTORES --- ")
        vectorstore = self._get_or_create_embeddings()

        print("--- 3- RAG SETUP --- ")
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
        prompt = PromptTemplate(
            template=prompt_template,
            input_variable=['context', 'question']
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=retriever,
            chain_type_kwargs={'prompt': prompt},
            return_source_documents=True
        )

    def _get_or_create_embeddings(self):
        """
        Create vector if any
        Fetch vectors if already existed
        :return:
        """

        faiss_index_file = os.path.join(STORAGE_DIR, "faiss_index.faiss")
        faiss_index_meta = os.path.join(STORAGE_DIR, "index.pkl")

        if os.path.exists(faiss_index_file) and os.path.exists(faiss_index_meta):
            # when we already have vectors
            #return FAISS.load_local(STORAGE_DIR, self.embeddings, index_meta="faiss_index")
            return FAISS.load_local(folder_path=STORAGE_DIR, embeddings=self.embeddings, index_name="faiss_index"
        )

        if not any(os.listdir(DATA_DIR)):
            # when there is no data
            return FAISS.from_document([Document(page_content="No document loaded")], self.embeddings)

        loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*",
            loader_cls=PyMuPDFLoader,
            silent_errors=True
        )

        print("Loading PDFs...")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=int(CHUNK_SIZE * 0.1)
        )

        print("Splitting documents...")
        texts = text_splitter.split_documents(documents)
        print(f"DEBUG: First chunk content: {texts[0].page_content[:200]}")
        print(f"DEBUG: Total chunks created: {len(texts)}")
        print("Creating embeddings...")
        vectorstore = FAISS.from_documents(texts, self.embeddings)


        #vectorstore.store_local(STORAGE_DIR, index_name="faiss_index")

        print("Saving FAISS index...")
        vectorstore.save_local(STORAGE_DIR, index_name="faiss_index")
        print("Datastore built successfully")

        return vectorstore

    def get_response(self, user_input):
        """
        take user input and send to AI Agent for the answer
        :param user_input:
        :return:
        """

        results = self.chain.invoke({'query': user_input})
        sources = []
        for doc in results.get('source_documents', []):
            source_path = doc.metadata.get("source", "Unknown Document")
            filename = os.path.basename(source_path)
            #sources.append(f"{filename} (Page: {doc.metadata('page', 'N/A')})")
            page = doc.metadata.get("page", "N/A")
            sources.append(f"{filename} (Page: {page})")
        answer = results.get("result", "")

        return {
            'answer': answer,
            'sources': list(set(sources))
        }
