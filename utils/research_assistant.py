import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Robust LangChain imports with version compatibility
RAG_AVAILABLE = False
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    
    # Try different import paths for ConversationalRetrievalChain
    try:
        from langchain.chains import ConversationalRetrievalChain
    except ImportError:
        try:
            from langchain_community.chains import ConversationalRetrievalChain
        except ImportError:
            from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    
    # Try different import paths for ConversationBufferMemory
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        from langchain_community.memory import ConversationBufferMemory
    
    RAG_AVAILABLE = True
    logger.info("LangChain RAG components loaded successfully.")
except ImportError as e:
    logger.warning(f"LangChain not fully available. RAG features disabled. Error: {e}")
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    FAISS = None
    OpenAIEmbeddings = None
    ChatOpenAI = None
    ConversationalRetrievalChain = None
    ConversationBufferMemory = None


class ResearchAssistant:
    # Project documentation files to auto-load as base context
    PROJECT_DOCS = [
        "Chrono-Trader_v2_Paper.md",
        "README.md",
    ]
    
    def __init__(self, upload_dir="data/papers"):
        self.upload_dir = upload_dir
        self.vector_store_path = "data/faiss_index"
        self.embeddings = None
        self.vector_store = None
        self.chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) if RAG_AVAILABLE else None
        self._index_loaded_at = None  # Track when index was loaded
        self._project_docs_loaded = False  # Track if project docs are loaded
        
        # Ensure directories exist
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Initialize if API Key exists
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and RAG_AVAILABLE:
            self._init_components()

    def _init_components(self):
        if not RAG_AVAILABLE:
            return
        try:
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            if os.path.exists(self.vector_store_path) and os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
                self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
                self._index_loaded_at = datetime.now()  # Track load time
                logger.info("Loaded existing FAISS index.")
            else:
                self.vector_store = None
            
            # Auto-load project documentation as base context
            if not self._project_docs_loaded:
                self._load_project_docs()
            
            if self.vector_store:
                self._build_chain()
        except Exception as e:
            logger.error(f"Failed to init RAG components: {e}")
    
    def _load_project_docs(self):
        """Auto-ingest project documentation as base context for relating external papers."""
        if not RAG_AVAILABLE or not self.embeddings:
            return
        
        try:
            from langchain.schema import Document
            
            # Find project root (parent of utils directory)
            project_root = os.path.dirname(os.path.dirname(__file__))
            
            documents = []
            for doc_name in self.PROJECT_DOCS:
                doc_path = os.path.join(project_root, doc_name)
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document with metadata indicating it's a project doc
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": doc_name,
                            "type": "project_documentation",
                            "description": f"Chrono-Trader v2 project documentation: {doc_name}"
                        }
                    )
                    documents.append(doc)
                    logger.info(f"Loaded project doc: {doc_name}")
                else:
                    logger.warning(f"Project doc not found: {doc_path}")
            
            if documents:
                # Split documents into chunks
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)
                
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(texts, self.embeddings)
                else:
                    self.vector_store.add_documents(texts)
                
                self.vector_store.save_local(self.vector_store_path)
                self._project_docs_loaded = True
                logger.info(f"Project documentation loaded: {len(texts)} chunks from {len(documents)} docs")
        except Exception as e:
            logger.error(f"Error loading project docs: {e}")

    def _build_chain(self):
        if not RAG_AVAILABLE:
            return
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=self.api_key)
        
        from langchain.prompts import PromptTemplate
        template = """You are a PhD-level research assistant for the **Chrono-Trader v2** cryptocurrency prediction project.
        
        IMPORTANT: The user's project (Chrono-Trader v2) uses:
        - Transformer with Contextual Positional Encoding (market index + historical similarity)
        - CNN/TCN for local pattern recognition  
        - ExplainableGatedFusion with Prototype Bank
        - GAN Decoder with MC Dropout for uncertainty
        - BTC+ETH weighted index as crypto-native market factor
        
        Use the following context (which includes both PROJECT DOCUMENTATION and EXTERNAL PAPERS) to answer questions.
        
        CRITICAL RULES:
        1. Always relate external research back to the Chrono-Trader v2 project.
        2. When discussing papers, explain how they could improve or relate to the current implementation.
        3. Cite sources in [Author, Year] format (e.g., [Vaswani et al., 2017]).
        4. If asked about project architecture, reference the Chrono-Trader v2 documentation.
        5. Be precise and academic, helping the user improve their thesis/project.

        Context: {context}

        Question: {question}
        Helpful Answer (relating to Chrono-Trader v2 where applicable):"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        logger.info("RAG Chain built successfully with gpt-4o-mini and Citation Enforcement.")

    def save_to_obsidian(self, title, content):
        """Save research notes to Obsidian/Markdown"""
        try:
            vault_path = os.getenv("OBSIDIAN_VAULT_PATH", os.path.join(os.path.dirname(self.upload_dir), 'obsidian_exports'))
            os.makedirs(vault_path, exist_ok=True)
            
            clean_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).rstrip()
            filename = os.path.join(vault_path, f"{clean_title}.md")
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"**Source:** AETHER Research Lab\n\n")
                f.write("---\n\n")
                f.write(content)
            
            return True, f"Saved to {filename}"
        except Exception as e:
            logger.error(f"Obsidian save failed: {e}")
            return False, str(e)


    def set_api_key(self, key):
        self.api_key = key
        os.environ["OPENAI_API_KEY"] = key
        
        try:
            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            lines = []
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    lines = f.readlines()
            
            lines = [l for l in lines if not l.startswith("OPENAI_API_KEY=")]
            lines.append(f"OPENAI_API_KEY={key}\n")
            
            with open(env_path, 'w') as f:
                f.writelines(lines)
            
            logger.info(f"API Key saved to {env_path}")
        except Exception as e:
            logger.error(f"Failed to save API key to .env: {e}")

        if RAG_AVAILABLE:
            self._init_components()

    def ingest_pdf(self, file_path):
        if not RAG_AVAILABLE:
            return False, "RAG not available. Install langchain packages."
        if not self.embeddings:
            return False, "API Key not configured."

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(texts, self.embeddings)
            else:
                self.vector_store.add_documents(texts)
            
            self.vector_store.save_local(self.vector_store_path)
            self._build_chain()
            return True, f"Processed {len(texts)} chunks from PDF."
        except Exception as e:
            logger.error(f"Error ingesting PDF: {e}")
            return False, str(e)

    def reindex_project_docs(self):
        """
        Force re-index project documentation.
        Call this when MD files are updated to sync Research Lab with latest content.
        """
        if not RAG_AVAILABLE or not self.embeddings:
            return False, "RAG not available or not initialized."
        
        try:
            # Delete existing index if it exists
            import shutil
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
                logger.info("Deleted existing FAISS index for rebuild.")
            
            # Reset state
            self.vector_store = None
            self._project_docs_loaded = False
            
            # Reload project docs
            self._load_project_docs()
            
            if self.vector_store:
                self._build_chain()
                return True, f"Project documentation re-indexed successfully."
            else:
                return False, "Failed to create vector store."
        except Exception as e:
            logger.error(f"Error re-indexing: {e}")
            return False, str(e)

    def chat(self, query):
        if not RAG_AVAILABLE:
            return "⚠️ RAG 기능이 비활성화되어 있습니다. LangChain 패키지를 설치해주세요."
        if not self.api_key:
            return "⚠️ OpenAI API Key가 설정되지 않았습니다. 'Set key [sk-...]' 라고 입력하거나 환경변수를 설정해주세요."
        
        # Auto-reload index if older than 5 minutes (to pick up new PDFs)
        if self._index_loaded_at and (datetime.now() - self._index_loaded_at).seconds > 300:
            logger.info("Index cache expired, reloading...")
            self._init_components()
        
        if not self.chain:
            return "📚 읽은 논문이 없습니다. PDF 파일을 먼저 업로드해주세요."

        try:
            result = self.chain.invoke({"question": query})
            return result['answer']
        except Exception as e:
            logger.error(f"Chat Error: {e}")
            return f"오류가 발생했습니다: {e}"

    def summarize_conversation(self, messages):
        """Summarize chat history into a structured research note"""
        if not self.chain or not RAG_AVAILABLE:
             return None
        
        try:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            prompt = f"""
            You are an expert Research Assistant. 
            Synthesize the following discussion between a Researcher and AI into a high-quality academic note for Obsidian.
            Do not just transcribe. Extract value.

            Chat History:
            {history_text}

            Output Format (Markdown):
            # [Title based on content]
            **Date:** {datetime.now().strftime('%Y-%m-%d')}
            **Tags:** #research #ai #summary

            ## 1. Research Context
            (What was discussed? What papers were questioned?)

            ## 2. Key Insights & Findings
            (Bullet points of the most important answers provided by AI)

            ## 3. Methodology / Technical Details
            (Any specific algorithms, formulas, or implementation details mentioned)

            ## 4. Next Steps / Action Items
            (What did the user plan to do? What suggestions were made?)
            """
            
            llm = self.chain.combine_docs_chain.llm_chain.llm
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Summarization Error: {e}")
            return f"Error generating summary: {e}"

# Global Instance
assistant = ResearchAssistant()

