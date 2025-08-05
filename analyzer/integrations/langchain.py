from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from typing import Optional

class LangChainIntegration:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LangChain components"""
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1000
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a senior Python developer analyzing code. Answer the question based on the context.
            Provide detailed explanations with code examples when relevant.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
        )
    
    def query(self, question: str, conversational: bool = False) -> str:
        """Execute a query using LangChain"""
        if conversational:
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.langchain_vector_store.as_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self.qa_prompt}
            )
            result = qa_chain({"question": question})
            return result["answer"]
        else:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.langchain_vector_store.as_retriever(),
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
            return qa_chain.run(question)