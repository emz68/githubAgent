from typing import Optional
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from ..integrations.openai import OpenAIInterface

class LangChainIntegration:
    def __init__(self, vector_store, openai_interface: OpenAIInterface):
        """
        Args:
            vector_store: Your VectorStoreManager instance
            openai_interface: Your existing OpenAIInterface
        """
        self.vector_store = vector_store
        self.openai = openai_interface
        self._initialize_components()

    def _initialize_components(self):
        """Initialize components using your existing OpenAI interface"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a senior developer analyzing code. Answer the question based on the context.
            Provide detailed explanations with code examples when relevant.
            Explain and answer questions about the code to the best of your ability. 
            If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
        )

    def query(self, question: str, conversational: bool = False) -> str:
        # Get relevant context from vector store
        context = "\n".join(
            doc.page_content for doc in 
            self.vector_store.vector_store.as_retriever().get_relevant_documents(question)
        )

        # Format the prompt
        prompt = self.qa_prompt.format(context=context, question=question)

        # Use your existing OpenAI interface
        messages = [
            {"role": "system", "content": "You are a technical assistant."},
            *self.memory.load_memory_variables({})["chat_history"],
            {"role": "user", "content": prompt}
        ]


        response = self.openai.client.chat.completions.create(
            model="o4-mini",
            messages=messages
        )

        try:
            response = self.openai.client.chat.completions.create(
                model="o4-mini",
                messages=messages
            )

            # Log usage (mirroring OpenAIInterface's logic)
            if hasattr(response, 'usage'):
                self.openai._log_usage(
                    operation="advanced_query",
                    model="o4-mini",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )

            # Update memory if conversational
            if conversational:
                self.memory.save_context(
                    {"input": question},
                    {"output": response.choices[0].message.content}
                )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in advanced_query: {e}")
            return "Query failed."
        

    def clear_memory(self):
        self.memory.clear()  