from typing import Optional
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from ..integrations.openai import OpenAIInterface

class LangChainIntegration:
    def __init__(self, vector_store, openai_interface: OpenAIInterface):
        """
        Args:
            vector_store:  VectorStoreManager instance
            openai_interface:  existing OpenAIInterface
        """
        self.vector_store = vector_store
        self.openai = openai_interface
        self._initialize_components()

    def _initialize_components(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            Answer the question based on the context.
            Provide detailed explanations with code examples when relevant.
            Explain and answer questions about the code to the best of your ability. 
            If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
        )

    def query(self, question: str, conversational: bool = False, context: Optional[str] = None) -> str:
        # If context isn't provided, retrieve it from vector store
        if context is None:
            context = "\n".join(
                doc.page_content for doc in 
                self.vector_store.vector_store.as_retriever().get_relevant_documents(question)
            )

        # Format the prompt with the context
        prompt = self.qa_prompt.format(context=context, question=question)

        messages = [
            {"role": "system", "content": "You are a senior developer analyzing code."},
            *self.memory.load_memory_variables({})["chat_history"],
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.openai.client.chat.completions.create(
                model="o4-mini",
                messages=messages
            )

            # Log usage
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
            print(f"Error in query: {e}")
            return "Query failed."
        

    def clear_memory(self):
        self.memory.clear()  