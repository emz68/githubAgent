from typing import Optional
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from ..integrations.openai import OpenAIInterface
from langchain.schema import HumanMessage, AIMessage

class LangChainIntegration:
    def __init__(self, vector_store, openai_interface: OpenAIInterface):

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
            if context is None:
                context = "\n".join(
                    doc.page_content
                    for doc in self.vector_store.vector_store
                                    .as_retriever()
                                    .get_relevant_documents(question)
                )

            prompt = self.qa_prompt.format(context=context, question=question)

            # load raw history
            raw_history = self.memory.load_memory_variables({})["chat_history"]

            # convert to a list of {"role":..., "content":...} dicts
            formatted_history = []
            for msg in raw_history:
                # if already getting back HumanMessage/AIMessage
                if isinstance(msg, HumanMessage):
                    formatted_history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_history.append({"role": "assistant", "content": msg.content})
                # if plain dicts, fall back:
                elif isinstance(msg, dict) and "input" in msg:
                    formatted_history.append({"role": "user",    "content": msg["input"]})
                elif isinstance(msg, dict) and "output" in msg:
                    formatted_history.append({"role": "assistant", "content": msg["output"]})
                else:
                    # drop anything else or raise
                    continue

            messages = [
                {"role": "system",    "content": "You are a senior developer analyzing code."},
                *formatted_history,
                {"role": "user",      "content": prompt},
            ]

            try:
                response = self.openai.client.chat.completions.create(
                    model="o4-mini",
                    messages=messages
                )

                #logging
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

                if conversational:
                    # save real messages into memory
                    self.memory.save_context(
                        {"question": question},
                        {"answer":   response.choices[0].message.content}
                    )

                return response.choices[0].message.content

            except Exception as e:
                print(f"Error in query: {e}")
                return "Query failed."
        

    def clear_memory(self):
        self.memory.clear()  