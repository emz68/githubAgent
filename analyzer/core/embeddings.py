from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
import torch

class EmbeddingManager:
    def __init__(self):
        self.model_name = "Salesforce/SFR-Embedding-Code-400M-R"  # Note: Correct hyphen
        self._initialize_models()

    def _initialize_models(self):
            self.model_name = "Salesforce/SFR-Embedding-Code-400M_R"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model.eval()

    def get_embeddings(self, texts: list[str], model_type: str = "hf") -> list[list[float]]:
        """Get embeddings for a list of texts"""
        if model_type == "hf":
            inputs = self.hf_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.hf_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).tolist()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")