import langchain.embeddings.huggingface as hf

class EmbeddingModelSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = 'cpu'
        self.embed_model = hf.HuggingFaceEmbeddings(
            model_name=self.embed_model_id,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': 32}
        )

    def embed_documents(self, text):
        return self.embed_model.embed_documents(text)

# Access the singleton instance and use its methods:

