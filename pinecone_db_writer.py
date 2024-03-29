from model import EmbeddingModelSingleton
import pinecone
import time
import os

PINECONE_KEY = os.getenv('PINECONE_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

class PineconeDbWritter:
    _instance = None
    _index = None
    _embed_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        pinecone.init(
            api_key=PINECONE_KEY,
            environment=PINECONE_ENV
        )
        self._embed_model = EmbeddingModelSingleton()

    def create_index(self, index_name):
        embedded = self._embed_model.embed_documents("get dim")
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                index_name,
                dimension=len(embedded[0]),
                metric='cosine'
            )
            
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)
        self._index = pinecone.Index(index_name)

    def write(self, files):
        if self._index is None:
            raise ValueError("Index not created. Call create_index first.")


        batch_size = 32

        for file_name in files:
            for file in files[file_name]:

                for i in range(0, len(file), batch_size):
                    i_end = min(len(file), i+batch_size)
                    batch = file.iloc[i:i_end]
                    texts = [x['chunk'] for i, x in batch.iterrows()]
                    embeds = self._embed_model.embed_documents(texts)
                    # get metadata to store in Pinecone
                    metadata = [
                        {'text': x['chunk'],
                        'source': x['source'],
                        'title': x['title']} for i, x in batch.iterrows()
                    ]
                    # add to Pinecone
                    self._index.upsert(vectors=zip( embeds, metadata))

