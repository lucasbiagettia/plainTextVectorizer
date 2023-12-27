from model import EmbeddingModelSingleton
import pinecone
import time


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
            api_key='571819e1-2c00-4ab3-a7b0-8690c0d0ae1a',
            environment='gcp-starter'
        )
        self._embed_model = EmbeddingModelSingleton()

    def create_index(self, index_name):
        embedded =        _embed_model.embed_documents("get dim")
        if index_name not in pinecone.list_indexes():
            self._index = pinecone.create_index(
                index_name,
                dimension=len(embedded[0]),
                metric='cosine'
            )
            
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1)

    def write(files):
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

