from model import EmbeddingModelSingleton
from text_opener import read_files_in_folder

files = read_files_in_folder('files')

embedder = EmbeddingModelSingleton()

for title, 
#embeddings = embedder.embed_documents(text_to_embed)



import os
import pinecone

# get API key from app.pinecone.io and environment from console
pinecone.init(
    api_key='571819e1-2c00-4ab3-a7b0-8690c0d0ae1a',
    environment='gcp-starter'
)

"""Now we initialize the index."""

import time

index_name = 'llama-2-rag'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

"""Now we connect to the index:"""

index = pinecone.Index(index_name)
index.describe_index_stats()

"""With our index and embedding process ready we can move onto the indexing process itself. For that, we'll need a dataset. We will use a set of Arxiv papers related to (and including) the Llama 2 research paper."""

# Importar las bibliotecas necesarias
import pandas as pd

# Leer el texto plano
texto_plano = contenido

# Dividir el texto plano en chunks
chunks = texto_plano.split("\n\n")

# Etiquetar los chunks
etiquetas = ["abstract"] * len(chunks)
etiquetas[0] = "introducción"
etiquetas[-1] = "conclusión"

# Guardar el dataset
df = pd.DataFrame({"chunk": chunks, "etiqueta": etiquetas})
data = df
df.head()

"""We will embed and index the documents like so:"""

#data = data.to_pandas()

batch_size = 32


for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]

    # Create unique identifiers
    ids = [f"{x['chunk']}-{x['etiqueta']}" for i, x in batch.iterrows()]

    # Extract text content
    texts = [x['chunk'] for i, x in batch.iterrows()]

    # Generate vector embeddings
    embeds = embed_model.embed_documents(texts)

    # Prepare metadata
    metadata = [
        {'text': x['chunk'],
         'title': x['etiqueta']} for i, x in batch.iterrows()
    ]

index.describe_index_stats()

"""## Initializing the Hugging Face Pipeline

The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:

* A LLM, in this case it will be `meta-llama/Llama-2-13b-chat-hf`.

* The respective tokenizer for the model.

We'll explain these as we get to them, let's begin with our model.

We initialize the model and move it to our CUDA-enabled GPU. Using Colab this can take 5-10 minutes to download and initialize the model.
"""

from torch import cuda, bfloat16
import transformers

model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()
print(f"Model loaded on {device}")

"""The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Llama 2 13B models were trained using the Llama 2 13B tokenizer, which we initialize like so:"""

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

"""Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code."""

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

"""Confirm this is working:

Now to implement this in LangChain
"""

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

"""We still get the same output as we're not really doing anything differently here, but we have now added **Llama 2 13B Chat** to the LangChain library. Using this we can now begin using LangChain's advanced agent tooling, chains, etc, with **Llama 2**.

## Initializing a RetrievalQA Chain

For **R**etrieval **A**ugmented **G**eneration (RAG) in LangChain we need to initialize either a `RetrievalQA` or `RetrievalQAWithSourcesChain` object. For both of these we need an `llm` (which we have initialized) and a Pinecone index — but initialized within a LangChain vector store object.

Let's begin by initializing the LangChain vector store, we do it like so:
"""

from langchain.vectorstores import Pinecone

text_field = 'text'  # field in metadata that contains text content

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

"""We can confirm this works like so:

Looks good! Now we can put our `vectorstore` and `llm` together to create our RAG pipeline.
"""

from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)

"""Let's begin asking questions! First let's try *without* RAG:"""

llm('Explicame que es el proletariado')

"""Hmm, that's not what we meant... What if we use our RAG pipeline?"""

rag_pipeline('Explicame que es el proletariado')

"""[texto del vínculo](https://)This looks *much* better! Let's try some more.

'result': ' In the Manifesto of the Communist Party, Karl Marx describes how capitalism divides society into two main classes: the bourgeoisie and the proletariat. The bourgeoisie refers to the modern capitalists who own the means of production and employ wage laborers. On the other hand, the proletariat consists of modern wage laborers who lack the means of production and must sell their labor to survive. This division leads to class antagonisms and conflict, driving historical change and revolution.'}
"""