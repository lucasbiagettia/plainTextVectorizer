from pinecone_db_writer import PineconeDbWritter
from text_opener import read_files_in_folder
index_name = 'trial'
files = read_files_in_folder('files')



pinecone = PineconeDbWritter()
pinecone.Index(index_name)
pinecone.write(files)