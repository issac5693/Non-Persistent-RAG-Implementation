import os
os.environ['OPENAI_API_KEY'] = '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'
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

from detectron2.config import get_cfg
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu' #GPU is recommended

text_folder = 'doc'
loaders = [UnstructuredPDFLoader(os.path.join(text_folder, fn)) for fn in os.listdir(text_folder)]

index = VectorstoreIndexCreator().from_loaders(loaders)

query = "who is doctor dun yat sen"
print(index.query(query))