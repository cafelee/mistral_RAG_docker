# llm_container/util/llm_rag.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from llama_cpp import Llama
from pydantic import PrivateAttr

class LlamaCppLLM(LLM):
    _llm: Llama = PrivateAttr()

    temperature: float = 0.5
    max_tokens: int = 512

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self._llm = Llama(model_path=model_path, n_ctx=4096)

    @property
    def _llm_type(self):
        return "llama_cpp"

    def _call(self, prompt, stop=None):
        output = self._llm(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        return output['choices'][0]['text'].strip()

def build_qa_pipeline():
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

    vectorstore = FAISS.load_local(
        folder_path="faiss_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    llm = LlamaCppLLM(model_path=model_path, temperature=0.5, max_tokens=512)

    prompt = PromptTemplate(
        template="""根據以下提供的資訊，回答問題。
資訊:
{context}

問題: {question}
答案:""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
