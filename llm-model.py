from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel

class VertexAIVectorStore:
    def __init__(self, project_id, location):
        # You should replace the following index and endpoint names with your actual values
        self.index_endpoint_name = "projects/{}/locations/{}/indexEndpoints/5891412000042385408".format(project_id, location)
        self.index_name = "projects/{}/locations/{}/indexes/2680908415680643072".format(project_id, location)
        self.gen_ai_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=self.index_endpoint_name)
        self.gen_ai_index = aiplatform.MatchingEngineIndex(index_name=self.index_name)

    def search(self, input_text, k=3):
        # Assuming you have a model initialized somewhere
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

        embedding_vec = model.get_embeddings([input_text])[0].values

        # Find neighbors using vector search
        neighbors = self.gen_ai_index_endpoint.find_neighbors(
            deployed_index_id="gen_ai_deployed_index",
            queries=[embedding_vec],
            num_neighbors=k,
        )[0]

        for nb in neighbors:
            print("id: " + nb.id + " | text: " + df.iloc[int(nb.id)]["text"] + " | dist: " + str(nb.distance))
