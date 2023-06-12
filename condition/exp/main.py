from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')

with open('evaluation_set.txt', 'r') as f:
    fileContent = f.readlines()

sentences1 = []
sentences2 = []

for i in range(0, len(fileContent), 3):
    sentences1.append(fileContent[i+1].strip())
    sentences2.append(fileContent[i+2].strip())

embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

cosine_scores = util.cos_sim(embeddings1, embeddings2)

for i in range(len(sentences1)):
    print(sentences1[i])
    print(sentences2[i])
    print('Score: {:.4f}'.format(cosine_scores[i][i]))
