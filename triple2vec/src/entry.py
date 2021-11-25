import math
import networkx as nx
import matplotlib.pyplot as plt
from models.Triple import Triple
import random
from gensim.models import Word2Vec
from multiprocessing import cpu_count

settings = {
    "file_path": r"C:\Users\zpengc\Desktop\karate dataset",
    "directed": False,
    "a": 1.0/3,
    "b": 1.0/3,
    "c": 1.0/3,
    "p": 1.0,
    "q": 1.0,
    "num_walks": 10,
    "walk_length": 100,
    "num_negative_samples": 10,
    "window_size": 10,
    "dimension": 128,
    "workers": 8,
    "iteration": 10,
    "output": "./results/embedding_file",
}


def init_graph(file_path):
    original_graph = nx.Graph()
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
    for line in lines:
        if line is None or len(line) < 2:
            return
        line = line.split(" ")
        from_node = str(line[0])
        to_node = str(line[1])
        if len(line) >= 3:
            original_graph.add_edge(from_node, to_node, predicate=str(line[2]))
        else:
            original_graph.add_edge(from_node, to_node, predicate="NULL")
    f.close()
    return original_graph


def get_triple_line_graph(original_graph, directed):
    if directed:
        triple_line_graph = nx.DiGraph()
    else:
        triple_line_graph = nx.Graph()
    edges = original_graph.edges
    for edge in edges:
        triple_line_graph.add_node(Triple(from_node=edge[0], to_node=edge[1],
                                          predicate=original_graph.get_edge_data(edge[0], edge[1])["predicate"]))
    nodes = triple_line_graph.nodes
    print("computing weight for triple line graph")
    centrality_dict = nx.current_flow_betweenness_centrality(original_graph)
    for triple1 in nodes:
        for triple2 in nodes:
            if triple2 != triple1 and connect(triple1, triple2):
                triple_line_graph.add_edge(triple1, triple2, weight=compute_weight(centrality_dict, triple_line_graph, triple1, triple2,
                                                                                   directed))
    return triple_line_graph


def connect(t1, t2):
    if t1.from_node == t2.from_node or t1.from_node == t2.to_node or t1.to_node == t2.to_node or t1.to_node == t2.from_node:
        return True
    return False


def compute_weight(centrality_dict, triple_line_graph, triple1, triple2, directed):
    if directed:
        c = 0
        for triple in triple_line_graph.nodes:
            if triple.predicate == triple1.predicate or triple.predicate == triple2.predicate:
                c += 1
        return math.log(1+c)
    else:
        i = triple1.from_node
        j = triple1.from_node
        k = triple2.to_node if (triple2.from_node == triple1.from_node or triple2.from_node == triple1.to_node)\
            else triple2.from_node
        return settings["a"] * centrality_dict[i] + settings["b"] * centrality_dict[j] + settings["c"] * centrality_dict[k]


def get_walks(triple_line_graph):
    walks = []
    nodes = triple_line_graph.nodes
    for start_node in nodes:
        walk = [start_node]
        current = start_node
        while len(walk) < settings["walk_length"]:
            neighbors = list(triple_line_graph.neighbors(current))
            if len(neighbors) == 0:
                break
            current = random.choice(neighbors)
            walk.append(str(current))
        walks.append(walk)

    return walks


def learn_embeddings(walks):
    model = Word2Vec(vector_size=settings["dimension"], window=settings["window_size"], sg=1, min_count=0, hs=0, negative=settings["num_negative_samples"], compute_loss=True, workers=cpu_count())
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=settings["iteration"])
    word_vec = model.wv
    return word_vec


def main():
    print("getting original graph")
    original_graph = init_graph(settings["file_path"])
    position = nx.spring_layout(original_graph)  # graph的形状动态生成，所以需要固定
    edge_labels = nx.get_edge_attributes(original_graph, 'predicate')
    nx.draw(original_graph, pos=position, with_labels=True)
    nx.draw_networkx_edge_labels(original_graph, pos=position, edge_labels=edge_labels)
    plt.savefig("./results/original_graph.png")
    # plt.show()

    print("getting triple line graph")
    triple_line_graph = get_triple_line_graph(original_graph, False)
    position = nx.spring_layout(triple_line_graph)
    edge_labels = nx.get_edge_attributes(triple_line_graph, 'weight')
    nx.draw(triple_line_graph, pos=position, with_labels=True)
    nx.draw_networkx_edge_labels(triple_line_graph, pos=position, edge_labels=edge_labels)
    plt.savefig("./results/triple_line_graph.png")
    # plt.show()

    print("simulating random walks")
    walks = get_walks(triple_line_graph)
    print(len(walks))

    print("computing embedding given walks")
    emb = learn_embeddings(walks)
    print(len(emb))


if __name__ == '__main__':
    main()
