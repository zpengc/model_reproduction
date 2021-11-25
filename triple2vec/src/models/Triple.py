

class Triple:
    def __init__(self, from_node, to_node, predicate):
        self.from_node = from_node
        self.to_node = to_node
        self.predicate = predicate

    def __repr__(self):
        return str(self.from_node) + "->" + str(self.predicate) + "->" + str(self.to_node)

    def __str__(self):
        return str(self.from_node) + "->" + str(self.predicate) + "->" + str(self.to_node)

