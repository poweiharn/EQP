import math
import random
import copy

sub_tree_identifier = {'00','01','10','11'}
branch_id_list = list(sub_tree_identifier)

class Node:
    def __init__(self, data, sensor_list, operator_list, depth):
        self.is_operator = False
        self.is_sensor = False
        self.data = data
        self.depth = depth
        self.branch = {}
        for branch_id in branch_id_list:
            self.branch[branch_id] = None
        if data in sensor_list:
            self.is_sensor = True
        if data in operator_list:
            self.is_operator = True

    def tree_size(self):
        # Return tree size in nodes
        if self.is_sensor:
            return 1
        count = 1
        for branch_id in branch_id_list:
            count = count + self.branch[branch_id].tree_size()

        return count

    def set_depth(self, depth):
        self.depth = depth
        for branch_id in branch_id_list:
            if self.branch[branch_id]:
                self.branch[branch_id].set_depth(depth+1)


    def get_tree_depth(self):
        # Return maximum tree depth
        if self.is_sensor:
            return self.depth

        count = self.depth
        for branch_id in branch_id_list:
            count = max(count, self.branch[branch_id].get_tree_depth())
        return count

    def print_parse_tree(self):
        node_string = ""
        for i in range(0,self.depth):
            node_string = node_string + "|"
        node_string = node_string + self.data
        print(node_string)
        for branch_id in branch_id_list:
            if self.branch[branch_id]:
                self.branch[branch_id].print_parse_tree()

    def parse_tree(self, node_string):
        if self.is_operator:
            for branch_id in branch_id_list:
                if self.branch[branch_id]:
                    node_string = node_string +self.branch[branch_id].parse_tree("")
            node_string = "("+node_string+")"
        else:
            node_string = node_string + self.data
        return node_string

    def get_sub_tree_uniform_nodes_selection(self, count):
        count[0] = count[0] - 1
        if count[0] == 0:
            return self
        else:
            result = None
            r_branch_id_list = copy.copy(branch_id_list)
            random.shuffle(r_branch_id_list)
            for branch_id in r_branch_id_list:
                if self.branch[branch_id] and count[0]>0:
                    result = self.branch[branch_id].get_sub_tree_uniform_nodes_selection(count)
            return result

    def get_sub_tree_uniform_depth_selection(self):
        sub_tree_array = []
        h = random.randint(0, self.get_tree_depth())
        self.get_current_level(h,sub_tree_array)
        return random.sample(sub_tree_array,k=1)[0]

    def get_current_level(self,level,tree_array):
        if level == 0:
            tree_array.append(self)
        elif level > 0:
            for branch_id in branch_id_list:
                if self.branch[branch_id]:
                    self.branch[branch_id].get_current_level(level-1,tree_array)

    def glue_sub_tree(self, glue_depth, mate):
        # Suppose glue_depth = 2
        # Randomly glue subtree to nodes selected from current tree depth: 0, 1, 2
        sub_tree_array = []
        for i in range(0, glue_depth + 1):
            self.get_current_level(i, sub_tree_array)

        node = random.sample(sub_tree_array, k=1)[0]

        # Glue subtree to selected node
        mate.set_depth(node.depth)
        node.data = mate.data
        node.depth = mate.depth
        for branch_id in branch_id_list:
            node.branch[branch_id] = mate.branch[branch_id]
        node.is_sensor = mate.is_sensor
        node.is_operator = mate.is_operator
        if hasattr(mate, 'shift_offset_x'):
            node.shift_offset_x = mate.shift_offset_x
        if hasattr(mate, 'shift_offset_y'):
            node.shift_offset_y = mate.shift_offset_y


def tree_full(depth, max_tree_depth, sensors, operators):
    if depth < max_tree_depth:
        node = Node(random.choice(operators), sensors, operators, depth)
        for branch_id in branch_id_list:
            node.branch[branch_id] = tree_full(depth + 1, max_tree_depth, sensors, operators)
    else:
        node = Node(random.choice(sensors), sensors, operators, depth)
    return node


def tree_grow(depth, max_tree_depth, sensors, operators):
    if depth < max_tree_depth:
        # 0.5 probability for operators and sensors
        if round(random.random()) == 0:
            node = Node(random.choice(operators), sensors, operators, depth)
        else:
            node = Node(random.choice(sensors), sensors, operators, depth)

        if node.is_operator:
            for branch_id in branch_id_list:
                node.branch[branch_id] = tree_grow(depth+1, max_tree_depth, sensors, operators)
    else:
        node = Node(random.choice(sensors), sensors, operators, depth)
    return node

class qtreeGenotype():
    def __init__(self):
        self.fitness = None
        self.gene = None

    def recombine(self, mate, init_args, **kwargs):
        child = self.__class__()
        child.gene = copy.deepcopy(self.gene)
        # Get subtree from mate
        mate_gene = copy.deepcopy(mate.gene)
        seq_length = init_args['seq_length']
        depth_limit = init_args['depth_limit']
        for i in range(seq_length):
            child_tree = child.gene[i]
            mate_tree = mate_gene[i]
            cutting_point = random.randint(1, mate_tree.tree_size())
            mate_sub_tree = mate_tree.get_sub_tree_uniform_nodes_selection([cutting_point])
            mate_sub_tree.set_depth(0)
            # Perform crossover
            max_glue_depth = (depth_limit-i) - mate_sub_tree.get_tree_depth()
            child_tree.glue_sub_tree(max_glue_depth, mate_sub_tree)
        return child

    def print(self):
        # TODO: return a string representation of self.gene
        #       (see assignment description doc for more info)
        string = ""
        for q_tree in self.gene:
            string = string + q_tree.parse_tree("") + '\n'

        return string

    def mutate(self, init_args, **kwargs):
        # TODO: mutate gene of copy
        sensors = list(init_args['leaf_nodes'])
        operators = list(init_args['internal_nodes'])

        child = self.__class__()
        child.gene = copy.deepcopy(self.gene)

        seq_length = init_args['seq_length']
        depth_limit = init_args['depth_limit']
        for i in range(seq_length):
            sub_tree = tree_grow(0, depth_limit-i, sensors, operators)
            max_glue_depth = (depth_limit-i) - sub_tree.get_tree_depth()
            child.gene[i].glue_sub_tree(max_glue_depth, sub_tree)
        return child

    @classmethod
    def initialization(cls, mu, init_args, **kwargs):
        population = [cls() for _ in range(mu)]
        seq_length = init_args['seq_length']
        depth_limit = init_args['depth_limit']
        sensors = list(init_args['leaf_nodes'])
        operators = list(init_args['internal_nodes'])
        # population using ramped half-and-half
        mid_index = len(population) // 2
        for i in range(len(population)):
            if i < mid_index:
                population[i].gene = [tree_grow(0, depth_limit-j,sensors, operators) for j in range(seq_length)]
            else:
                population[i].gene = [tree_full(0, depth_limit-j, sensors, operators) for j in range(seq_length)]

        return population