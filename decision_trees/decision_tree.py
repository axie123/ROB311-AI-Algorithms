import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS


def dt_entropy(goal, examples):  # Correct
    """
    Compute entropy over discrete random variable for decision trees.
    Utility function to compute the entropy (which is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """

    # INSERT YOUR CODE HERE.
    entropy = 0.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0

    # Rewrite this code by yourself:

    label_counter = [0] * len(goal[1])  # The frequency of each goal state.
    p_v = [0] * len(goal[1])  # The prior probability of each goal state.

    for i in range(len(examples)):  # Goes through each example.
        for j in range(len(goal[1])):  # Goes through each case of goal state.
            if examples[i, -1] == j:
                label_counter[j] += 1  # Adds a count to the designated goal state.

    for k in range(len(p_v)):
        if len(examples) != 0:
            p_v[k] = label_counter[k]/len(examples)  # Calculates the prior probability of the goal states.

    for x in range(len(p_v)):
        if p_v[x] != 0:
            entropy -= p_v[x] * np.log2(p_v[x])  # Calculates the total entropy of the dataset.

    return entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):  # Fixed (Correct)
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.

    cond_entropy = 0.0

    p_x = [0] * len(attribute[1])  # Prior probability of the a particular attribute label showing up.

    for i, ex in enumerate(examples):
        for j in range(len(attribute[1])):
            if ex[col_idx] == j:
                p_x[j] += 1  # Records the frequency of each attribute label for the attribute.
    p_x = [x / len(examples) for x in p_x]  # Calculates the prior probability of each label for the attribute.

    H_VX = [0] * len(attribute[1])  # Probabilistic entropy of each goal state dependent on the attribute label.
    for k in range(len(attribute[1])):
        attr_examples = examples[np.where(examples[:, col_idx] == k)]  # Getting a subset of examples with the particular label for the selected attribute.
        H_VX[k] = dt_entropy(goal, attr_examples)  # Calculates the probabilistic entropy from the subset of examples with the label.

    for x in range(len(attribute[1])):
        if H_VX[x] != 0:
            cond_entropy += p_x[x] * H_VX[x]  # Calculates the conditional entropy from the prior and the probabilistic entropy.

    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """
    # INSERT YOUR CODE HERE.

    # The information gain is the difference between the entropy and conditional entropy.
    info_gain = dt_entropy(goal, examples) - dt_cond_entropy(attribute, col_idx, goal, examples)

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples): # Correct
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    intrinsic_info = 0.

    sub_ratio = [0] * len(attribute[1])  # Values for ratio q = (p_k + n_k)/(p + n)
    log_vals = [0] * len(attribute[1])  # Values for log2(q)
    p_n = len(examples)

    for i in range(len(attribute[1])):
        attr_examples = examples[np.where(examples[:, col_idx] == i)]  # Getting a subset of examples with the particular label for the selected attribute.
        sub_ratio[i] = len(attr_examples) / p_n  # Calculates ratio q = (p_k + n_k)/(p + n).
        q = sub_ratio[i]  #
        if q == 0:
            log_vals[i] = 0
        else:
            log_vals[i] = np.log2(q)  # Calculates the values of log2(q).

    for j in range(len(log_vals)):
        intrinsic_info -= sub_ratio[j] * log_vals[j]  # Calculates the intrinsic information as the product of q and log2(q).

    return intrinsic_info


def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Avoid NaN examples by treating 0.0/0.0 = 0.0

    # The gain ratio is the quotient between the information and the intrinsic information.
    gain = dt_info_gain(attribute, col_idx, goal, examples)
    intrinsic_info = dt_intrinsic_info(attribute, col_idx, examples)

    if intrinsic_info == 0:
        gain_ratio = 0
    else:
        gain_ratio = gain / intrinsic_info

    return gain_ratio


def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.
    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """

    node = None
    labels = examples[:, -1]

    # 1. Do any examples reach this point?
    if examples.shape[0] == 0:
        return TreeNode(parent, attributes, examples, True, plurality_value(goal, parent.examples))

    # 2. Or do all examples have the same class/label? If so, we're done!
    elif sum(examples[:, -1]) / len(examples[:, -1]) == 1 or sum(examples[:, -1]) == 0:
        return TreeNode(parent, attributes, examples, True, examples[0][examples.shape[1] - 1])

    # 3. No attributes left? Choose the majority class/label.
    elif len(attributes) == 0:
        return TreeNode(parent, attributes, examples, True, plurality_value(goal, examples))

    # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!
    else:
        # Evaluate the score of each attribute
        importance_cost = float("-inf")
        opt_attr = None  # Stores the most optimal attribute.
        opt_idx = -4  # Index of most optimal attribute.

        for c_idx, attr in enumerate(attributes):  # Goes through all the attributes:
            new_importance = score_fun(attr, c_idx, goal, examples)  # Figures out the importance of the attribute based on the score function.
            if new_importance > importance_cost:
                importance_cost = new_importance
                opt_attr = attr  # Updates the most important attribute.
                opt_idx = c_idx  # Updates the index of the important attribute.

        # Create a new internal node using the best attribute
        curr_node = TreeNode(parent, opt_attr, examples, False, 0)

        # Now, recurse down each branch (operating on a subset of examples below).
        for c_val in range(len(opt_attr[1])):
            attr_subclass = examples[np.where(examples[:, opt_idx] == c_val)]  # Getting a subset of examples with the particular label for the optimal attribute.
            attr_subclass = np.delete(attr_subclass, opt_idx, axis=1)  # Updates a subset of examples without the labels of the most optimal attribute.
            attributes_new = np.delete(attributes, opt_idx, axis=0)  # Updates a new atrribute list without the most optimal attribute.
            curr_subtree = learn_decision_tree(curr_node, attributes_new, goal, attr_subclass, score_fun)  # Recurses down the branches by creating a subtree until termination.

            # You should append to node.branches in this recursion
            curr_node.branches.append(curr_subtree)

    return curr_node


def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """
    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root


if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])


    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(None, attributes, goal, examples, dt_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
