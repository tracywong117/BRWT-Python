import numpy as np
from collections import deque

class TreeNode:
    """
    A class to represent a node in the binary tree.
    Each node stores an index vector and pointers to its left and right children.
    """
    def __init__(self, index_vector):
        self.index_vector = index_vector  # The binary index vector
        self.left = None  # Left child
        self.right = None  # Right child

class BRWT:
    """
    Binary Representation Wavelet Tree (BRWT) class for efficient storage and querying of binary matrices.
    """
    def __init__(self, matrix):
        """
        Initialize the BRWT with the input binary matrix.
        """
        self.column_num = matrix.shape[1] 
        self.root = None  # Root of the binary tree
        self.build_tree(matrix)  # Build the binary tree structure

    def build_tree(self, matrix):
        """
        Build the BRWT tree as a binary tree using BFS.
        """
        
        # Initialize the root of the tree
        self.root = TreeNode(np.bitwise_or.reduce(matrix, axis=1))  # Root index vector

        # Use a queue to process nodes in BFS order
        queue = deque([(matrix, self.root)])  # (current_matrix, parent_node)

        while queue:
            current_matrix, parent_node = queue.popleft()

            # Stop splitting if there's only one column
            if current_matrix.shape[1] == 1:
                continue

            # Split the matrix into left and right halves
            mid = (current_matrix.shape[1] + 1) // 2 # to make sure the left matrix has more columns than the right matrix
            left_matrix = current_matrix[:, :mid]
            right_matrix = current_matrix[:, mid:]

            # Prune rows based on the parent index vector
            active_rows = parent_node.index_vector > 0

            # Compute left and right index vectors
            left_index_vector = np.bitwise_or.reduce(left_matrix[active_rows], axis=1) if left_matrix.shape[1] > 0 else None
            right_index_vector = np.bitwise_or.reduce(right_matrix[active_rows], axis=1) if right_matrix.shape[1] > 0 else None

            # Create child nodes and attach them to the parent node
            if left_index_vector is not None:
                parent_node.left = TreeNode(left_index_vector)
                queue.append((left_matrix[active_rows], parent_node.left))

            if right_index_vector is not None:
                parent_node.right = TreeNode(right_index_vector)
                queue.append((right_matrix[active_rows], parent_node.right))

    def print_tree_preorder(self, node=None, level=0):
        """
        Preorder traversal to print the tree structure.
        """
        if node is None:
            node = self.root  # Start from the root

        print(f"{'  ' * level}Level {level}: {node.index_vector.astype(int)}")

        if node.left:
            self.print_tree_preorder(node.left, level + 1)
        if node.right:
            self.print_tree_preorder(node.right, level + 1)

    def compute_storage(self):
        """
        Compute and print the memory usage of the original matrix and the BRWT structure.
        """
        
        # Memory usage of the BRWT tree (sum of all index vectors)
        brwt_size = self._compute_tree_memory(self.root)

        print(f"BRWT Memory Usage: {brwt_size} bytes")
        print(f"Compression Ratio: {brwt_size / original_size:.2f}")

    def _compute_tree_memory(self, node):
        """
        Recursively compute the memory usage of a tree.
        """
        if node is None:
            return 0

        # Memory for the current node's index vector
        current_size = node.index_vector.nbytes

        # Add memory of the left and right subtrees
        left_size = self._compute_tree_memory(node.left)
        right_size = self._compute_tree_memory(node.right)

        return current_size + left_size + right_size

    def fill_skipped_positions_in_place(self, parent, child):
        child_idx = 0  # Pointer for the child array

        for i in range(len(parent)):  # Iterate through the parent array
            if parent[i] == 1:  # Replace only if position in parent is 1
                parent[i] = child[child_idx]
                child_idx += 1  # Move to the next child value

        return parent  # Modified parent is the result

    def reconstruct_column(self, column_index):
        """
        Reconstruct a specific column from the BRWT tree using the binary tree structure.
        """
        current_node = self.root
        parent_index = current_node.index_vector.copy() # why copy? because we are going to modify it in place

        start, end = 0, self.column_num - 1
        
        # Binary search path to determine left/right traversal and find the leaf node
        while start != end:
            mid = (start + end) // 2
            if column_index <= mid: # left child has more columns than right child
                direction = "left"
                end = mid
                current_node = current_node.left
            else:
                direction = "right"
                start = mid + 1
                current_node = current_node.right

            child_index = current_node.index_vector
            
            parent_index = self.fill_skipped_positions_in_place(parent_index, child_index) # it is modified child_index but parent_index of next iteration so here directly assign to parent_index
            
        return parent_index


# Example Usage
if __name__ == "__main__":
    # Example binary matrix
    n_rows, n_cols = 10, 10
    density = 0.1
    
    binary_matrix = np.zeros((n_rows, n_cols), dtype=bool)
    
    # Randomly set 1s in each column
    ones_in_col = int(n_rows * density)
    print("Ones in each column:", ones_in_col)
    for i in range(n_cols):
        random_rows = np.random.choice(n_rows, ones_in_col, replace=False)
        binary_matrix[random_rows, i] = 1
    
    # # Randomly set 1s in each row
    # ones_in_row = int(n_cols * density)
    # print("Ones in each row:", ones_in_row)
    # for i in range(n_rows):
    #     random_cols = np.random.choice(n_cols, ones_in_row, replace=False)
    #     binary_matrix[i, random_cols] = 1
    
    # binary_matrix = np.array([
    #     [0, 1, 0, 1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1, 0, 0, 1],
    #     [0, 0, 0, 0, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 0, 0, 0, 0]
    # ])

    print("Original Matrix:")
    print(binary_matrix.astype(int))
    
    # Build the BRWT
    brwt = BRWT(binary_matrix)

    # Print the tree structure
    print("\nTree Structure:")
    brwt.print_tree_preorder()

    # Storage comparison
    print("\nStorage Comparison:")
    original_size = binary_matrix.nbytes
    print(f"Original Matrix Memory Usage: {original_size} bytes")
    brwt.compute_storage()

    # for i in range(n_cols):
    #     reconstructed_column = brwt.reconstruct_column(i)
    #     print(f"Column {i} Reconstruction Matches:", np.array_equal(reconstructed_column, binary_matrix[:, i]))

                                                                                    