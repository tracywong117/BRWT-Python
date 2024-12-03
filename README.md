# Binary Representation Wavelet Tree (BRWT)

This implementation is based on concepts introduced in the paper:

**"Compact Binary Relation Representations with Rich Functionality"**  
*(arXiv:1201.3602v1 [cs.DS], 17 Jan 2012)*  
Authors: Jérémy Barbay, Francisco Claude, and Gonzalo Navarro  

---

## Description

The **Binary Representation Wavelet Tree (BRWT)** is a data structure designed for efficient representation and querying of sparse binary matrices. It enables:

- **Efficient Compression**: Reduces the memory footprint of the matrix.
- **Rich Query Functionality**: Enables fast reconstruction of matrix columns.
- **Space Efficiency**: Optimized for sparse binary relations.

---

## Features

1. **Tree-Based Matrix Representation**:
   - Compresses the matrix into a wavelet tree structure.

2. **Efficient Column Reconstruction**:
   - Reconstructs any column in the matrix without decompressing the entire structure.

3. **Memory Usage Comparison**:
   - Compares the memory usage of the original matrix and the BRWT structure.

---

## Dependencies

- **Python 3.7+**
- **NumPy**

---

## Reference

Barbay, J., Claude, F., & Navarro, G. (2012).  
**"Compact Binary Relation Representations with Rich Functionality."**  
arXiv:1201.3602v1 [cs.DS].

