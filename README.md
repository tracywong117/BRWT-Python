# Binary Representation Wavelet Tree (BRWT)

This implementation is based on a data structure introduced in the paper:

**"Sparse Binary Relation Representations for Genome Graph Annotation"**  
*J Comput Biol. 2020 Apr;27(4):626-639.*  
Authors: Mikhail Karasikov, Harun Mustafa, Amir Joudaki, Sara Javadzadeh-No, Gunnar Rätsch, André Kahles  

## Description

The **Binary Representation Wavelet Tree (BRWT)** is the data structure designed for efficient representation and querying of sparse binary matrices. It enables:

- **Efficient Compression**: Reduces the memory footprint of the matrix.
- **Rich Query Functionality**: Enables fast reconstruction of matrix columns.
- **Space Efficiency**: Optimized for sparse binary relations.

## Features

1. **Tree-Based Matrix Representation**:
   - Compresses the matrix into a wavelet tree structure.

2. **Efficient Column Reconstruction**:
   - Reconstructs any column in the matrix without decompressing the entire structure.

3. **Memory Usage Comparison**:
   - Compares the memory usage of the original matrix and the BRWT structure.

## Dependencies

- **Python 3.7+**
- **NumPy**

## Reference

Karasikov, M., Mustafa, H., Joudaki, A., Javadzadeh-No, S., Rätsch, G., & Kahles, A. (2020).  
**"Sparse Binary Relation Representations for Genome Graph Annotation."**  
J Comput Biol. 2020 Apr;27(4):626-639.  
DOI: [10.1089/cmb.2019.0324](https://doi.org/10.1089/cmb.2019.0324)  


