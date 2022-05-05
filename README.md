# graph-neural-networks

## Notes

The `notes/` directory contains a bunch of theory notes collected throughout my PhD. They can be compiled using:

```bash
cd notes/

# PDF
pandoc --toc -o notes.pdf --pdf-engine=/Library/TeX/texbin/pdflatex --top-level-division=chapter --citeproc main.md maths/set-theory.md maths/linear-algebra.md maths/calculus.md maths/graph-theory.md references.md

# Website
pandoc --toc -o notes.html -s --citeproc --katex main.md maths/set-theory.md maths/linear-algebra.md maths/calculus.md maths/graph-theory.md references.md
```

## GNN Implementations

This repo contains a number of graph neural network implementations. The GNNs are implementated in a Jupyter Notebook and contain explanations alongisde the code. 

-  [Graph Convolutional Network (by hand)](graph-neural-networks/graph-convolutional-network.ipynb)