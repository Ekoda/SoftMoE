This project is an implementation of the Vision Transformer (Dosovitskiy et al., 2021), but with the twist of using multi-query attention as opposed to standard attention for improved performance.

The example.ipynb contains a training run and an inference on the mnist dataset.

---
## What is a Vision Transformer?
The Vision Transformer (ViT) proposed by the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2021) demonstrated that Transfromers could achieve state of the art results in computer vision. This is done by essentially breaking down an image into a sequence of fixed-size patches (like "words" in NLP), linearly projecting these patches into embedding vectors, and feeding them into a standard Transformer encoder.

More specifically we take an image, $\mathbf{x}$

$$
\mathbf{x} \in \mathbb{R}^{H \times W \times C}
$$

and reshape it into a sequence of flattened 2D patches, $\mathbf{x}_p$

$$
\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}
$$

H and W represent the resolution of the image, while C is the number of channels which represent color (3 in RGB). (P, P) is the resolution of each image patch, meaning that the total number of patches (or, the sequence length of the picture) can be represented by

[!Vision transformer flowchart](https://github.com/Ekoda/VisionTransformer/blob/main/ViT.png)

$$
N = \frac{HW}{P^2}
$$

These patches can then be mapped to vector fitting the model dimension trough a linear projection, also called a patch embedding. From there the data flows like in a standard transformer, long live the transformer.

---
## Requirements
- Python 3.10 or later

Dependencies are listed in the `requirements.txt` file. To install these dependencies, navigate to the project directory and run the following command:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

---
## References
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ArXiv, cs.CV. https://arxiv.org/abs/2010.11929
