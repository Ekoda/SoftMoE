This project is an implementation of the Vision Transformer (Dosovitskiy et al., 2021), but with the twist of using multi-query attention as opposed to standard attention for improved performance.

The example.ipynb contains a training run and an inference on the mnist dataset.

---
## What is a Vision Transformer?
The Vision Transformer (ViT) proposed by the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2021) demonstrated that Transfromers could achieve state of the art results in computer vision. This is done by essentially breaking down an image into a sequence of fixed-size patches (like "words" in NLP), linearly projecting these patches into embedding vectors, and feeding them into a standard Transformer encoder.

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