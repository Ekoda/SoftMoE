Apologies to any disappointed moe anime loves, this repository implements a soft mixture of experts visual transformer model as described in the "From Sparse to Soft Mixtures of Experts" paper (Puigcerver et al., 2023).

---
## What are Soft Mixture of Expert Architectures?
The Soft Mixture of Experts architecture seeks to address the limitations of MoE's such as token dropping, scaling difficulties, ineffective finetunining and training instability. This is done through passing different weighted combinations of the input tokens to each expert. This way each input token only fractionally activates all model parameters. 

Soft MoE models exist in between sparse MoE models, where each token is sent only to a corresponding expert, and dense models, where every token is passed to all experts. As the authors put it, the advantege of the soft architecture lies in it being able to avoid typicall routing alorithms used in sparse MoE's such as top-k which are not suited for hardware accelerators - causing soft models to perform well in terms of speed when compared to sparse models.

---
## How to use
The model config is implemented in the `config.py` file, which is initiated and passed on to tokenizer, model and the data preparation function if used. If `should_load` is specified, and a path is provided the model will load the weights and model config it. Likewise it will save both params and config in the specified path.

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
Puigcerver, J., Riquelme, C., Mustafa, B., & Houlsby, N. (2023). From Sparse to Soft Mixtures of Experts. arXiv preprint arXiv:2308.00951.