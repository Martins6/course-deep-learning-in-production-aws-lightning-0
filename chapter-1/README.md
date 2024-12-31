# Chapter 1: PyTorch Lightning Fundamentals and Data Processing

This is the first chapter for deep learning in production with PyTorch Lightning (PL) and
AWS. It covers essential concepts such as model training, validation, and testing, along
with data preprocessing techniques, all using the specific frameworks with PyTorch,
HuggingFace and Pytorch Lightning.

Everything I'm going to say here, was already taught in the
[documentations](https://lightning.ai/docs/pytorch/stable/levels/core_skills.html) of
Pytorch Lightning. I encourage you to go read it for a more in-depth explanation of all
of their components. This lesson is meant to be an overview focused on the core skills
required to work with PyTorch Lightning effectively.

## Table of Contents

1. **Introduction to PyTorch Lightning**
2. **Data Preprocessing in PyTorch Lightning**
3. **Model Training with PyTorch Lightning && HuggingFace Transformers**
4. **Validation and Testing with PyTorch Lightning**
5. **Best Practices for Model inference**

## 1. Introduction to PyTorch Lightning

PyTorch Lightning is a high-level library that simplifies the process of building,
training, validating, and deploying machine learning models in PyTorch. Pytorch being one
of the most popular deep learning frameworks in Python.

PL provides a set of tools and utilities that make it easier to develop and
deploy models, including support for distributed computing, model checkpointing, and
more. The great thing about PyTorch Lightning is that it abstracts away many of the
complexities, essentially it is a boilerplate that is quite flexible and easy to use.
It is very much into the [ZEN of Python](https://peps.python.org/pep-0020/) and it
encourages developers to write clean, maintainable and efficient code.

The way I like to divide PL is in three different parts, which are some of the classes
provided by the framework:

- **LightningModule**
- **Trainer**
- **DataModule**

### 1.1 LightningModule

This class contains the main logic of your model. It defines how the
model should be trained, validated, tested and also its inference process. The
LightningModule is divided into hooks that allow you to customize the behavior of the
model during all of the steps of a standard deep learning procedure. I'd advise you
to only learn about a hook as per-specific use-case.

The necessaries # TODO: ...

You can see an example of how to define a LightningModule in PL:

```python
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

class SimpleLinearModel(L.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)
```

As you can see there's a lot of boilerplate code that is provided by PL to make it
easier for us to write our models.