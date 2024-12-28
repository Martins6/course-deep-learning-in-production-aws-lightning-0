# course-deep-learning-in-production-aws-lightning-0

This is a open course on how to deploy Deep Learning models using PyTorch, PyTorch
Lightning, AWS and Terraform.

## Introduction

This repository is divided into folders that are the chapters of each course. Each folder
contains a full Python project so you can code along, alongside notebooks, scripts and
markdown files.

I like to learn and to teach using a problem-solution framework. We have a problem and
we need a solution. As the course focuses on establishing a strong foundation in scalable
 AI development and deployment with an emphasis on PyTorch and PyTorch Lightning, I'd
 propose the following breakdown of problems.

1. How to oganise complex PyTorch code for better reproducibility and readability.
2. Efficient management of training loops and inference steps across multiple GPUs and
accelerators.
3. Framing of ML model into an ML API.
4. How to deploy an ML API using AWS resources.

What techniques would you learn, then?

* Object Oriented Programming for Neural Networks with Pytorch and PyTorch Lightning.
* GPU usage with Pytorch and PyTorch Lightning
* Logging experiments in an efficient manner.
* Break down of an AI project into three different pipelines: feature engineering, training and deployment.
* Parallelization being applied for large scale data processing.
* Infrastructure as a Code for an ML application.

And what would be the main technologies used?

* Pytorch
* Pytorch Lightning (Lightning, LitServe libraries)
* Tensorboard
* Loguru
* Terraform

## Table of Contents

* [Chapter 1: PyTorch Lightning Fundamentals and Data Processing](chapter-1)

## Installation

All of the chapters Python projects are managed by `uv`, to install it check out
the [official Github repo](https://github.com/astral-sh/uv). Then, you just need to run:

```bash
uv sync
```

This will install all Python packages. To run the scripts just use:

```bash
uv run scripts/{SCRIPT NAME}.py
```

Other than that to run the notebooks, you need to use [Jupyter](https://jupyter.org/) framework.

Regarding AWS deployment, we are going to be using
[Terraform](https://github.com/hashicorp/terraform) to manage this for us via code.
Terraform is a great framework for Infrastructure as Code (IaC). Generically, we are
going to be using AWS services like S3, API Gateway, Lambda, ECS and Fargate.

## Contributions

I welcome contributions from anyone who is interested in improving the chapters.
Please feel free to open an issue or a pull request with your suggestions.
