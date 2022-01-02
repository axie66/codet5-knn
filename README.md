# CodeT5 + kNN for Code Generation

This repository contains code for code generation on the
[Concode](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/text-to-code) dataset using
CodeT5 augmented with a nearest neighbor language model component.

The code currently only supports creating a kNN datastore from Concode training
data; while this is already fairly large with 100k examples, I plan to
augment it with CodeSearchNet examples as well.
