# Attention Mechanisms: Decoding the Mathematical Heart of Transformers

## Overview
This project provides a ground-up implementation of the **Scaled Dot-Product Attention** mechanism, the core engine behind modern LLMs like GPT-4 and Claude. By stripping away high-level abstractions, this notebook explores how specific mathematical operations—such as scaling, masking, and dot-product similarity—allow a model to synthesize context and resolve linguistic ambiguity.

## Methods
To investigate the mechanics of attention, I utilized the following approaches:
* **Modular Implementation:** Built the attention formula using PyTorch tensors, allowing for independent testing of the scaling and masking components.
* **Variance Stress-Testing:** Simulated high-dimensional environments ($d_k=512$) to observe the statistical "peaking" effect in the Softmax function.
* **Semantic Probing:** Manually defined feature-rich embeddings for homonyms (e.g., "Bank") to verify that the attention mechanism functions as a dynamic semantic filter.

## Conceptual Architecture
The implementation follows the standard Transformer attention pipeline, structured into four distinct experimental modules:
1.  **Input Projection:** Simulates the transformation of input embeddings into **Query (Q)**, **Key (K)**, and **Value (V)** matrices.
2.  **Compatibility Function:** Computes the raw similarity scores via $QK^T$.
3.  **The Scaling & Masking Layer:** * **Scaling:** Normalizes scores by $1/\sqrt{d_k}$ to stabilize gradients.
    * **Masking:** Injects $-1e9$ to "cloak" invalid tokens (Padding) before the exponential step.
4.  **Distribution Layer:** Applies **Softmax** to convert scores into a probabilistic distribution, which is then used to compute the weighted sum of the Values.

## Key Findings: Final Submission Summary
Through the experiments conducted in `lab_3.ipynb`, I have identified three critical behaviors of the attention mechanism:

### 1. The Necessity of Scaling for Gradient Health
My variance analysis demonstrated that in high dimensions ($d_k=512$), the raw dot product produces scores with massive variance. 
Softmax output where one token receives $100\%$ attention, effectively "killing" the gradients for all other tokens. Scaling the scores by $\sqrt{d_k}$ reduced the variance from $\approx 512.0$ to $\approx 1.0$, maintaining a "smooth" distribution essential for backpropagation.

### 2. Semantic Disambiguation via Feature Alignment
Using the **"River-Water-Bank"** experiment, I proved that attention is not just about keyword matching but **semantic alignment**. By defining a **Nature/Shore** feature dimension, the model successfully ignored the **Financial** definition of "Bank." This confirms that attention acts as a **Word Sense Disambiguation (WSD)** tool, "coloring" a word's vector with the meaning of its neighbors.

### 3. Computational Trade-offs of the $O(n^2)$ Bottleneck
While Multi-Head Attention allows for diverse "detective" perspectives (capturing grammar and time simultaneously), my analysis highlights its quadratic cost. The attention matrix grows by the square of the sequence length, identifying a critical limitation for processing long-form documents and justifying the industry's shift toward sparse attention methods.

## How to Run
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/attention-mechanisms.git](https://github.com/your-username/attention-mechanisms.git)
    cd attention-mechanisms
    ```
2.  **Install Requirements:**
    ```bash
    pip install torch matplotlib seaborn
    ```
3.  **Execute the Analysis:**
    Open `lab_3.ipynb` in your preferred environment (VS Code or Jupyter) and run all cells.

## Requirements
* **PyTorch 2.0+**: For tensor operations and `masked_fill` logic.
* **Matplotlib & Seaborn**: For generating the heatmaps found in the `output/` folder.
* **Python 3.9+**