# 📈 Portfolio Optimization using Genetic Algorithms

## 🧠 Abstract

This project explores the application of **Genetic Algorithms (GAs)** for optimizing investment portfolios. Portfolio optimization involves selecting an ideal mix of assets to **maximize returns while minimizing risk**. Traditional approaches like the **Markowitz Mean-Variance model** struggle with large datasets and non-linear constraints.

Genetic Algorithms, inspired by the process of **natural selection**, provide a powerful alternative. In this project, **real-valued encoding** is used to represent portfolios, and the **Sharpe Ratio** serves as the fitness function. The GA evolves a population of portfolios using **selection, crossover, and mutation** operations. The results demonstrate that GAs can effectively discover optimized asset allocations and are a promising tool in financial decision-making.

---

## 📝 1. Introduction

Investment portfolio optimization is a cornerstone of financial decision-making, focusing on how to allocate capital among various assets to achieve a desirable balance between **risk and return**.

### ❌ Limitations of Traditional Methods
- Dependence on assumptions (e.g., normally distributed returns)
- Difficulty in handling **non-linear constraints**
- Inefficiency with **large datasets**

### ✅ Why Genetic Algorithms?
Genetic Algorithms (GAs) are a type of evolutionary algorithm inspired by **Darwinian principles of natural selection**. GAs:

- Don’t require **derivatives** or **convex functions**
- Handle **complex, multi-objective optimization**
- Can **encode real-world constraints** naturally

This project applies GAs to optimize asset allocation using historical stock data, focusing on maximizing the **Sharpe Ratio**, a measure of risk-adjusted return.

---

## 🔄 2. Genetic Algorithm (GA) Overview

### 2.1 Key Concepts
- **Chromosome**: Represents a portfolio (array of asset weights)
- **Gene**: Weight of an individual asset
- **Population**: A group of portfolios
- **Fitness Function**: Measures portfolio quality using the **Sharpe Ratio**

### 2.2 Why GA for Portfolio Optimization
- Suitable for **non-linear** and **multi-modal** objective functions
- Can work with **large and complex** search spaces
- Easily incorporates **constraints**, like budget limits or weight bounds

---

## ⚙️ 3. GA Components in This Project

### 3.1 Chromosome Representation
- **Real-valued encoding** of asset weights  
- Example for 4 assets: `[0.2, 0.3, 0.1, 0.4]`  
- Sum of weights = 1 (normalized)

### 3.2 Initialization
- Random weight vectors are generated  
- Weights are normalized so that the total sum = 1

### 3.3 Fitness Function
Based on the **Sharpe Ratio**:

\[
\text{Sharpe Ratio} = \frac{R_p - R_f}{\sigma_p}
\]

Where:
- \( R_p \): Portfolio return  
- \( R_f \): Risk-free rate  
- \( \sigma_p \): Portfolio standard deviation

### 3.4 Selection Method
- **Tournament Selection**  
  - Randomly selects a subset of portfolios  
  - Chooses the one with the highest fitness

### 3.5 Crossover Method
- **Arithmetic Crossover**  
  - Combines two parents using a **weighted average**  
  - Ensures resulting weights are still valid

### 3.6 Mutation Method
- **Random Mutation**  
  - Randomly adjusts one or more asset weights  
  - Weights are **re-normalized** to ensure sum = 1

### 3.7 Termination Criteria
- Fixed number of generations (e.g., **100**)
- Early stopping if **fitness plateaus** (no improvement over several generations)

