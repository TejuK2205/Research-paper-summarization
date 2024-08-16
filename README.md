# Research Paper Summarization and Comparison Tool

Welcome to the Research Paper Summarization and Comparison Tool! This Streamlit application enables users to upload research papers (PDF format), generate summaries based on user-defined prompts, and evaluate the generated summaries using various metrics. The tool leverages BART for summarization, ROUGE and BLEU scores for evaluation, and includes a basic human evaluation of the summaries.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)

## Project Overview

This tool allows users to:

- Extract text from a PDF research paper.
- Generate a summary based on a user-defined prompt using a pre-trained BART model.
- Evaluate the generated summary using ROUGE and BLEU scores.
- Perform a basic human evaluation of the summary's readability, coherence, and relevance.

## Features

- **PDF Text Extraction**:
  - Extracts and displays text from the uploaded PDF file.

- **Summary Generation**:
  - Uses the BART model to generate summaries based on user-defined prompts.
  
- **Evaluation Metrics**:
  - Computes ROUGE scores (`rouge1`, `rouge2`, `rougeL`).
  - Calculates BLEU score to assess translation quality.
  - Provides a basic human evaluation of the summary's readability, coherence, and relevance.

- **User Interaction**:
  - Allows users to input prompts for summarization.
  - Compares generated summaries with user-provided original abstracts.

## Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/research-paper-summarization.git
    ```

2. **Navigate to the Project Directory**:

    ```bash
    cd research-paper-summarization
    ```

3. **Install Dependencies**:

    Ensure you have Python 3.x installed. Then, install the required packages using pip:

    ```bash:
    - streamlit
    - PyPDF2
    - nltk
    - transformers
    - rouge_score
    - matplotlib
    ```

4. **Download the Model**:

    The BART model will be automatically downloaded by the `transformers` library when the application is run.

## Usage

1. **Run the Streamlit Application**:

    Execute the following command to start the Streamlit app:

    ```bash
    streamlit run research_paper_summary.py
    ```

2. **Access the Application**:

    Open your web browser and navigate to `http://127.0.0.1:8501/` to view and interact with the application.

## Code Structure

- `research_paper_summary.py`: Main application file containing the Streamlit interface, text extraction, summarization, and evaluation logic.
