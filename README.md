# CHALLENGE_1B

This project implements a semantic search and section extraction solution for PDF documents. It builds upon a PDF outline extraction utility (Challenge 1A) to identify relevant sections and sub-sections within PDFs based on a user-defined persona and job-to-be-done.

## 1. Approach

The core idea is to semantically understand user intent (defined by a persona and a job-to-be-done) and then rank sections within PDF documents based on their relevance to this intent.

1.  **PDF Outline Extraction**: First, the project leverages a PDF processing module to extract the hierarchical outline (table of contents) from input PDF documents.
2.  **Full Text Sectioning**: For each entry in the extracted outline, the system attempts to find the corresponding full text block(s) within the PDF, creating discrete "sections" with their title and content.
3.  **Semantic Query Generation**: A detailed semantic query is constructed by combining the user's `persona` and `job_to_be_done` with additional keywords to enhance search relevance (e.g., focusing on "Roman ruins, medieval architecture" for a history enthusiast).
4.  **Embedding Generation**:
    *   The generated semantic query is converted into a numerical vector (embedding) using a pre-trained sentence transformer model.
    *   Similarly, each extracted PDF section's text (including its title for emphasis) is converted into an embedding.
5.  **Relevance Ranking (Sections)**: Cosine similarity is used to calculate the semantic similarity between the query embedding and each section embedding. Sections are then ranked based on these scores.
6.  **Sub-Section Analysis**: From the top-ranked sections, a further analysis is performed to identify the single most relevant sentence within each of these sections. This is done by embedding each sentence and comparing it against the original semantic query.
7.  **Output Generation**: The final output is a JSON file (`challenge1b_output.json`) containing metadata, a list of the top 5 most relevant sections (with their title, document, page, and importance rank), and a list of the top 5 most relevant sub-sections (the most relevant sentence from each of the top 5 sections).

## 2. Models and Libraries Used

*   **PyMuPDF (`fitz`)**: Used for robust PDF parsing, including outline extraction and text extraction from pages.
*   **`sentence-transformers`**: A Python framework for state-of-the-art sentence, text and image embeddings.
    *   **Model**: `all-MiniLM-L6-v2` is specifically used for generating embeddings. This model is chosen for its balance of performance and efficiency, making it suitable for CPU-based inference.
*   **`torch`**: The underlying deep learning library required by `sentence-transformers`. The solution is configured to use the CPU version of PyTorch for broader compatibility.
*   **Standard Python Libraries**: `os`, `json`, `time`, `pathlib`.

## 3. How to Build and Run Your Solution



### 1.  **Navigate to the project directory**:
    Open your terminal or command prompt and change your current directory to the `CHALLENGE_1B` folder, where the `Dockerfile` and `main.py` are located.

    ```
    cd C:\My projects\challenge_1b
    ```

### 2.  **Prepare Input Files**:
    *   Place your PDF documents in the `input/` directory within this `CHALLENGE_1B` folder.
    *   Ensure you have a `persona.json` file in the `input/` directory with the required `persona` and `job_to_be_done` fields.

### 3.  **Build the Docker Image**:
    This command builds the Docker image based on the `Dockerfile`. This might take some time on the first run as it downloads base images and installs dependencies.

    ```
    docker build -t pdf-semantic-search-1b .
    ```
    *   `-t CHALLENGE_1B`: Tags the image with a readable name. You can use any name you prefer.
    *   `.`: Specifies that the `Dockerfile` is in the current directory.

### 4.  **Run the Docker Container**:
    This command runs your application within a Docker container. The `-v` flags mount your local `input` and `output` directories into the container, allowing the container to access your PDFs and write the results back to your local machine.

    ```
    docker run -it --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" pdf-semantic-search-1b
    ```
    *   `docker run`: Command to run a container.
    *   `-it`: Runs in interactive mode, allowing you to see logs and output.
    *   `--rm`: Automatically removes the container after it exits (keeps your system clean).
    *   `-v "$(pwd)/input:/app/input"`: Mounts your local `input` folder to `/app/input` inside the container.
    *   `-v "$(pwd)/output:/app/output"`: Mounts your local `output` folder to `/app/output` inside the container.
    *   `CHALLENGE_1B`: The name of the Docker image you built.

    Upon successful execution, `challenge1b_output.json` will appear in your local `output/` directory.

### 5.  **Verify Output (Optional)**:
    After the container finishes, you can verify the generated output JSON using the `verify_results.py` script locally:
    ```
    python verify_results.py
    ```
    This script will check the structure and basic content of the `challenge1b_output.json` file.
