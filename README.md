# Zindi_Workshop_26_04_2025



Welcome to the Zindi Workshop: Building Intelligent Chatbots with Retrieval-Augmented Generation (RAG): From Theory to Practice




## Installation and Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.10 or Anaconda




### Steps to execute the code
1. Clone the repository:
      ```bash
      git clone https://github.com/nguefackuriel/Zindi_Workshop_26_04_2025.git
      cd Zindi_Workshop_26_04_2025
      ```

  2. If you work with anaconda, follow these steps, otherwise move to the step 3:
     - Create a conda environment using:
         ```bash
         conda create -n zindi_workshop python=3.10  
         ```
     - Activate the environment using:
       ```bash
       conda activate zindi_workshop
       ```
3. Install the requirements:
     ```bash
      pip install -r requirements.txt
     ```
4. To run the code, you need an LLM model, for this workshop we use, Llama3.2 pulled from Ollama. Visit this page to know how to use [Ollama](https://ollama.com/download). Depending on the system you have just follow the instructions. For Linux users, open a terminal and type:

    ```bash
      curl -fsSL https://ollama.com/install.sh | sh
    ```
    It is going to download Ollama, then you can pull whatever model you want. For Windows, just downlad the .exe file and execute it.

   To pull llama3.2, and mxbai-embed-large used for the embedding, use the following commands in the terminal:
   
    ```bash
      ollama pull llama3.2
    ```

    ```bash
      ollama pull mxbai-embed-large
    ```

5. Run the streamlit code:
   ```bash
      streamlit run my_code.py
   ```
A web page will open, now you can start using your Chatbot app and ask any questions related to your document!!!

Example of test.

![alt text](https://github.com/nguefackuriel/Zindi_Workshop_26_04_2025/blob/main/result_test.png?raw=true)

### Evaluate Your RAG system
1. Open the my_code.ipynb jupyter notebook
2. To use deepeval, you will need an OPEN_API_KEY as shown in the notebook, make sure you have one, otherwise it won't work. You can define it directly in the notebook or you can store it in a .env file and load it using load_env as I did in the notebook.
3. Follow the instructions in the notebook

You did it!!! 


