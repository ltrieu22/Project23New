# Project23New
Project 23 new


## Instructions for running


1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd Project23New
   ```

Tested on Python 3.11

2. **Install Pip and Poetry/requirements.txt**

   ```bash
   # Install pip if not already installed
   python -m ensurepip --upgrade
   
   # Install Poetry
   pip install poetry
   ```
   Poetry instructions: https://python-poetry.org/docs/


3. **Install dependencies**

   ```bash
   poetry install --no-root
   ```

NOTE: poetry is set to use torch with basic CPU running. This however can take quite a lot of time when running the interference and finetuning. Depending on your GPU please use CUDA or ROCM
 https://pytorch.org/get-started/locally/


 4. **Install HUMMUS dataset**

 download: https://gitlab.com/felix134/connected-recipe-data-set/-/blob/master/data/hummus_data/preprocessed/pp_recipes.zip?ref_type=heads

 extract and place it under Project23New/pp_recipes.csv



main.ipynb contains tasks 1-4. 
fine_tuning.ipynb contains 5-10.


To test the whole project:

1. First you need to run the whole main.ipynb notebook in order (you can modify the number of examples used for data in the last cell by changing num_examples=1000 for both single and multi turn)

2. After the files are generated run the whole fine_tuning.ipynb notebook in order

To test the fine tuning part: You can use the git tracked provided jsonl files so main.ipynb does not have to be run again.
