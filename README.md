![Label Meister Logo](./label_meister.png)

# Getting Started
## Requirements
Make sure you have Docker and Python3.12 installed on your system, and that it is functioning properly.
Before running the script, make sure you have the following dependencies installed (listed in requirements.txt). Use a virtual environment to manage these packages:
```bash
pip install -r requirements.txt
```

## Step 1: Start the CLIP Server in a Docker Container
#### In your terminal, run the following command to start the Docker container that hosts the CLIP server:
```bash
docker run -p 51009:51000 -v $HOME/.cache:/home/cas/.cache jinaai/clip-server
```
This command will pull and run the Jina AI CLIP server container, exposing it on port 51009 for local communication. 
* add "--gpus all" flag to the command if you have a GPU, and have docker configured for GPU acceleration with nvidia-container-toolkit. (This will result in your datset being labeled much more quickly)
* add "--stats" flag to get a stats.csv that includes insightful stats

## Step 2: Run the Dataset Labeling Script
#### In a second terminal window, ensure you are in the root directory of this cloned repository. With a virtual environment activated and all required packages from requirements.txt installed, run the following command to label a dataset:
```bash
python main.py --file test.csv --categories "cat,dog,bird,centipede,word,any words,any text string will work,this command will work,another example,you can add more"
```
* add "--batch_size" flag to adjust the number of images that are processed in each batch. Increase if you have a lot of RAM. Default is 100

### Input File Options:
* CSV File: Each element in the CSV should be on its own line.
* ZIP File of Images: A zip file containing images can be provided for labeling.
### Note:
* Images Work Best: The model performs better with images compared to text datasets.
* For text datasets, it is recommended to use more advanced options like the ChatGPT API for higher accuracy.




============================================================

## Licensing Information

This project uses the CLIP server provided by [Jina AI](https://github.com/jina-ai/clip-as-service) through a Docker container, which is licensed under the **Apache License 2.0**.

Portions of the CLIP server model, specifically `model.py` and `simple_tokenizer.py`, are licensed under the **MIT License** via [OpenCLIP](https://github.com/mlfoundations/open_clip).

For more details on the license terms, please refer to:
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT License](https://opensource.org/licenses/MIT)
