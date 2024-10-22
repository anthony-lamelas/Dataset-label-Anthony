## Step 1. In a teminal window, run this command to start the docker container that will run the dataset labler model:
```bash
docker run -p 51009:51000 -v $HOME/.cache:/home/cas/.cache jinaai/clip-server
```

## Step 2. In a 2nd terminal window 
### while in the root directory of this cloned repo, with a virtual environment activated and all the packages in requirements.txt installed, run a command like this sample one

It can be a zip file of images, or a csv file of text with each element in the CSV on its own line 

```bash
python main.py --file test.csv --categories "cat,dog,bird"
```
#### Note: Images work much better, there are better options such as CHatGPT API if you want to label a text dataset



============================================================

## Licensing Information

This project uses the CLIP server provided by [Jina AI](https://github.com/jina-ai/clip-as-service) through a Docker container, which is licensed under the **Apache License 2.0**.

Portions of the CLIP server model, specifically `model.py` and `simple_tokenizer.py`, are licensed under the **MIT License** via [OpenCLIP](https://github.com/mlfoundations/open_clip).

For more details on the license terms, please refer to:
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT License](https://opensource.org/licenses/MIT)
