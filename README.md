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
