# Example Command (Assuming 4 GPUs with indices 0-3):
```bash
CUDA_VISIBLE_DEVICES=RR0:4
cat optimized_main.yml | docker run -i -p 51009:51000 -v $HOME/.cache:/home/cas/.cache --gpus all jinaai/clip-server -i
```
# Then run optimized_main.py in a separate terminal window
```bash
python -m venv venv
sleep(2)
source venv/bin/activate
pip -r install requirements.txt
sleep(3)
python optimized_main.py --file /path/to/file --stats --batch_size 10000 --prefetch 1 --categories "cat,dog,bear,donkey,hippo"
``` 

### GPU Assignment with 8 Replicas:
|Replica ID | Assigned GPU|
|-----------|-------------|
|0	          |GPU 0|
|1	          |GPU 1|
|2	          |GPU 2|
|3	          |GPU 3| 
|4	          |GPU 0| 
|5	          |GPU 1|
|6	          |GPU 2|
|7	          |GPU 3|  

## Adjust According to Your Hardware:
* If you have more GPUs, adjust RR0:N where N is the number of GPUs. Increase replicas in the YAML to create more replicas per GPU.
* Larger batch_size: Allows GPUs to process more data per forward pass.
* Higher prefetch: Maintains a steady stream of data to the GPUs.
* Ensure that GPUs are close to 100% utilization. Monitor memory usage to avoid exceeding VRAM capacity.

## If GPUs Are Underutilized:
* Increase replicas: Add more replicas to increase GPU workload.
* Increase batch_size: Up to the point where it fits in GPU memory.
* Adjust prefetch: Ensure enough batches are in-flight.

## If CPUs Are a Bottleneck:
* Ensure that the client machine has sufficient CPU resources.
* Reduce any unnecessary CPU-bound processing in your script.

## Finetune:
There's a trade-off between the number of replicas and batch size per replica.
Experiment to find the optimal balance for your hardware.

High CPU load can bottleneck data feeding to GPUs.
Ensure that disk I/O is not a bottleneck when reading data.

Start with smaller datasets to fine-tune the settings.
Gradually scale up to your full dataset.
