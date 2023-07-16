# cusutom yolov8

This is self-build up yolov8 for custmization.
Also this is packaged by poetry.

<img src="https://github.com/madara-tribe/custom-yolov8/assets/48679574/fcc0ce4f-71f5-4311-8a6d-ab8232525d4d" width="600px" height="400px"/>


<img src="https://github.com/madara-tribe/custom-yolov8/assets/48679574/ab990417-7131-4944-9306-e54b1cef7b08" width="600px" height="400px"/>


# how to setup
```sh
$ ./install.sh
```

# change model size
change file <code>custom/nn/tasks.py</code>
```python
def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    type = 'Yolov8n' or 'Yolov8s' or 'Yolov8m'
    model_type=type
```
# train/valid/predict/onnx_export
```sh
# train
python3 main.py --mode train

# valid
python3 main.py --mode valid -w <weight_path>

# predict
python3 main.py --mode predict -w <weight_path>

# onnxe export
python3 main.py --mode onnx -w <weight_path>
```


# References
- [poetry](https://qiita.com/ksato9700/items/b893cf1db83605898d8a)
