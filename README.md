# cusutom yolov8

This is self-build up yolov8 for custmization.
Also this is packaged by poetry.

<img src="https://github.com/madara-tribe/custom-yolov8/assets/48679574/4acea454-cfc0-42ac-819e-51d495ec131b" width="850px" height="200px"/>

<img src="https://github.com/madara-tribe/custom-yolov8/assets/48679574/33da605b-f188-4dde-9e0f-e2fa4d1a43b9" width="850px" height="200px"/>


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
