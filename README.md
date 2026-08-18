# Bridging the Data Gap in Incremental Object Counting via Text-Semantic Adaptation

This repository is for incremental object counting. The model learns different classes sequentially across incremental phases, and during testing, it simultaneously performs both class identification and count estimation for the target objects.


## The network
Overall architecture of our method. To learn incremental tasks, the network fine-tunes a frozen CLIP, mainly by updating two proposed components:
a structure-anchored modulation adapter and an incremental cross-semantic weaver.

## Installation
* Clone this repository
* Organize your datasets as required
* Install Python dependencies. We use python 3.8 and pytorch 1.8.0
```
pip install -r requirements.txt
```
## Organize the test dataset

### Dataset structures:

```text
<DATASET_ROOT>/
├── <stage_1_directory>/
│   └── test_data/
│       ├── images/
│       │   └── <stage_1_prefix>*.jpg
│       └── ground_truth/
│           └── <same_image_name>.npy
├── <stage_2_directory>/
│   └── test_data/
│       ├── images/
│       │   └── <stage_2_prefix>*.jpg
│       └── ground_truth/
│           └── <same_image_name>.npy
├── ...
└── <stage_n_directory>/
    └── test_data/
        ├── images/
        │   └── <stage_n_prefix>*.jpg
        └── ground_truth/
            └── <same_image_name>.npy
```

`DATASET_ROOT` is the root directory containing all incremental counting stages. Each image must have a same-named `.npy` annotation file in the corresponding `ground_truth` directory. 


## Dataset Configuration

Dataset-specific stage directories, class names, and filename prefixes are defined in `dataset_config.json`.

```json
{
  "background_class": "background",
  "prompt_file": "prompt/class_prompt.json",
  "stages": [
    {
      "directory": "category_1",
      "class_name": "class_1",
      "file_prefix": "class1_"
    },
    {
      "directory": "category_2",
      "class_name": "class_2",
      "file_prefix": "class2_"
    }
  ]
}
```

Configuration fields:

- `background_class`: name of the background class;
- `prompt_file`: path to the JSON file containing text prompts;
- `directory`: directory of the dataset introduced at this incremental stage;
- `class_name`: class identifier and the corresponding key in the prompt file;
- `file_prefix`: prefix used to identify foreground images during training. If omitted, `class_name` is used.

## Configuration and Testing

### Set the Dataset Path

Open `main.py` and locate the following line:

```python
dataset_dir = '/path/to/your/dataset'
dataset_config_path = 'dataset_config.json'
test_phase = None
```

Replace `dataset_dir` with the actual root directory of your dataset. `dataset_config_path` can point to a different configuration file when evaluating another dataset.

`test_phase = None` evaluates the final stage listed in `dataset_config.json`. To evaluate an earlier stage, set its zero-based index explicitly. For example:

```python
test_phase = 2
```

###  Run the Test

Because the code uses relative paths, run the test from the project root containing `main.py`, `prompt`, and `checkpoint`:

```bash
python main.py
```


