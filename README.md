### Setup

Install [PyTorch](https://pytorch.org/get-started/locally/), NumPy, and
[other common packages](https://github.com/ohjay/sdae/blob/master/requirements.txt) if you don't have them already.
```
pip install -r requirements.txt
```
might work.

There are two datasets you'll need to download manually (see below).
I suggest you create a `data` folder and unpack the relevant files into it.
Later, you will be able to specify the dataset paths as command line arguments.

| | | |
|-|-|-|
| Olshausen    | [Link](http://www.rctn.org/bruno/sparsenet)                       |
| CUB-200-2011 | [Link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) |

### Experiments

You can run one of the provided experiment scripts using the command
`./experiments/xx.sh`, where `xx` is the experiment number (e.g. `01`).
I have included a brief experiment description at the top of each script.
You will need to change arguments such as `olshausen_path` and `cub_folder` which refer to dataset paths.

### Sample Results

coming soon!
