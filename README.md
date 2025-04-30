# Face-Recognition

We intend to perform face recognition. Face recognition means that for a given image you can tell the subject id. Our database of subject is very simple. It has 40 subjects. Below we will show the needed steps to achieve the goal of the assignment.

## How To Run

1- Virtaul Environment Setup

```
python3 -m venv face_rec
source face_rec/bin/activate
pip install -r requirements.txt
```

2- Running Tests

```
python main.py --reduction pca --cluster gmm
python main.py --reduction autoencoder --cluster kmeans
```
