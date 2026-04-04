# Running the Models

## Running the Text-Based Emotion Classification Model
``` bash
python src/text_emotion.py
```

## Running the Facial Emotion Model
The data must be downloaded from the Kaggle link in data.txt and the "dataset" folder from the zip must be placed in the data folder. 

### Expected Structure:
project-ei-ai/   
├── data/   
│   ├── data.csv   
│   ├── data.txt   
│   ├── dataset    
│   └── emotions.csv   
└── src/   
    ├── DL_text_classification.ipynb   
    ├── facial_recognition.py   
    └── text_emotion.py   

If you would like to run a different folder structure, it must be updated in facial_recognition.py to reflect the changes. 

``` bash
python src/facial_recognition.py
```
