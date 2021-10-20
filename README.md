
# Streamlit Monitoring Tool

The aim of this project is to monitor machine learning models on new input data via a broad range of metrics.

## Setup and Run

```
pip3 install -r requirements.txt
cd src/
streamlit run main_front.py
```

## Configuration File

You can modify the configuration file found in */params/conf/conf.json*. 

- Add a *dict_info_files* information containing your data and an equivalent mapping to the one shown below.

```
{
    "target": "VARIABLE TO PREDICT",
    "numerical_features": [NAME OF NUMERICAL COLUMNS],
    "categorical_features": [NAME OF CATEGORICAL COLUMNS]
}
```

- Choose the correct dict_info_file in the *selected_dataset* parameter of the configuration file.

## Video Demo of the App

https://user-images.githubusercontent.com/41548545/138106754-7504ee48-26a5-4a27-84e6-3ed1952a960a.mp4
