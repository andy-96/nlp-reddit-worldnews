# Alexa, Give me your Hot Take

![Cover](assets/cover.png)

*This repo was created as part of the Applied Deep Learning for NLP in held in WS 2020/2021 at the Technical University of Munich*

>We've built an **Alexa skill** that comments on any given **news headline** based on information that was gathered from **Reddit**.

## Motivation

There are always situations where you don't know the perfect answer to a political topic and with this project, we have solved this problem!

## Quickstart

### OpenNMT model

1. Download pretrained OpenNMT model [here](https://drive.google.com/drive/folders/17wA8XxT-7rQMqboWUBxBH94fLA5Oifhp?usp=sharing)
2. Serve the model using Tensorflow Serving (Inspired by [this example](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving/tensorflow_serving))

   ```docker run -p 9000:9000 -v $PWD:/models --name tensorflow_serving --entrypoint tensorflow_model_server tensorflow/serving \
        --enable_batching=true --port=9000 --model_base_path=/models/ende --model_name=ende```

3. Install pipenv: `pip install pipenv`
4. Setup virtual environment using pipenv: `pipenv --python 3.8`
5. Install all dependencies: `pipenv install`
6. Create environment file based on the `.env.example`
7. Start server: `pipenv run python3 api.main --model openmnt`
8. Test: `curl -X POST localhost:8000/generate-comment -d '{"headline": "This is amazing"}'`

### Self-trained Transformer model

1. Download the 50k_comment_model or the 200k_comment_model
2. Move them into `api/model/pretrained`
3. Install pipenv: `pip install pipenv`
4. Setup virtual environment using pipenv: `pipenv --python 3.8`
5. Install all dependencies: `pipenv install`
6. Create environment file based on the `.env.example`
7. Start server: `pipenv run python3 api.main --model 50k_comment_model --preprocessed_data_path=data`
8. Test: `curl -X POST localhost:8000/generate-comment -d '{"headline": "This is amazing"}'`

## Documentation

### Data Acquisition

We used [PRAW](https://praw.readthedocs.io/en/latest/index.html) and [pushshift.io](https://pushshift.io/api-parameters/) to crawl Reddit posts and top-level comments from the subreddits *worldnews, news, politics, uplifitingnews, truenews* from over the last 3 years.

### Data Preprocessing

We filtered the data using two approaches:

- Filtering the negative scores
- Filtering based on keywords (deleted, removed, tl;dr)
- Keeping only one sentence per comment to reduce complexity in training

### Model

As this task can be considered as a translation task, we used a Transformer-based architecture. In total, we trained three models:

1. 50k_comments_model

2. 200k_comments_model

3. OpenNMT Transformer model

### Deployment

For demonstration purposes, we deployed our project into a production environment using the OpenNMT model. On a AWS-server instance, we started two Docker containers serving the Tensorflow model using Tensorflow Serving and the Python code interacting between Alexa and the model and pre/postprocessing the data.

### Extension

Even though, the performance is considerably good, we propose following extensions:

- Improved preprocessing of Reddit comments, e.g. using spellchecker
- Use multi-sentence comments as target data
- Larger amount of data
- Extension to further domains than news/politics
- Deploying this service as a bot on Reddit

## Examples