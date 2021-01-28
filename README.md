![Cover](assets/cover.png)

*This repo was created as part of the Applied Deep Learning for NLP in held in WS 2020/2021 at the Technical University of Munich*

## Introduction

There are always situations where you don't know the perfect answer to a political topic and with this project, we have solved this problem!

>We've built an **Alexa skill** that comments on any given **news headline** based on information that was gathered from **Reddit**.

[see examples](#examples)

## Quickstart

### OpenNMT model

1. Download pretrained OpenNMT model [here](https://drive.google.com/drive/folders/17wA8XxT-7rQMqboWUBxBH94fLA5Oifhp?usp=sharing)
2. Serve the model using Tensorflow Serving (Inspired by [this example](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving/tensorflow_serving))
   ```bash
   docker run -p 9000:9000 -v $PWD:/models --name tensorflow_serving --entrypoint tensorflow_model_server tensorflow/serving --enable_batching=true --port=9000 --model_base_path=/models/hot-take-model --model_name=hot-take-model
   ```
3. Install pipenv: `pip install pipenv`
4. Setup virtual environment using pipenv: `pipenv --python 3.8`
5. Install all dependencies: `pipenv install`
6. Create environment file based on the `.env.example`
7. Start server: `pipenv run python3 api.main --model openmnt`
8. Test: `curl -X POST localhost:8000/generate-comment -d '{"headline": "This is amazing"}'`

### Self-trained Transformer model

1. Download the the [200k_comment_model](https://drive.google.com/drive/folders/1Q8X8osJwx7EklLvoXuSeP2dDA-Go8tL_?usp=sharing)
2. Move them into `api/model/pretrained`
3. Install pipenv: `pip install pipenv`
4. Setup virtual environment using pipenv: `pipenv --python 3.8`
5. Install all dependencies: `pipenv install`
6. Create environment file based on the `.env.example`
7. Start server: `pipenv run python3 api.main --model 200k_comment_model --preprocessed_data`
8. Test: `curl -X POST localhost:8000/generate-comment -d '{"headline": "This is amazing"}'`

## Documentation

### Structure of project

`/alexa-skill`: Code of the skill. Can be uploaded to the alexa developer console directly or on Amazon Lambda

`/api`: Main repo of the API and model

`/assets`: Images, PDFs, etc.

`/data`: Raw and processed data

`/data-acquisition`: Jupyter notebooks and Python code for acquiring the data

### Data Acquisition

We used [PRAW](https://praw.readthedocs.io/en/latest/index.html) and [pushshift.io](https://pushshift.io/api-parameters/) to crawl Reddit posts and top-level comments from the subreddits *worldnews, news, politics, uplifitingnews, truenews* from over the last 3 years.

### Data Preprocessing

We filtered the data using two approaches:

- Filtering the negative scores
- Filtering based on keywords (deleted, removed, tl;dr)
- Keeping only one sentence per comment to reduce complexity in training

### Model

As this task can be considered as a translation task, we used a Transformer-based architecture. In total, we trained three models from scratch:

1. 50k_comments_model (implementation based on the lectures code, not provided in this repo)
    - Did not properly learn grammatical structure
    - Could not understand the context of the headline

2. 200k_comments_model (implementation based on the lectures code)
    - One epoch took approx. 4 hours, thus not feasible to train with only access to Google Colab's GPUs

3. OpenNMT Transformer model
    - Learned grammatical structure
    - Could give coherent answers to new unseen headlines
    - Is quite opinionated
    - Learned correlation between topics were visible, e.g. connecting Republicans/Democrats with Socialist/Communists

### Deployment

For demonstration purposes, we deployed our project into a production environment using the OpenNMT model. On a AWS-server instance, we started two Docker containers serving the Tensorflow model using Tensorflow Serving and the Python code interacting between Alexa and the model and pre/postprocessing the data.

### Potential Extensions

Even though, the performance is considerably good, we propose following extensions:

- Improved preprocessing of Reddit comments, e.g. using spellchecker
- Use multi-sentence comments as target data
- Classify comments on sentiment and thus creating a dynamic comment generator
- Larger amount of data
- Extension to further domains than news/politics
- Deploying this service as a bot on Reddit

## Examples

## Learnings

- Playing around with and understand the data is crucial to anticipate patterns the model might learn early
- It makes sense to use a library like OpenNMT to have a good baseline and improve upon this baseline
- Even though, we "only" collected roughly over 250k comments, the model learned the grammatical structure of the English language and could give coherent answers

> If there are any questions or problems, feel free to create an issue!

*With lots of ❤️ by Maja & Andy*
