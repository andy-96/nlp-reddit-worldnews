import pickle

if __name__ == "__main__":
    with open(f'./data/2021-01-18_day_upliftingnews_comments.pkl', 'rb') as f:
        output = pickle.load(f)

    print(output)