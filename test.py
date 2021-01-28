import pickle

if __name__ == "__main__":
    with open(f'./data/1609801200_to_1611097199_worldnews_posts.pkl', 'rb') as f:
        output = pickle.load(f)

    print(output)