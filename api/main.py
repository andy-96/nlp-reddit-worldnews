import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import argparse

from api.model.comment_generator import CommentGenerator
from api.model.comment_generator_opennmt import CommentGenerator2

load_dotenv()

class Headline(BaseModel):
    headline: str

app = FastAPI()

@app.post("/generate-comment")
def generate_comment(input: Headline):
    print(f'Received new query: {input.headline}')
    comment = comment_generator.generate(input.headline)
    print('Finished!')
    return comment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start uvicorn server')
    parser.add_argument('--model', help="Choose a model")
    parser.add_argument('--preprocessed_path', help="Path of preprocessed data")
    args = parser.parse_args()

    print(f'{args.model} was chosen!')
    if args.model == 'opennmt':
        comment_generator = CommentGenerator2()
    else:
        comment_generator = CommentGenerator(args.model, args.preprocessed_path)
    uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT')), log_level='info')