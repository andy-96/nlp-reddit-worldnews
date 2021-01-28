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
    parser.add_argument('--model')
    args = parser.parse_args()
    print(args)
    # comment_generator_alt = CommentGenerator2()
    # comment_generator = CommentGenerator()
    # uvicorn.run(app, host='0.0.0.0', port=int(os.getenv('PORT')), log_level='info')