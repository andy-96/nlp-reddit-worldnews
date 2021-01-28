from fastapi import FastAPI
from pydantic import BaseModel
from api.model.comment_generator import CommentGenerator
from api.model.comment_generator_opennmt import CommentGenerator2

class Headline(BaseModel):
    headline: str

app = FastAPI()

comment_generator = CommentGenerator()

@app.post("/generate-comment")
def generate_comment(input: Headline):
    print(f'Received new query: {input.headline}')
    comment = comment_generator.generate(input.headline)
    print('Finished!')
    return comment

comment_generator_alt = CommentGenerator2()

@app.post("/generate-comment-openmnt")
def generate_comment_alt(input: Headline):
    print(f'Received new query: {input.headline}')
    comment = comment_generator_alt.generate(input.headline)
    print('Finished!')
    return comment