from api.model.comment_generator_alt import CommentGenerator2
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError, validator
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from api.model.comment_generator import CommentGenerator
from api.model.comment_generator_alt import CommentGenerator2

class Headline(BaseModel):
    headline: str

app = FastAPI()
comment_generator = CommentGenerator()
comment_generator_alt = CommentGenerator2()

@app.post("/generate-comment")
def generate_comment(input: Headline):
    print(f'Received new query: {input.headline}')
    comment = comment_generator.generate(input.headline)
    print('Finished!')
    return comment

def generate_comment_alt(input: Headline):
    print(f'Received new query: {input.headline}')
    comment = comment_generator_alt.generate(input.headline)
    print('Finished!')
    return comment