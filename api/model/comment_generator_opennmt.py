import tensorflow as tf
import pyonmttok
import grpc
from dotenv import load_dotenv
import os

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from api.utils import preprocess_sentence
from api.config import MODEL_NAME, SENTENCEPIECE_MODEL, TIMEOUT, MAX_LENGTH

load_dotenv()
class CommentGenerator2():
    def __init__(self):
        channel = grpc.insecure_channel("%s:%d" % (os.getenv("TF_HOST"), int(os.getenv("TF_PORT"))))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.tokenizer = pyonmttok.Tokenizer("none", sp_model_path=SENTENCEPIECE_MODEL)

    def pad_batch(self, batch_tokens):
        """Pads a batch of tokens."""
        lengths = [len(tokens) for tokens in batch_tokens]
        for tokens, length in zip(batch_tokens, lengths):
            if MAX_LENGTH > length:
                tokens += [""] * (MAX_LENGTH - length)
        return batch_tokens, lengths, MAX_LENGTH


    def extract_prediction(self, result):
        """Parses a translation result.
        Args:
        result: A `PredictResponse` proto.
        Returns:
        A generator over the hypotheses.
        """
        batch_lengths = tf.make_ndarray(result.outputs["length"])
        batch_predictions = tf.make_ndarray(result.outputs["tokens"])
        for hypotheses, lengths in zip(batch_predictions, batch_lengths):
            # Only consider the first hypothesis (the best one).
            best_hypothesis = hypotheses[0].tolist()
            best_length = lengths[0]
            if best_hypothesis[best_length - 1] == b"</s>":
                best_length -= 1
            yield best_hypothesis[:best_length]


    def send_request(self, batch_tokens):
        """Sends a translation request.
        Args:
        tokens: A list of tokens.
        Returns:
        A future.
        """
        batch_tokens, lengths, max_length = self.pad_batch(batch_tokens)
        batch_size = len(lengths)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = MODEL_NAME
        request.inputs["tokens"].CopyFrom(
            tf.make_tensor_proto(
                batch_tokens, dtype=tf.string, shape=(batch_size, max_length)
            )
        )
        request.inputs["length"].CopyFrom(
            tf.make_tensor_proto(lengths, dtype=tf.int32, shape=(batch_size,))
        )
        return self.stub.Predict.future(request, TIMEOUT)


    def generate(self, batch_text):
        """Translates a batch of sentences.
        Args:
        batch_text: A list of sentences.
        Returns:
        A generator over the detokenized predictions.
        """
        preprocessed = preprocess_sentence(batch_text)
        batch_input = [self.tokenizer.tokenize(text)[0] for text in [preprocessed]]
        future = self.send_request(batch_input)
        result = future.result()
        output = [pred for pred in self.extract_prediction(result)][0]
        output = [word.decode("utf-8") for word in output]
        batch_output = output[0]
        for word in output[1:]:
            if word == 's' or word == 'm' or word == 'd':
                batch_output = "'".join([batch_output, word])
            elif word == '.' or word == '?' or word == '!':
                batch_output = "".join([batch_output, word])
            else:
                batch_output = " ".join([batch_output, word])
        return batch_output

if __name__ == "__main__":
    commentGenerator2 = CommentGenerator2()
    print(commentGenerator2.generate("I love you"))