"""
Question answering is a common NLP task with several variants.
In some variants, the task is multiple-choice: A list of possible answers are supplied with each question,
and the model simply needs to return a probability distribution over the options.
A more challenging variant of question answering, which is more applicable to real-life tasks,
is when the options are not provided.
Instead, the model is given an input document -- called context -- and a question about the document,
and it must extract the span of text in the document that contains the answer.
In this case, the model is not computing a probability distribution over answers,
but two probability distributions over the tokens in the document text,
representing the start and end of the span containing the answer.
This variant is called "extractive question answering".
"""

from datasets import load_dataset

datasets = load_dataset("squad")
# uncomment to test dataset loading, prints single training exmp
# print(datasets["train"][0])

# preprocessing training data.

from transformers import AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# max return of a feature(question and context)

max_length = 384
doc_stride = (
    128
)


def prepare_train_features(examples):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    examples["context"] = [c.lstrip() for c in examples["context"]]
    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length = max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
    offset_mapping = tokenized_examples.pop('offset_mapping')

    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples['input_ids'][i]
        cls_index = input_ids.index(tokenizer.cls_toekn_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers['answer_start']) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples['end_positions'].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the
            # CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the
                # answer.
                # Note: we could go after the last offset if the answer is the last word (edge
                # case).
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples


tokenized_datasets = datasets.map(
    prepare_train_features,
    batched=True,
    remove_columns=datasets["train"].column_names,
    num_proc=3,
)

train_set = tokenized_datasets["train"].with_format("numpy")[
    :
]  # Load the whole dataset as a dict of numpy arrays
validation_set = tokenized_datasets["validation"].with_format("numpy")[:]

from transformers import TFAutoModelForQuestionAnswering

model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

import tensorflow as tf

from tensorflow import keras

optimizer = keras.optimizers.Adam(learning_rate=5e-5)

keras.mixed_precision.set_global_policy("mixed_float16")

model.compile(optimizer=optimizer)

context = """Keras is an API designed for human beings, not machines. Keras follows best
practices for reducing cognitive load: it offers consistent & simple APIs, it minimizes
the number of user actions required for common use cases, and it provides clear &
actionable error messages. It also has extensive documentation and developer guides. """
question = "What is Keras?"

inputs = tokenizer([context], [question], return_tensors="np")
outputs = model(inputs)
start_position = tf.argmax(outputs.start_logits, axis=1)
end_position = tf.argmax(outputs.end_logits, axis=1)
print(int(start_position), int(end_position[0]))

answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
print(tokenizer.decode(answer))