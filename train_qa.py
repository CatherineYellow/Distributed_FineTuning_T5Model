import os
import numpy as np
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import ray.train.huggingface.transformers
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
import pandas as pd
from evaluate import load

def train_func(hyperparameters):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        flattened_predictions = np.argmax(predictions, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(flattened_predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        ids = [str(i) for i in range(len(decoded_preds))]
        predictions_dict = [{"id": id_, "prediction_text": pred, "no_answer_probability": 0.0} for id_, pred in zip(ids, decoded_preds)]
        references_dict = [{"id": id_, "answers": [{"text": label, "answer_start": 0}]} for id_, label in zip(ids, decoded_labels)]

        squad_v2_metric = load("/data/lab/proj2/squad_v2")
        results = squad_v2_metric.compute(predictions=predictions_dict, references=references_dict)
        return results

    train_df = pd.read_parquet("/data/lab/proj2/data/train_df.parquet")
    valid_df = pd.read_parquet("/data/lab/proj2/data/valid_df.parquet")

    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # 选择前 10 个样本
    # train_dataset = train_dataset.select(range(1))
    # valid_dataset = valid_dataset.select(range(10))

    model_path = "/data/lab/proj2/flan-t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    def tokenize_function(examples):
        inputs = examples["Processed input"]
        targets = examples["Processed target"]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=1024, truncation=True, padding='max_length').input_ids
        model_inputs["labels"] = labels
        return model_inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=valid_dataset.column_names)

    model = T5ForConditionalGeneration.from_pretrained(model_path)

    # training_args = TrainingArguments(
    #     output_dir=f"/data/lab/proj2/output_step_{hyperparameters['learning_rate']}_{hyperparameters['per_device_train_batch_size']}_{hyperparameters['num_train_epochs']}",
    #     evaluation_strategy="epoch",
    #     learning_rate=hyperparameters['learning_rate'],
    #     per_device_train_batch_size=hyperparameters['per_device_train_batch_size'],
    #     per_device_eval_batch_size=4,
    #     num_train_epochs=hyperparameters['num_train_epochs'],
    #     weight_decay=0.01,
    #     save_total_limit=1,
    #     report_to="none",
    #     save_strategy="epoch",
    # )
    training_args = TrainingArguments(
        output_dir=f"/data/lab/proj2/output_step_{hyperparameters['learning_rate']}_{hyperparameters['per_device_train_batch_size']}",
        evaluation_strategy="steps",
        eval_steps=5,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        max_steps=10,
        weight_decay=0.01,
        save_steps=5, 
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        compute_metrics=compute_metrics,
    )

    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)

    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    result = trainer.train()
    
    return result

hyperparameter_space = [
    {'learning_rate': 1e-5, 'per_device_train_batch_size': 4},
    {'learning_rate': 2e-5, 'per_device_train_batch_size': 8},
]

best_hyperparameters = None
best_metric = 0.0
best_checkpoint_path = None

for hyperparameters in hyperparameter_space:
    ray_trainer = TorchTrainer(
        lambda: train_func(hyperparameters),
        scaling_config=ScalingConfig(num_workers=1, use_gpu=False),
    )

    result: ray.train.Result = ray_trainer.fit()

    metrics = result.metrics
    validation_metric = metrics['eval_f1']

    if validation_metric > best_metric:
        best_metric = validation_metric
        best_hyperparameters = hyperparameters
        best_checkpoint_path = result.checkpoint

with best_checkpoint_path.as_directory() as checkpoint_dir:
    best_model_path = os.path.join(
        checkpoint_dir,
        ray.train.huggingface.transformers.RayTrainReportCallback.CHECKPOINT_NAME,
    )
    print(f"Loading the best model from: {best_model_path}")
    best_model = T5ForConditionalGeneration.from_pretrained(best_model_path)

best_model.save_pretrained("/data/lab/proj2/best_model")
print("Best model saved to /data/lab/proj2/best_model")
print(f"Best hyperparameters: {best_hyperparameters}")
print(f"Best validation metric: {best_metric}")