# Spark Project
## Q1 Question and answering system

Design and build a generative question answering system. The training data set is from SQuAD v2 Dataset (also located in the server path /shareddata/data/project2 ).

(1) Write python code to process data. Use the context and question as input, and the answer as output. Use the official validation set as test set, and split the original training set into training set and validation set (5000 samples for valid set, the rest for train set). Prepare the data according to the requirements of model training (Can refer to the original T5 and Flan-T5 paper for data format). You can use either Pyspark or pure python code.

(2) Write the bash script and Ray-train python code to train the QA model by further funetuning the Flan-T5-small model. As no GPU is available in the server, you can use pytorch-cpu to debug your code, train the model for a few hours and save a checkpoint. Note that the validation set is used for designing the hyperparameters and selecting the model checkpoint. You can also refer to the training examples in the huggingface repo. You can also rent the GPU server in AutoDL.

(3) Deploy the finetuned QA model with Spark-NLP, and answer the questions of the test set in a streaming processing manner using Kafka. You can search the spark-nlp model here. Note that the deployed model is not the original Flan-T5- small model.
## Solution:
### (1) Data Preprocessing

1. Refer to the data section of the [T5](https://arxiv.org/pdf/1910.10683) paper for the SQuAD v2 Dataset format:

   **Processed input:** question: {***question***} context: {***context***}

   **Processed target:** *{**answers*** 'text' part}

2. Data Splitting:

   The original training set contains 5000 samples for the test set, while the rest are for the training set, using `train_test_split` for partitioning.

   The original validation set is used as the test set.

   Save the dataset in Parquet format.
   
### (2) Fine-tuning the Model

**Bash script** in `train_qa.sh` is responsible for running the Python file.

**Ray-train Python code** in `train_qa.py`.

Code Overview:

1. Training Function Definition: Define the `train_func` function, which encapsulates the following content:

   * Data Preprocessing: Load the training and validation datasets (`train_df` and `valid_df`) and convert them into Hugging Face's `Dataset` object using `Dataset.from_pandas`.

   * Tokenization: Load the pre-trained Flan-T5-small tokenizer and define the `tokenize_function` for encoding inputs and targets. Use the `Dataset.map` method to apply the `tokenize_function` to the training and validation datasets to generate tokenized datasets for model training.

   * Model Loading and Training Parameter Setting: Load the pre-trained Flan-T5-small model and use the `TrainingArguments` class to set training parameters, such as output directory, evaluation strategy, learning rate, batch size, number of training epochs, etc.

   * Define the `compute_metrics` function to calculate evaluation metrics. This function decodes the prediction results and computes evaluation results using the `squad_v2` metrics from the `evaluate` library.

   * Create a trainer object using Hugging Face's `Trainer` class, passing the model, training parameters, training and validation datasets, and the evaluation metrics computation function.

   * Use Ray Train's `RayTrainReportCallback` to report metrics to Ray Train.

   * Use the `ray.train.huggingface.transformers.prepare_trainer` function to prepare the Trainer object for distributed training.

2. Hyperparameter Search: Define the hyperparameter space `hyperparameter_space`, which includes different combinations of learning rates and batch sizes. Iterate through the hyperparameter space and perform the following steps for each set of hyperparameters:

   * Create a distributed trainer object using Ray Train's `TorchTrainer` class, passing the `train_func` function and the `ScalingConfig` settings (set the number of worker nodes to 1 and not using GPU).

   * Call the `TorchTrainer.fit` method to start distributed training and obtain training results.

   * Retrieve evaluation metrics (F1 score) on the validation set from the training results and compare them with the current best metrics. If the current metrics are better, update the best metrics, best hyperparameters, and best checkpoint path.

3. Best Model Saving and Loading: Use the `best_checkpoint_path.as_directory` method to save the best checkpoint to the local directory, and load the best model using `T5ForConditionalGeneration.from_pretrained` from the checkpoint directory. Finally, save the best model to the specified path.

4. Results Output: Output the best hyperparameters and best validation metrics.

### (3) Testing the Model

[`HuggingFace_ONNX_in_Spark_NLP_T5.ipynb`](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/onnx/HuggingFace_ONNX_in_Spark_NLP_T5.ipynb#scrollTo=AcVmXaYCWVb7):

Deploy the Hugging Face model (the best model fine-tuned in the second part) as a Spark NLP model.

`kafka.sh` file:

1. Start Zookeeper and Kafka services.
2. Create a Kafka topic named "qaaa".
3. Run the `kafka_producer.py` script to send the preprocessed data to Kafka.

`kafka_producer.py` file:

1. Read the preprocessed Parquet format data, which includes the "context" and "question" fields.
2. Create a Kafka producer that converts each row of the test set into a dictionary containing the "context" and "question" fields, and sends it to the "qaaa" topic.

`project2.ipynb` file:

1. Configure the SparkSession and Kafka-related parameters.
2. Read data from the "qaaa" topic in Kafka and parse the JSON-formatted messages.
3. Merge the "context" and "question" columns into a single text data column and write the preprocessed data to memory.
4. Define the document assembler and load the fine-tuned T5 model.
5. Define a Spark pipeline that includes the document assembler and T5 model.
6. Read data from memory, apply the pipeline for processing, and generate answers to the questions.


### 环境
1. Create a new virtual environment:
```bash
python3.10 -m venv myenv
source myenv/bin/activate
```

2. In the new virtual environment, install PyTorch:

```bash
pip install torch
```

3. Then, install the latest stable version of the transformers library:

```bash
pip install transformers
```

4. Next, install other necessary libraries, such as pandas, scikit-learn, etc.

```bash
pip install pandas scikit-learn datasets evaluate transformers ray
pip install -U "ray[data,train,tune,serve]"
pip install "ray[tune]"
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentencepiece
pip install accelerate -U
```
