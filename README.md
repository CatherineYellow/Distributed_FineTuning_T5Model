# Spark Project
## Q1 Question and answering system
Design and build a generative question answering system. The training data set is from SQuAD v2 Dataset (also located in the server path /shareddata/data/project2 ).
(1) Write python code to process data. Use the context and question as input, and the answer as output. Use the official validation set as test set, and split the original training set into training set and validation set (5000 samples for valid set, the rest for train set). Prepare the data according to the requirements of model training (Can refer to the original T5 and Flan-T5 paper for data format). You can use either Pyspark or pure python code.
(2) Write the bash script and Ray-train python code to train the QA model by further funetuning the Flan-T5-small model. As no GPU is available in the server, you can use pytorch-cpu to debug your code, train the model for a few hours and save a checkpoint. Note that the validation set is used for designing the hyperparameters and selecting the model checkpoint. You can also refer to the training examples in the huggingface repo. You can also rent the GPU server in AutoDL.
(3) Deploy the finetuned QA model with Spark-NLP, and answer the questions of the test set in a streaming processing manner using Kafka. You can search the spark-nlp model here. Note that the deployed model is not the original Flan-T5- small model.
## Solution:
### (1) 数据预处理

1. 参考 [T5](https://arxiv.org/pdf/1910.10683) 文章 SQuAD v2 Dataset 的数据部分，格式为：

   **Processed input:** question: {***question***} context: {***context***}

   **Processed target:** *{**answers*** 的 'text' 部分}

2. 数据划分：

   原始training set中5000个样本为测试集，其余为训练集，使用`train_test_split`划分

   原始validation set作为测试集

   把数据集存成parquet文件
   
### (2) 微调模型

**bash script**在`train_qa.sh`里，负责运行python文件

**Ray-train python code**在`train_qa.py`里

代码思路:

1. 训练函数定义:定义 `train_func` 函数,该函数封装了以下内容:

   * 数据预处理:加载训练和验证数据集(train_df和valid_df),并使用 `Dataset.from_pandas` 将其转换为 Hugging Face 的 `Dataset` 对象。

   * Tokenization:加载预训练的 Flan-T5-small tokenizer,并定义 `tokenize_function` 对输入和目标进行编码。使用 `Dataset.map` 方法将 `tokenize_function` 应用于训练和验证数据集,生成用于模型训练的 tokenized 数据集。

   * 模型加载和训练参数设置:加载预训练的 Flan-T5-small 模型,并使用 `TrainingArguments` 类设置训练参数,如输出目录、评估策略、学习率、批量大小、训练轮数等。

   * 定义 `compute_metrics` 函数用于计算评估指标。该函数对预测结果进行解码,并使用 `evaluate` 库中的 `squad_v2` 指标计算评估结果。

   * 使用 Hugging Face 的 `Trainer` 类创建训练器对象,传入模型、训练参数、训练和验证数据集以及评估指标计算函数。

   * 使用 Ray Train 的 `RayTrainReportCallback` 回调函数将指标报告给 Ray Train。 

   * 使用 `ray.train.huggingface.transformers.prepare_trainer` 函数准备 Trainer 对象以适应分布式训练。
2. 超参数搜索:定义超参数空间 `hyperparameter_space`,包含不同的学习率和批量大小组合。遍历超参数空间,对每组超参数执行以下步骤:

   * 使用 Ray Train 的 `TorchTrainer` 类创建分布式训练器对象,传入 `train_func` 函数和 `ScalingConfig` 配置(设置工作节点数为1和不使用 GPU)。

   * 调用 `TorchTrainer.fit` 方法启动分布式训练,并获取训练结果。

   * 从训练结果中获取验证集上的评估指标(F1 score),并与当前最佳指标进行比较。如果当前指标更好,则更新最佳指标、最佳超参数和最佳检查点路径。
3. 最佳模型保存和加载:使用 `best_checkpoint_path.as_directory` 方法将最佳检查点保存到本地目录,并使用 `T5ForConditionalGeneration.from_pretrained` 方法从检查点目录加载最佳模型。最后,将最佳模型保存到指定路径。
4. 结果输出:输出最佳超参数和最佳验证指标。
### (3) 测试模型

[`HuggingFace_ONNX_in_Spark_NLP_T5.ipynb`](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/onnx/HuggingFace_ONNX_in_Spark_NLP_T5.ipynb#scrollTo=AcVmXaYCWVb7) ：

将huggingface的model （采用第二问微调后最佳的model）部署为sparknlp的model

`kafka.sh`文件：

1. 启动Zookeeper和Kafka服务
2. 创建名为"qaaa"的Kafka主题
3. 运行`kafka_producer.py`脚本，将预处理后的数据发送到Kafka

`kafka_producer.py`文件：

1. 读取预处理后的Parquet格式数据，包含"context"和"question"字段
2. 创建Kafka生产者，将测试集的每行数据转换为包含"context"和"question"字段的字典，并发送到"qaaa"主题

`project2.ipynb`文件：

1. 配置SparkSession和Kafka相关参数
2. 从Kafka的"qaaa"主题中读取数据，并解析JSON格式的消息
3. 将"context"和"question"列合并为一列文本数据，并将预处理后的数据写入内存
4. 定义文档组装器和加载微调后的T5模型
5. 定义Spark管道，包括文档组装器和T5模型
6. 从内存中读取数据，应用管道进行处理，生成问题的答案
   
