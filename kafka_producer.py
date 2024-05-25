from kafka import KafkaProducer
from pyspark.sql import SparkSession
import json

# 创建SparkSession
spark = SparkSession.builder.getOrCreate()

# 读取预处理后的数据
data_df = spark.read.parquet("/data/HW/proj2/data/test_df.parquet")

# 创建Kafka生产者
producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# 将数据发送到Kafka主题
for row in data_df.toLocalIterator():
    message = {
        "context": row['context'],
        "question": row['question']
    }
    producer.send('qaaa', message)

# 关闭生产者
producer.close()
