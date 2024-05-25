# 启动Zookeeper
echo "Starting Zookeeper..."
workloc="/data/HW/HW3/kafka_2.13-3.7.0"
$workloc/bin/zookeeper-server-start.sh $workloc/config/zookeeper.properties

# 启动Kafka
echo "Starting Kafka..."
workloc="/data/HW/HW3/kafka_2.13-3.7.0"
$workloc/bin/kafka-server-start.sh $workloc/config/server.properties

# 创建Kafka主题
echo "Creating Kafka topic 'qaaa'..."
$workloc/bin/kafka-topics.sh --create --topic qaaa --bootstrap-server localhost:9092

echo "Running Kafka producer..."
python /data/HW/proj2/kafka_producer.py

echo "ok!"