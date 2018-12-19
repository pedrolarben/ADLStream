import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.core.Utils;
import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.serialization.LongDeserializer;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.io.*;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CountDownLatch;

public class MOAConsumerForNABDataset {

    private final static String BOOTSTRAP_SERVERS = "localhost:9092";

    private static Consumer<Long, String> createConsumer(String topic, boolean fromBeginning) {

        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
        props.put(ConsumerConfig.CLIENT_ID_CONFIG, "MOAConsumer");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "MOAConsumerGroup_" + UUID.randomUUID().toString());
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, LongDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        if (fromBeginning) props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        final Consumer<Long, String> consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Collections.singletonList(topic));
        return consumer;
    }


    /**
     * @param topicName Topic name. If it does not exists, the program will create it on Kafka.
     * @param classifierName Classifier name. Any class name that implements the moa.classifiers.Classifier interface.
     *                       For example: HoeffdingTree
     *                       See https://www.cs.waikato.ac.nz/~abifet/MOA/API/interfacemoa_1_1classifiers_1_1_classifier.html
     *
     */
    private static void runConsumer(String topicName, String classifierName, String datasetName,
                                    boolean fromBeginning, String resultsPath) throws FileNotFoundException {
        final Consumer<Long, String> consumer = createConsumer(topicName, fromBeginning);
        final int giveUp = 100;
        int noRecordsCount = 0;
        Map<String, Classifier> csvClassifiersMap = new HashMap<>();

        final DateTimeFormatter dateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        StringBuilder sbHeader = new StringBuilder();
        sbHeader.append("timestamp");
        sbHeader.append(",");
        sbHeader.append("readable_timestamp");
        sbHeader.append(",");
        sbHeader.append("value");
        sbHeader.append(",");
        sbHeader.append("anomaly_score");
        sbHeader.append(",");
        sbHeader.append("label");
        sbHeader.append(",");
        sbHeader.append("producer_timestamp");
        sbHeader.append(",");
        sbHeader.append("producer_readable_timestamp");
        sbHeader.append(",");
        sbHeader.append("consumer_timestamp");
        sbHeader.append(",");
        sbHeader.append("consumer_readable_timestamp");
        sbHeader.append(",");
        sbHeader.append("output_timestamp");
        sbHeader.append(",");
        sbHeader.append("output_readable_timestamp");
        sbHeader.append(",");
        sbHeader.append("producer_delay");
        sbHeader.append("\n");

        boolean isFirstLine = true;
        PrintWriter pw = null;
        String lastCsvFileName = null;

        while(true){
            final ConsumerRecords<Long, String> consumerRecords = consumer.poll(100);

            if(consumerRecords.count()==0){
                noRecordsCount++;
                if(noRecordsCount > giveUp) break;
                else continue;
            }

            String dirName = resultsPath + File.separator + "MOA_" +
                    classifierName + File.separator + datasetName;
            File directory = new File(dirName);
            if (!directory.exists()) {
                directory.mkdirs();
            }

            for (ConsumerRecord<Long, String> record : consumerRecords) {
                long consumerTimestamp = System.currentTimeMillis();
                String consumerReadableTimestamp = Instant.ofEpochMilli(consumerTimestamp)
                        .atZone(ZoneId.systemDefault())
                        .format(dateTimeFormatter);
                String value = record.value();
                Instance instance = null;
                NABJsonToMoaInstanceConverter converter = new NABJsonToMoaInstanceConverter();
                try {
                    instance = converter.moaInstance(value);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(converter.getDatasetName().equalsIgnoreCase("NAB_"+datasetName)){
                    String csvFileName = converter.getCsvFileName();
                    double producerDelay = converter.getProducerDelay();
                    long producerTimestamp = converter.getProducerTimestamp();
                    String producerReadableTimestamp = converter.getProducerReadableTimestamp();

                    if (lastCsvFileName == null || !lastCsvFileName.equalsIgnoreCase(csvFileName)){
                        if (pw != null) pw.close();
                        lastCsvFileName = csvFileName;
                        String fileName = dirName + File.separator + "MOA_" + classifierName + "_" + csvFileName;
                        pw = new PrintWriter(new FileOutputStream(new File(fileName), true));

                        if(isFirstLine){
                            pw.write(sbHeader.toString());
                            isFirstLine = false;
                        }
                    }

                    Classifier classifier;
                    if (csvClassifiersMap.containsKey(csvFileName)){
                        classifier = csvClassifiersMap.get(csvFileName);
                    }else{
                        classifier = MOAClassifierFactory.getClassifier(classifierName);
                        classifier.prepareForUse();
                        csvClassifiersMap.put(csvFileName, classifier);
                        isFirstLine = true;
                    }

                    double anomalyScore = Utils.maxIndex(classifier.getVotesForInstance(instance));
                    classifier.trainOnInstance(instance);

                    long outputTimestamp = System.currentTimeMillis();
                    String outputReadableTimestamp = Instant.ofEpochMilli(outputTimestamp)
                            .atZone(ZoneId.systemDefault())
                            .format(dateTimeFormatter);

                    StringBuilder sbLine = new StringBuilder();
                    sbLine.append(converter.getTimestamp());
                    sbLine.append(",");
                    sbLine.append(converter.getReadableTimestamp());
                    sbLine.append(",");
                    sbLine.append(converter.getValue());
                    sbLine.append(",");
                    sbLine.append(anomalyScore);
                    sbLine.append(",");
                    sbLine.append(instance.classValue());
                    sbLine.append(",");
                    sbLine.append(producerTimestamp);
                    sbLine.append(",");
                    sbLine.append(producerReadableTimestamp);
                    sbLine.append(",");
                    sbLine.append(consumerTimestamp);
                    sbLine.append(",");
                    sbLine.append(consumerReadableTimestamp);
                    sbLine.append(",");
                    sbLine.append(outputTimestamp);
                    sbLine.append(",");
                    sbLine.append(outputReadableTimestamp);
                    sbLine.append(",");
                    sbLine.append(producerDelay);
                    sbLine.append("\n");

                    pw.write(sbLine.toString());

                }
            }
            consumer.commitAsync();

        }
        consumer.close();
        if (pw != null) pw.close();
        System.out.println("DONE");
        System.exit(0);
    }

    public static void main(String[] args) throws Exception{

        String topicName = "";
        boolean fromBeginning = false;
        String datasetName = "";
        String resultsPath = "";
        String classifierName = "";

        if (args.length == 5){
            topicName = args[0];
            System.out.println("Topic name: " + topicName);
            classifierName = args[1];
            System.out.println("Classifier name: " + classifierName);
            datasetName = args[2];
            System.out.println("Dataset name: " + datasetName);
            fromBeginning = Boolean.parseBoolean(args[3]);
            System.out.println("From beginning: " + fromBeginning);
            resultsPath = args[4];
            System.out.println("Output results path:" + resultsPath);
        }else{
            System.out.println("Program arguments: [topicName] [classifierName] [datasetName] [fromBeginning] [outputResultsPath]");
            System.out.println("-classifierName: Any class name that implements the moa.classifiers.Classifier interface. \n" +
                    "For example: HoeffdingTree\n" +
                    "See https://www.cs.waikato.ac.nz/~abifet/MOA/API/interfacemoa_1_1classifiers_1_1_classifier.html");
            System.out.println("-outputResultsPath: Path where results will be saved.");
            System.exit(1);
        }


        final CountDownLatch latch = new CountDownLatch(1);
        // attach shutdown handler to catch control-c
        Runtime.getRuntime().addShutdownHook(new Thread("streams-shutdown-hook") {
            @Override
            public void run() {
                latch.countDown();
            }
        });

        try {
            if(!KafkaUtils.existsTopic(topicName)){
                KafkaUtils.createTopic(topicName, 1,1);
            }
            System.out.println("Classifying NAB data with " + classifierName);
            try{
                runConsumer(topicName, classifierName, datasetName, fromBeginning, resultsPath);
            }catch (Exception e){
                e.printStackTrace();
                System.out.println("Processing error with classifier: " + classifierName);
            }

            latch.await();
        } catch (Throwable e) {
            e.printStackTrace();
            System.exit(1);
        }
        System.exit(0);
    }

}
