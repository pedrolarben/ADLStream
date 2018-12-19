import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.core.Utils;
import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.serialization.LongDeserializer;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.io.*;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

public class DSClassificationMOAConsumer {

    public final static String DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092";
//    private final static String BOOTSTRAP_SERVERS = "PLAINTEXT://hal9k.lsi.us.es:9092";

    private static Consumer<Long, String> createConsumer(String topic, String bootstrapServer, boolean fromBeginning) {
        if(bootstrapServer == "")
            bootstrapServer = DEFAULT_BOOTSTRAP_SERVERS;
        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServer);
        props.put(ConsumerConfig.CLIENT_ID_CONFIG, "MOAConsumer_" + UUID.randomUUID().toString());
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
    public static void runConsumer(String topicName, String classifierName, String bootstapServer, boolean fromBeginning, String resultsPath) {
        System.out.println("Topic " + topicName + " with classifier " + classifierName);
        final Consumer<Long, String> consumer = createConsumer(topicName, bootstapServer, fromBeginning);
        final int giveUp = 100;
        int noRecordsCount = 0;

        final Classifier classifier = MOAClassifierFactory.getClassifier(classifierName);
        classifier.prepareForUse();
        AtomicInteger numSamples = new AtomicInteger();
        AtomicInteger numSamplesCorrect = new AtomicInteger();

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        String datasetName = topicName.replace("streams_","");
        String dirName = resultsPath + File.separator + datasetName +
                File.separator + "MOA_" + classifierName;
        File directory = new File(dirName);
        if (!directory.exists()) {
            directory.mkdirs();
        }

        String dataCSV = dirName + File.separator + "data.csv";
        String metricsCSV = dirName + File.separator + "metrics.csv";

        PrintWriter pwData = null;
        PrintWriter pwMetrics = null;
        try {
            pwData = new PrintWriter(new FileOutputStream(new File(dataCSV)));
            pwMetrics = new PrintWriter(new FileOutputStream(new File(metricsCSV)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        StringBuilder sbHeader = new StringBuilder();
        sbHeader.append("total");
        sbHeader.append(",");
        sbHeader.append("train_time");
        sbHeader.append(",");
        sbHeader.append("test_time");
        sbHeader.append(",");
        sbHeader.append("accuracy");
        sbHeader.append("\n");
        pwMetrics.write(sbHeader.toString());

        boolean dataHeaderWritten = false;

        while(true){
            final ConsumerRecords<Long, String> consumerRecords = consumer.poll(1000);

            if(consumerRecords.count()==0){
                noRecordsCount++;
                if(noRecordsCount > giveUp) break;
                else continue;
            }

            for (ConsumerRecord<Long, String> record : consumerRecords) {
                String value = record.value();
                DSJsonToMoaInstanceConverter converter = new DSJsonToMoaInstanceConverter(topicName);
                Instance instance = null;
                try {
                    instance = converter.moaInstance(value);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if (!dataHeaderWritten) {
                    StringBuilder sbh = new StringBuilder();
                    for (String s : converter.getKeysList()) {
                        sbh.append(s);
                        sbh.append(",");
                    }
                    sbh.append("prediction");
                    sbh.append("\n");
                    pwData.print(sbh.toString());
                    dataHeaderWritten = true;
                }

                if (classifier.correctlyClassifies(instance)) {
                    numSamplesCorrect.incrementAndGet();
                }

                long testStartTime = System.nanoTime();
                double prediction = Utils.maxIndex(classifier.getVotesForInstance(instance));
                long testEndTime = System.nanoTime();
                long testTime = (testEndTime - testStartTime);
                double testTimeSeconds = (double)testTime / 1_000_000_000.0;

                long trainStartTime = System.nanoTime();
                classifier.trainOnInstance(instance);
                long trainEndTime = System.nanoTime();
                long trainTime = (trainEndTime - trainStartTime);
                double trainTimeSeconds = (double)trainTime / 1_000_000_000.0;


                numSamples.getAndIncrement();
                double accuracy = 100.0 * (double) numSamplesCorrect.get() / numSamples.get();

                StringBuilder sb = new StringBuilder();
                for (Double v : converter.getValuesList()) {
                    sb.append(v);
                    sb.append(",");
                }
                sb.append(prediction);
                sb.append("\n");
                pwData.write(sb.toString());

                sb.setLength(0);
                sb.append(numSamples.get());
                sb.append(",");
                sb.append(trainTimeSeconds);
                sb.append(",");
                sb.append(testTimeSeconds);
                sb.append(",");
                sb.append(accuracy);
                sb.append("\n");
                pwMetrics.write(sb.toString());
            }

            consumer.commitAsync();

        }
        consumer.close();
        if (pwData != null){
            pwData.close();
        }
        if (pwMetrics != null){
            pwMetrics.close();
        }

        double accuracy = 100.0 * (double) numSamplesCorrect.get() / (double) numSamples.get();
        System.out.println(classifierName + " processed " + numSamples.get() +
                " instances from topic " + topicName + " with " + accuracy + "% accuracy.");
        System.out.println("DONE");

    }

    public static void main(String[] args) throws Exception{

        String topicName = "";
        String classifierName = "";
        String bootstrapServer = DEFAULT_BOOTSTRAP_SERVERS;
        boolean fromBeginning = false;
        String resultsPath = "";

        if (args.length == 5){
            topicName = args[0];
            System.out.println("Topic name: " + topicName);
            bootstrapServer = args[1];
            System.out.println("Bootstrap server: " + bootstrapServer);
            fromBeginning = Boolean.parseBoolean(args[2]);
            System.out.println("From beginning: " + fromBeginning);
            classifierName = args[3];
            System.out.println("Classifier: " + classifierName);
            resultsPath = args[4];
            System.out.println("Output results path:" + resultsPath);
        }else{
            System.out.println("Program arguments: [topicName] [fromBeginning] [classifierName] [outputDirPath]");
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
            runConsumer(topicName, classifierName, bootstrapServer, fromBeginning, resultsPath);
            latch.await();
        } catch (Throwable e) {
            System.out.println("Error processing topic " + topicName + " with classifier " + classifierName);
            e.printStackTrace();
        }
        System.exit(0);
    }

}
