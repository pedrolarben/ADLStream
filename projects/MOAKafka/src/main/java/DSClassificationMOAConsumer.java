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
    public final static int CHUNK_SIZE = 10;

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
        sbHeader.append("train_time_S");
        sbHeader.append(",");
        sbHeader.append("test_time_s");
        sbHeader.append(",");
        sbHeader.append("accuracy");
        sbHeader.append("\n");
        pwMetrics.write(sbHeader.toString());

        boolean dataHeaderWritten = false;

        AtomicInteger numSamples = new AtomicInteger();
        List<ConsumerRecord<Long, String>> chunkInstances =  new ArrayList<ConsumerRecord<Long, String>>();

        while(true){
            final ConsumerRecords<Long, String> consumerRecords = consumer.poll(1000);

            if(consumerRecords.count()==0){
                noRecordsCount++;
                if(noRecordsCount > giveUp) break;
                else continue;
            }
            for ( ConsumerRecord<Long, String> record : consumerRecords) {
                //add record to chunk
                chunkInstances.add(record);

                // if chunk completed: test-then-train
                if( chunkInstances.size() % CHUNK_SIZE == 0){

                    AtomicInteger numSamplesCorrect = new AtomicInteger();
                    StringBuilder sb = new StringBuilder();

                    // Test every instance of the chunk
                    long testTime = 0l;
                    for(ConsumerRecord<Long, String> r: chunkInstances ) {
                        //Transform json string to instance
                        String value = r.value();
                        DSJsonToMoaInstanceConverter converter = new DSJsonToMoaInstanceConverter(topicName);
                        Instance instance = null;
                        try {
                            instance = converter.moaInstance(value);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        // Check accuracy of the chunk
                        if (classifier.correctlyClassifies(instance)) {
                            numSamplesCorrect.incrementAndGet();
                        }
                        numSamples.incrementAndGet();

                        // Get prediction value and count the test time
                        long testStartTime = System.nanoTime();
                        double prediction = Utils.maxIndex(classifier.getVotesForInstance(instance));
                        long testEndTime = System.nanoTime();
                        testTime += (testEndTime - testStartTime);

                        // Add row to data file (values, class and predicted class)
                        for (Double v : converter.getValuesList()) {
                            sb.append(v);
                            sb.append(",");
                        }
                        sb.append(prediction);
                        sb.append("\n");
                    }
                    pwData.write(sb.toString());
                    double testTimeSeconds = (double)testTime / 1_000_000_000.0;
                    double accuracy = (double) numSamplesCorrect.get() / chunkInstances.size();

                    // Train chunk
                    long trainTime = 0l;
                    for(ConsumerRecord<Long, String> r: chunkInstances){
                        //Transform json string to instance
                        String value = r.value();
                        DSJsonToMoaInstanceConverter converter = new DSJsonToMoaInstanceConverter(topicName);
                        Instance instance = null;
                        try {
                            instance = converter.moaInstance(value);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        long trainStartTime = System.nanoTime();
                        classifier.trainOnInstance(instance);
                        long trainEndTime = System.nanoTime();
                        trainTime += (trainEndTime - trainStartTime);
                    }
                    double trainTimeSeconds = (double)trainTime / 1_000_000_000.0;

                    // Add chunk metrics to a new row in metrics file
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

                    // restart chunk
                    chunkInstances =  new ArrayList<ConsumerRecord<Long, String>>();
                }

            }

            consumer.commitAsync();

        }

        // If the last chunk is incomplete:
        if(!chunkInstances.isEmpty()){

            AtomicInteger numSamplesCorrect = new AtomicInteger();
            StringBuilder sb = new StringBuilder();

            // Test every instance of the chunk
            long testTime = 0l;
            for(ConsumerRecord<Long, String> r: chunkInstances ) {
                //Transform json string to instance
                String value = r.value();
                DSJsonToMoaInstanceConverter converter = new DSJsonToMoaInstanceConverter(topicName);
                Instance instance = null;
                try {
                    instance = converter.moaInstance(value);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                // Check accuracy of the chunk
                if (classifier.correctlyClassifies(instance)) {
                    numSamplesCorrect.incrementAndGet();
                }
                numSamples.incrementAndGet();

                // Get prediction value and count the test time
                long testStartTime = System.nanoTime();
                double prediction = Utils.maxIndex(classifier.getVotesForInstance(instance));
                long testEndTime = System.nanoTime();
                testTime += (testEndTime - testStartTime);

                // Add row to data file (values, class and predicted class)
                for (Double v : converter.getValuesList()) {
                    sb.append(v);
                    sb.append(",");
                }
                sb.append(prediction);
                sb.append("\n");
            }
            pwData.write(sb.toString());
            double testTimeSeconds = (double)testTime / 1_000_000_000.0;
            double accuracy = (double) numSamplesCorrect.get() / chunkInstances.size();

            // Train chunk
            long trainTime = 0l;
            for(ConsumerRecord<Long, String> r: chunkInstances){
                //Transform json string to instance
                String value = r.value();
                DSJsonToMoaInstanceConverter converter = new DSJsonToMoaInstanceConverter(topicName);
                Instance instance = null;
                try {
                    instance = converter.moaInstance(value);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                long trainStartTime = System.nanoTime();
                classifier.trainOnInstance(instance);
                long trainEndTime = System.nanoTime();
                trainTime += (trainEndTime - trainStartTime);
            }
            double trainTimeSeconds = (double)trainTime / 1_000_000_000.0;

            // Add chunk metrics to a new row in metrics file
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

        consumer.close();
        if (pwData != null){
            pwData.close();
        }
        if (pwMetrics != null){
            pwMetrics.close();
        }

        String accuracy = "UNKNOWN";//100.0 * (double) numSamplesCorrect.get() / (double) numSamples.get();
        System.out.println(classifierName + " processed " + numSamples.get() +
                " instances from topic " + topicName + " with " + accuracy + " accuracy.");
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
