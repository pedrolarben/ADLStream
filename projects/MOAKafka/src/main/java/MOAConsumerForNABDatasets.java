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

public class MOAConsumerForNABDatasets {

    private final static String BOOTSTRAP_SERVERS = "localhost:9092";

    private static Consumer<Long, String> createConsumer(String topic, boolean fromBeginning) {

        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, BOOTSTRAP_SERVERS);
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
    private static void runConsumer(String topicName, String classifierName, boolean fromBeginning,
                                    String sourceName, String resultsPath, String datasetName, String csvFileName) throws FileNotFoundException {
        final Consumer<Long, String> consumer = createConsumer(topicName, fromBeginning);
        final int giveUp = 100;
        int noRecordsCount = 0;

        final Classifier classifier = MOAClassifierFactory.getClassifier(classifierName);
        classifier.prepareForUse();
        AtomicInteger numSamples = new AtomicInteger();
        AtomicInteger numSamplesCorrect = new AtomicInteger();

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        String dirName = resultsPath + File.separator + "MOA_" +
                classifierName + File.separator + datasetName;
        File directory = new File(dirName);
        if (!directory.exists()) {
            directory.mkdirs();
        }
        String fileName = dirName + File.separator + "MOA_" + classifierName + "_" + csvFileName;
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new FileOutputStream(new File(fileName)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        StringBuilder sbHeader = new StringBuilder();
        sbHeader.append("timestamp");
        sbHeader.append(",");
        sbHeader.append("value");
        sbHeader.append(",");
        sbHeader.append("anomaly_score");
        sbHeader.append(",");
        sbHeader.append("label");
        sbHeader.append(",");
        sbHeader.append("accuracy");
        sbHeader.append(",");
        sbHeader.append("training_time_ns");
        sbHeader.append(",");
        sbHeader.append("test_time_ns");
        sbHeader.append("\n");
        pw.write(sbHeader.toString());

        while(true){
            final ConsumerRecords<Long, String> consumerRecords = consumer.poll(10);

            if(consumerRecords.count()==0){
                noRecordsCount++;
                if(noRecordsCount > giveUp) break;
                else continue;
            }

            for (ConsumerRecord<Long, String> record : consumerRecords) {
                String value = record.value();
                Instance instance = null;
                Object converter = null;
                if (sourceName.equalsIgnoreCase("MOA")) {
                    json.instance.Instance jsonInstance = null;
                    try {
                        jsonInstance = mapper.readValue(value, json.instance.Instance.class);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    if (jsonInstance != null) {
                        converter = new MoaJsonToMoaInstanceConverter();
                        instance = ((MoaJsonToMoaInstanceConverter) converter).moaInstance(jsonInstance);
                    }
                } else if (sourceName.equalsIgnoreCase("NAB")) {
                    converter = new NABJsonToMoaInstanceConverter();
                    try {
                        instance = ((NABJsonToMoaInstanceConverter) converter).moaInstance(value);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                long testStartTime = System.nanoTime();
                if (classifier.correctlyClassifies(instance)) {
                    numSamplesCorrect.incrementAndGet();
                }
                long testEndTime = System.nanoTime();
                long testDuration = (testEndTime - testStartTime);

                long trainingStartTime = System.nanoTime();
                double anomalyScore = Utils.maxIndex(classifier.getVotesForInstance(instance));
                classifier.trainOnInstance(instance);
                long trainingEndTime = System.nanoTime();
                long trainingDuration = (trainingEndTime - trainingStartTime);
                numSamples.getAndIncrement();
                double accuracy = 100.0 * (double) numSamplesCorrect.get() / numSamples.get();
                if (converter instanceof NABJsonToMoaInstanceConverter) {
                    NABJsonToMoaInstanceConverter nabConverter = (NABJsonToMoaInstanceConverter) converter;
                    if (csvFileName.equalsIgnoreCase(nabConverter.getCsvFileName())){

                        StringBuilder sbLine = new StringBuilder();
                        sbLine.append(nabConverter.getReadableTimestamp());
                        sbLine.append(",");
                        sbLine.append(nabConverter.getValue());
                        sbLine.append(",");
                        sbLine.append(anomalyScore);
                        sbLine.append(",");
                        sbLine.append(instance.classValue());
                        sbLine.append(",");
                        sbLine.append(accuracy);
                        sbLine.append(",");
                        sbLine.append(trainingDuration);
                        sbLine.append(",");
                        sbLine.append(testDuration);
                        sbLine.append("\n");

                        pw.write(sbLine.toString());
                    }
                }
            }

            consumer.commitAsync();

        }
        consumer.close();
        pw.close();
        double accuracy = 100.0 * (double) numSamplesCorrect.get() / (double) numSamples.get();
        System.out.println(numSamples.get() + " instances processed with " + accuracy + "% accuracy and " + classifierName + " as classifier.");
        System.out.println(classifierName + "," + accuracy);
        System.out.println("DONE");
        if (sourceName.equalsIgnoreCase("NAB")){
            PrintWriter pwResults = new PrintWriter(new FileOutputStream(
                    new File(resultsPath + File.separator + "accuracy_results.csv"), true));
            StringBuilder sb = new StringBuilder();
            sb.append(datasetName);
            sb.append(",");
            sb.append(csvFileName);
            sb.append(",");
            sb.append(classifierName);
            sb.append(",");
            sb.append(accuracy);
            sb.append("\n");

            pwResults.write(sb.toString());
            pwResults.close();
        }

    }

    public static void main(String[] args) throws Exception{

        List<String> classifiers = Arrays.asList(
                "ActiveClassifier",
                "NaiveBayes",
                //"NaiveBayesMultinomial",
                "SingleClassifierDrift",
                "MajorityClass",
                "Perceptron",
                "SGD",
                "SPegasos",
                //"AccuracyWeightedEnsemble",
                //"AccuracyUpdatedEnsemble",
                "LeveragingBag",
                //"LimAttClassifier",
                "OCBoost",
                "OzaBag",
                "OzaBagASHT",
                "OzaBagAdwin",
                "OzaBoost",
                "OzaBoostAdwin",
                "WeightedMajorityAlgorithm",
                "WEKAClassifier",
                "DecisionStump",
                "HoeffdingOptionTree",
                "AdaHoeffdingOptionTree",
                "HoeffdingTree",
                "ASHoeffdingTree",
                "HoeffdingAdaptiveTree",
                //"LimAttHoeffdingTree",
                "RandomHoeffdingTree"
        );

        List<String> realKnownCause = Arrays.asList(
                "ambient_temperature_system_failure.csv",
                "cpu_utilization_asg_misconfiguration.csv",
                "ec2_request_latency_system_failure.csv",
                "machine_temperature_system_failure.csv",
                "nyc_taxi.csv",
                "rogue_agent_key_hold.csv",
                "rogue_agent_key_updown.csv"
        );

        List<String> artificialNoAnomaly = Arrays.asList(
                "art_daily_no_noise.csv",
                "art_daily_perfect_square_wave.csv",
                "art_daily_small_noise.csv",
                "art_flatline.csv",
                "art_noisy.csv"
        );

        List<String> artificialWithAnomaly = Arrays.asList(
                "art_daily_flatmiddle.csv",
                "art_daily_jumpsdown.csv",
                "art_daily_jumpsup.csv",
                "art_daily_nojump.csv",
                "art_increase_spike_density.csv",
                "art_load_balancer_spikes.csv"
        );

        List<String> realAWSCloudwatch = Arrays.asList(
                "ec2_cpu_utilization_24ae8d.csv",
                "ec2_cpu_utilization_53ea38.csv",
                "ec2_cpu_utilization_5f5533.csv",
                "ec2_cpu_utilization_77c1ca.csv",
                "ec2_cpu_utilization_825cc2.csv",
                "ec2_cpu_utilization_ac20cd.csv",
                "ec2_cpu_utilization_c6585a.csv",
                "ec2_cpu_utilization_fe7f93.csv",
                "ec2_disk_write_bytes_1ef3de.csv",
                "ec2_disk_write_bytes_c0d644.csv",
                "ec2_network_in_257a54.csv",
                "ec2_network_in_5abac7.csv",
                "elb_request_count_8c0756.csv",
                "grok_asg_anomaly.csv",
                "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
                "rds_cpu_utilization_cc0c53.csv",
                "rds_cpu_utilization_e47b3b.csv"
        );

        List<String> realAdExchange = Arrays.asList(
                "exchange-2_cpc_results.csv",
                "exchange-2_cpm_results.csv",
                "exchange-3_cpc_results.csv",
                "exchange-3_cpm_results.csv",
                "exchange-4_cpc_results.csv",
                "exchange-4_cpm_results.csv"
        );

        List<String> realTraffic = Arrays.asList(
                "occupancy_6005.csv",
                "occupancy_t4013.csv",
                "speed_6005.csv",
                "speed_7578.csv",
                "speed_t4013.csv",
                "TravelTime_387.csv",
                "TravelTime_451.csv"
        );

        List<String> realTweets = Arrays.asList(
                "Twitter_volume_AAPL.csv",
                "Twitter_volume_AMZN.csv",
                "Twitter_volume_CRM.csv",
                "Twitter_volume_CVS.csv",
                "Twitter_volume_FB.csv",
                "Twitter_volume_GOOG.csv",
                "Twitter_volume_IBM.csv",
                "Twitter_volume_KO.csv",
                "Twitter_volume_PFE.csv",
                "Twitter_volume_UPS.csv"
        );

        Map<String, List<String>> datasetsMap = new HashMap<>();
        datasetsMap.put("realKnownCause", realKnownCause);
        datasetsMap.put("artificialNoAnomaly", artificialNoAnomaly);
        datasetsMap.put("artificialWithAnomaly", artificialWithAnomaly);
        datasetsMap.put("realAWSCloudwatch", realAWSCloudwatch);
        datasetsMap.put("realAdExchange", realAdExchange);
        datasetsMap.put("realTraffic", realTraffic);
        datasetsMap.put("realTweets", realTweets);

        String topicName = "";
        boolean fromBeginning = false;
        String sourceName = "";
        String resultsPath = "";

        if (args.length == 4){
            topicName = args[0];
            System.out.println("Topic name: " + topicName);
            fromBeginning = Boolean.parseBoolean(args[1]);
            System.out.println("From beginning: " + fromBeginning);
            sourceName = args[2];
            System.out.println("Source name: " + sourceName);
            resultsPath = args[3];
            System.out.println("Output results path:" + resultsPath);
        }else{
            System.out.println("Program arguments: [topicName] [fromBeginning] [sourceName] [outputResultsPath]");
            System.out.println("-classifierName: Any class name that implements the moa.classifiers.Classifier interface. \n" +
                    "For example: HoeffdingTree\n" +
                    "See https://www.cs.waikato.ac.nz/~abifet/MOA/API/interfacemoa_1_1classifiers_1_1_classifier.html");
            System.out.println("-sourceName: MOA or NAB");
            System.out.println("-outputResultsPath: Path where results will be saved.");
            System.exit(1);
        }

        if(!sourceName.equalsIgnoreCase("MOA") && !sourceName.equalsIgnoreCase("NAB")){
            System.out.println("Source name must be MOA or NAB");
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
            for (String datasetName : datasetsMap.keySet()){
                for (String csvFileName : datasetsMap.get(datasetName)){
                    for (String classifier : classifiers){
                        System.out.println("Classifying NAB data with " + classifier);
                        try{
                            runConsumer(topicName, classifier, fromBeginning, sourceName, resultsPath,
                                    datasetName, csvFileName);
                        }catch (Exception e){
                            System.out.println("Processing error with classifier: " + classifier);
                        }
                    }
                }
            }
            System.exit(0);

            latch.await();
        } catch (Throwable e) {
            e.printStackTrace();
            System.exit(1);
        }
        System.exit(0);
    }

}
