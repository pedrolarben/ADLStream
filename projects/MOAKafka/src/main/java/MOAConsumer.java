import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import org.apache.kafka.clients.consumer.Consumer;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.LongDeserializer;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.io.IOException;
import java.util.Collections;
import java.util.Properties;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

public class MOAConsumer {

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
    private static void runConsumer(String topicName, String classifierName, boolean fromBeginning, String sourceName) {
        final Consumer<Long, String> consumer = createConsumer(topicName, fromBeginning);
        final int giveUp = 100;
        int noRecordsCount = 0;

        final Classifier classifier = MOAClassifierFactory.getClassifier(classifierName);
        classifier.prepareForUse();
        AtomicInteger numSamples = new AtomicInteger();
        AtomicInteger numSamplesCorrect = new AtomicInteger();

        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        while(true){
            final ConsumerRecords<Long, String> consumerRecords = consumer.poll(1000);

            if(consumerRecords.count()==0){
                noRecordsCount++;
                if(noRecordsCount > giveUp) break;
                else continue;
            }

            consumerRecords.forEach(record -> {
                String value = record.value();
                Instance instance = null;
                if (sourceName.equalsIgnoreCase("MOA")){
                    json.instance.Instance jsonInstance = null;
                    try {
                        jsonInstance = mapper.readValue(value,json.instance.Instance.class);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    if (jsonInstance != null){
                        MoaJsonToMoaInstanceConverter converter = new MoaJsonToMoaInstanceConverter();
                        instance = converter.moaInstance(jsonInstance);
                    }
                }else if (sourceName.equalsIgnoreCase("NAB")){
                    NABJsonToMoaInstanceConverter converter = new NABJsonToMoaInstanceConverter();
                    try {
                        instance = converter.moaInstance(value);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                if(classifier.correctlyClassifies(instance)){
                    numSamplesCorrect.incrementAndGet();
                }
                classifier.trainOnInstance(instance);
                numSamples.getAndIncrement();
                double accuracy = 100.0 * (double) numSamplesCorrect.get() / numSamples.get();
                System.out.println(numSamples.get() + " instances processed with " + accuracy + "% accuracy.");

            });

            consumer.commitAsync();

        }
        consumer.close();
        double accuracy = 100.0 * (double) numSamplesCorrect.get() / (double) numSamples.get();
        System.out.println(numSamples.get() + " instances processed with " + accuracy + "% accuracy.");
        System.out.println("DONE");

    }

    public static void main(String[] args) throws Exception{

        String topicName = "";
        String classifierName = "";
        boolean fromBeginning = false;
        String sourceName = "";

        if (args.length == 4){
            topicName = args[0];
            System.out.println("Topic name: " + topicName);
            fromBeginning = Boolean.parseBoolean(args[1]);
            System.out.println("From beginning: " + fromBeginning);
            classifierName = args[2];
            System.out.println("Classifier name: " + classifierName);
            sourceName = args[3];
            System.out.println("Source name " + sourceName);
        }else{
            System.out.println("Program arguments: [topicName] [fromBeginning] [classifierName] [sourceName]");
            System.out.println("-classifierName: Any class name that implements the moa.classifiers.Classifier interface. \n" +
                    "For example: HoeffdingTree\n" +
                    "See https://www.cs.waikato.ac.nz/~abifet/MOA/API/interfacemoa_1_1classifiers_1_1_classifier.html");
            System.out.println("-sourceName: MOA or NAB");
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
            runConsumer(topicName, classifierName, fromBeginning, sourceName);
            latch.await();
        } catch (Throwable e) {
            e.printStackTrace();
            System.exit(1);
        }
        System.exit(0);
    }

}
