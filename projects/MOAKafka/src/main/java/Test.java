import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.Classifier;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Test {

    public static void main(String[] args) throws IOException {
        String json = "{\"is_anomaly\": 0," +
                "\"timestamp\": 1427057273," +
                "\"value\": 14," +
                "\"dataset\":\"NAB_realTweets\"}";
        ObjectMapper mapper = new ObjectMapper();
        JsonNode jsonNode = mapper.readTree(json);
        long timestamp = jsonNode.get("timestamp").asLong();
        double isAnomaly = jsonNode.get("is_anomaly").asDouble();
        double value = jsonNode.get("value").asDouble();
        String datasetName = jsonNode.get("dataset").asText();

        String attrNames[] = new String[2];
        attrNames[0] = "value";
        attrNames[1] = "class";

        double weight = 1.0;
        double attrValues[] = new double[2];
        attrValues[0] = value;
        attrValues[1] = isAnomaly;

        int indexValues[] = new int[2];
        for(int i = 0; i < indexValues.length; i++){
            indexValues[i] = i;
        }

        Instance instance = new DenseInstance(weight, attrValues);

        List<Attribute> attInfo = new ArrayList();

        for(int i = 0; i < attrValues.length; ++i) {
            Attribute attr = new Attribute(attrNames[i]);
            attInfo.add(attr);
        }

        Instances instanceInformation = new Instances(datasetName, attInfo, 0);
        int classIndex = attrValues.length - 1;
        instanceInformation.setClassIndex(classIndex);

        instance.setDataset(instanceInformation);
        instance.setClassValue(attrValues[classIndex]);

        final Classifier classifier = MOAClassifierFactory.getClassifier("HoeffdingTree");
        classifier.prepareForUse();

        if(classifier.correctlyClassifies(instance)){
            System.out.println("CORRECTO");
        }
        classifier.trainOnInstance(instance);

        System.out.println("Clasificado");


    }
}
