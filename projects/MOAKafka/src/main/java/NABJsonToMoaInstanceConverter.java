import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NABJsonToMoaInstanceConverter implements Serializable {

    private long timestamp;
    private String readableTimestamp;
    private double isAnomaly;
    private double value;
    private String datasetName;
    private String csvFileName;
    private double producerDelay;
    private long producerTimestamp;
    private String producerReadableTimestamp;

    public NABJsonToMoaInstanceConverter() {
    }

    public Instance moaInstance(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode jsonNode = mapper.readTree(json);

        // Read json values
        timestamp = jsonNode.get("timestamp").asLong();
        readableTimestamp = jsonNode.get("readable_timestamp").asText();
        isAnomaly = jsonNode.get("is_anomaly").asDouble();
        value = jsonNode.get("value").asDouble();
        datasetName = jsonNode.get("dataset").asText();
        csvFileName = jsonNode.get("csv_file").asText();
        producerDelay = jsonNode.get("producer_delay").asDouble();
        producerTimestamp = jsonNode.get("producer_timestamp").asLong();
        producerReadableTimestamp = jsonNode.get("producer_readable_timestamp").asText();

        String attrNames[] = new String[2];
        attrNames[0] = "value";
        attrNames[1] = "class";

        double attrValues[] = new double[2];
        attrValues[0] = value;
        attrValues[1] = isAnomaly;

        Instance instance = new DenseInstance(1.0, attrValues);

        List<Attribute> attInfo = new ArrayList<Attribute>();

        for(int i = 0; i < attrValues.length; ++i) {
            Attribute attr = new Attribute(attrNames[i]);
            attInfo.add(attr);
        }

        Instances instanceInformation = new Instances(datasetName, attInfo, 0);
        int classIndex = attrValues.length - 1;
        instanceInformation.setClassIndex(classIndex);

        instance.setDataset(instanceInformation);
        instance.setClassValue(attrValues[classIndex]);

        return instance;

    }

    public long getTimestamp() {
        return timestamp;
    }

    public String getReadableTimestamp() {
        return readableTimestamp;
    }

    public double getIsAnomaly() {
        return isAnomaly;
    }

    public double getValue() {
        return value;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public String getCsvFileName() {
        return csvFileName;
    }

    public double getProducerDelay() {
        return producerDelay;
    }

    public long getProducerTimestamp() {
        return producerTimestamp;
    }

    public String getProducerReadableTimestamp() {
        return producerReadableTimestamp;
    }
}
