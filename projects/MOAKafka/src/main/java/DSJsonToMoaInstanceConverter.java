import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

public class DSJsonToMoaInstanceConverter implements Serializable {

    private String datasetName;
    private List<String> keysList = new ArrayList<>();
    private List<Double> valuesList =  new ArrayList<>();

    public DSJsonToMoaInstanceConverter(String datasetName) {
        this.datasetName = datasetName;
    }

    public Instance moaInstance(String json) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode rootNode = mapper.readTree(json);

        for (Iterator<String> it = rootNode.fieldNames(); it.hasNext(); ) {
            String key = it.next();

            if(!key.equalsIgnoreCase("class") && !key.equalsIgnoreCase("classes")){
                keysList.add(key);
                valuesList.add(rootNode.get(key).asDouble());
            }
        }

        // Add 'class' key and value
        keysList.add("class");
        Double class_value = rootNode.get("class").asDouble();
        if(class_value == -1.){
            class_value = 0.;
        }
        valuesList.add(class_value);

        String attrNames[] = keysList.toArray(new String[0]);
        double attrValues[] = valuesList.stream().mapToDouble((d -> d)).toArray();

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

    public List<String> getKeysList() {
        return keysList;
    }

    public List<Double> getValuesList() {
        return valuesList;
    }

}
