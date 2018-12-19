import com.yahoo.labs.samoa.instances.*;
import json.instance.InstanceHeader;
import org.apache.commons.lang3.ArrayUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class MoaJsonToMoaInstanceConverter implements Serializable{
    protected Instances moaInstanceInformation;

    public MoaJsonToMoaInstanceConverter() {
    }

    public Instance moaInstance(json.instance.Instance instance) {
        Object moaInstance = null;
        double[] attrValues = ArrayUtils.toPrimitive(instance.getInstanceData().getAttributeValues()
                .stream()
                .toArray(Double[]::new));
        int[] indexValues = ArrayUtils.toPrimitive(instance.getInstanceHeader()
                .getInstanceInformation().getAttributesInformation().getIndexValues()
                .stream()
                .toArray(Integer[]::new));

        if (instance.getJavaClass().equals(SparseInstance.class.getName())) {
            moaInstance = new SparseInstance(instance.getWeight(), attrValues, indexValues, attrValues.length);
        } else if (instance.getJavaClass().equals(DenseInstance.class.getName())) {
            moaInstance = new DenseInstance(instance.getWeight(), attrValues);
        } else if (instance.getJavaClass().equals(InstanceImpl.class.getName())){
            moaInstance = new InstanceImpl(instance.getWeight(), attrValues, indexValues, attrValues.length);
        }

        if (this.moaInstanceInformation == null) {
            this.moaInstanceInformation = this.moaInstancesInformation(instance);
        }

        ((Instance)moaInstance).setDataset(this.moaInstanceInformation);
        ((Instance)moaInstance).setClassValue(instance.getInstanceData().getAttributeValues().get(
                instance.getInstanceHeader().getInstanceInformation().getClassIndex()));
        return (Instance) moaInstance;
    }

    public Instances moaInstances(json.instance.Instance instance) {
        Instances moaInstances = moaInstancesInformation(instance);
        this.moaInstanceInformation = moaInstances;

        for(int i = 0; i < instance.getInstanceHeader().getInstances().size(); ++i) {
            moaInstances.add(moaInstance((json.instance.Instance)instance.getInstanceHeader().getInstances().get(i)));
        }

        return moaInstances;
    }

    public Instances moaInstancesInformation(json.instance.Instance instance) {
        InstanceHeader header = instance.getInstanceHeader();
        List<Attribute> attInfo = new ArrayList();

        for(int i = 0; i < header.getInstanceInformation().getAttributesInformation().getNumberAttributes(); ++i) {
            attInfo.add(moaAttribute(i, header.getInstanceInformation().getAttributesInformation().getAttributes().get(i)));
        }

        Instances moaInstances = new Instances(header.getRelationName(), attInfo, 0);
        moaInstances.setClassIndex(header.getInstanceInformation().getClassIndex());
        return moaInstances;
    }

    protected Attribute moaAttribute(int index, json.instance.Attribute attribute) {
        Attribute moaAttribute;
        if (attribute.getIsNominal()) {
            moaAttribute = new Attribute(attribute.getName(), attribute.getAttributeValues());
        } else {
            moaAttribute = new Attribute(attribute.getName());
        }

        return moaAttribute;
    }

}
