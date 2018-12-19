package streams.moagenerators;

import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;

public class ArffFileStream extends moa.streams.ArffFileStream implements MOAStreamGenerator {

    public ArffFileStream(String arffFileName, int classIndex){
        super(arffFileName, classIndex);
    }

    public void prepareForUse(){
        super.prepareForUse();
    }

    public InstancesHeader getHeader() {
        return super.getHeader();
    }

    public boolean hasMoreInstances(){
        return super.hasMoreInstances();
    }

    public InstanceExample nextInstance(){
        return super.nextInstance();
    }
}
