package streams.moagenerators;

import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;

public class LEDGenerator extends moa.streams.generators.LEDGenerator implements MOAStreamGenerator {

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
