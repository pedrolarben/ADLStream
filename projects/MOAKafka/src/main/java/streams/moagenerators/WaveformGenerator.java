package streams.moagenerators;

import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;

public class WaveformGenerator extends moa.streams.generators.WaveformGenerator implements MOAStreamGenerator {

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
