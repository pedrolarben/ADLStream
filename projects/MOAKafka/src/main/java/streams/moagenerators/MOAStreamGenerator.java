package streams.moagenerators;

import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;

public interface MOAStreamGenerator {

    void prepareForUse();
    InstancesHeader getHeader();
    boolean hasMoreInstances();
    InstanceExample nextInstance();
}
