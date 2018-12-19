import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.TimingUtils;
import moa.streams.generators.RandomRBFGenerator;

import java.io.IOException;

public class Experiment {

    public Experiment(){}

    public void run(int numInstances, boolean isTesting){
        Classifier classifier = new HoeffdingTree();
        RandomRBFGenerator stream = new RandomRBFGenerator();
        stream.prepareForUse();

        classifier.setModelContext(stream.getHeader());
        classifier.prepareForUse();

        int numberSamplesCorrect = 0;
        int numberSamples = 0;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime= TimingUtils.getNanoCPUTimeOfCurrentThread();
        while (stream.hasMoreInstances() && numberSamples < numInstances) {
            Instance trainInst = stream.nextInstance().getData();
            if(isTesting){
                if(classifier.correctlyClassifies(trainInst)){
                    numberSamplesCorrect++;
                }
            }
            numberSamples++;
            classifier.trainOnInstance(trainInst);
        }
        double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
        double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
        System.out.println(numberSamples + " instances processed with " + accuracy + "% accuracy in "+time+" seconds.");

    }

    public static void main(String[] args) throws IOException {
        Experiment exp = new Experiment();
        exp.run(100, true);
    }

}
