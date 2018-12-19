import moa.classifiers.Classifier;
import moa.classifiers.active.ActiveClassifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.bayes.NaiveBayesMultinomial;
import moa.classifiers.drift.SingleClassifierDrift;
import moa.classifiers.functions.MajorityClass;
import moa.classifiers.functions.Perceptron;
import moa.classifiers.functions.SGD;
import moa.classifiers.functions.SPegasos;
import moa.classifiers.meta.*;
import moa.classifiers.trees.*;

public class MOAClassifierFactory {

    public static Classifier getClassifier(String name){

        if(name.equals(ActiveClassifier.class.getSimpleName())){
            return new ActiveClassifier();
        }else if(name.equals(NaiveBayes.class.getSimpleName())){
            return new NaiveBayes();
        }else if(name.equals(NaiveBayesMultinomial.class.getSimpleName())){
            return new NaiveBayesMultinomial();
        }else if(name.equals(SingleClassifierDrift.class.getSimpleName())){
            return new SingleClassifierDrift();
        }else if(name.equals(MajorityClass.class.getSimpleName())){
            return new MajorityClass();
        }else if(name.equals(Perceptron.class.getSimpleName())){
            return new Perceptron();
        }else if(name.equals(SGD.class.getSimpleName())){
            return new SGD();
        }else if(name.equals(SPegasos.class.getSimpleName())){
            return new SPegasos();
        }else if(name.equals(AccuracyWeightedEnsemble.class.getSimpleName())){
            return new AccuracyWeightedEnsemble();
        }else if(name.equals(AccuracyUpdatedEnsemble.class.getSimpleName())){
            return new AccuracyUpdatedEnsemble();
        }else if(name.equals(LeveragingBag.class.getSimpleName())){
            return new LeveragingBag();
        }else if(name.equals(LimAttClassifier.class.getSimpleName())){
            return new LimAttClassifier();
        }else if(name.equals(OCBoost.class.getSimpleName())){
            return new OCBoost();
        }else if(name.equals(OzaBag.class.getSimpleName())){
            return new OzaBag();
        }else if(name.equals(OzaBagASHT.class.getSimpleName())){
            return new OzaBagASHT();
        }else if(name.equals(OzaBagAdwin.class.getSimpleName())){
            return new OzaBagAdwin();
        }else if(name.equals(OzaBoost.class.getSimpleName())){
            return new OzaBoost();
        }else if(name.equals(OzaBoostAdwin.class.getSimpleName())){
            return new OzaBoostAdwin();
        }else if(name.equals(WeightedMajorityAlgorithm.class.getSimpleName())){
            return new WeightedMajorityAlgorithm();
        }else if(name.equals(WEKAClassifier.class.getSimpleName())){
            return new WEKAClassifier();
        }else if(name.equals(DecisionStump.class.getSimpleName())){
            return new DecisionStump();
        }else if(name.equals(HoeffdingOptionTree.class.getSimpleName())){
            return new HoeffdingOptionTree();
        }else if(name.equals(AdaHoeffdingOptionTree.class.getSimpleName())){
            return new AdaHoeffdingOptionTree();
        }else if(name.equals(HoeffdingTree.class.getSimpleName())){
            return new HoeffdingTree();
        }else if(name.equals(ASHoeffdingTree.class.getSimpleName())){
            return new ASHoeffdingTree();
        }else if(name.equals(HoeffdingAdaptiveTree.class.getSimpleName())){
            return new HoeffdingAdaptiveTree();
        }else if(name.equals(LimAttHoeffdingTree.class.getSimpleName())){
            return new LimAttHoeffdingTree();
        }else if(name.equals(RandomHoeffdingTree.class.getSimpleName())){
            return new RandomHoeffdingTree();
        }else{
            System.out.println("Classifier not found. Returning an instance of HoeffdingTree");
            return new HoeffdingTree();
        }
    }

}
