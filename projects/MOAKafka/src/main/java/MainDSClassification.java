import org.apache.log4j.BasicConfigurator;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainDSClassification {

    private static List<String> CLASSIFIERS = Arrays.asList("KUE", "AdaptiveRandomForest", "ActiveClassifier","NaiveBayes","SingleClassifierDrift","MajorityClass","Perceptron","SGD","SPegasos","LeveragingBag","OCBoost","OzaBag","OzaBagASHT","OzaBagAdwin","OzaBoost","OzaBoostAdwin","WeightedMajorityAlgorithm","DecisionStump","HoeffdingOptionTree","AdaHoeffdingOptionTree","HoeffdingTree","ASHoeffdingTree","HoeffdingAdaptiveTree","RandomHoeffdingTree");
//    private static List<String> CLASSIFIERS = Arrays.asList("KUE");//, "AdaptiveRandomForest", "LeveragingBag", "NaiveBayes", "HoeffdingTree", "HoeffdingAdaptiveTree");
    private static List<String> TOPICS = Arrays.asList("streams_airlinesnorm", "streams_bankmarketingnorm", "streams_breast", "streams_carnorm", "streams_covtype", "streams_electricity", "streams_higgs200k", "streams_kddcup99norm", "streams_ozone", "streams_pendigits", "streams_pokerhand", "streams_spambase", "streams_susy100k", "streams_DiatomSizeReduction", "streams_DistalPhalanxOutlineAgeGroup", "streams_DistalPhalanxOutlineCorrect", "streams_DistalPhalanxTW", "streams_ECG200", "streams_ECG5000", "streams_Earthquakes", "streams_ElectricDevices", "streams_FaceAll", "streams_FaceFour", "streams_FacesUCR", "streams_FiftyWords", "streams_Fish", "streams_FordA", "streams_FordB", "streams_GunPoint", "streams_Ham", "streams_HandOutlines", "streams_Haptics", "streams_Herring", "streams_InlineSkate", "streams_InsectWingbeatSound", "streams_ItalyPowerDemand", "streams_LargeKitchenAppliances", "streams_Lightning2", "streams_Lightning7", "streams_Mallat", "streams_Meat", "streams_MedicalImages", "streams_MiddlePhalanxOutlineAgeGroup", "streams_MiddlePhalanxOutlineCorrect", "streams_MiddlePhalanxTW", "streams_MoteStrain", "streams_NonInvasiveFetalECGThorax1", "streams_NonInvasiveFetalECGThorax2", "streams_OSULeaf", "streams_OliveOil", "streams_PhalangesOutlinesCorrect", "streams_Phoneme", "streams_Plane", "streams_ProximalPhalanxOutlineAgeGroup", "streams_ProximalPhalanxOutlineCorrect", "streams_ProximalPhalanxTW", "streams_RefrigerationDevices", "streams_ScreenType", "streams_ShapeletSim", "streams_ShapesAll", "streams_SmallKitchenAppliances", "streams_SonyAIBORobotSurface1", "streams_SonyAIBORobotSurface2", "", "streams_Strawberry", "streams_SwedishLeaf", "streams_Symbols", "streams_SyntheticControl", "streams_ToeSegmentation1", "streams_ToeSegmentation2", "streams_Trace", "streams_TwoLeadECG", "streams_TwoPatterns", "", "streams_UWaveGestureLibraryX", "streams_UWaveGestureLibraryY", "streams_UWaveGestureLibraryZ", "streams_Wafer", "streams_Wine", "streams_WordSynonyms", "streams_Worms", "streams_Yoga", "streams_Adiac", "streams_ArrowHead", "streams_Beef", "streams_BeetleFly", "streams_BirdChicken", "streams_CBF", "streams_Car", "streams_ChlorineConcentration", "streams_CinCECGtorso", "streams_Coffee", "streams_Computers", "streams_ECGFiveDays");
//    private static List<String> TOPICS = Arrays.asList( "streams_RTGg", "streams_ARGWg-F2F5F8", "streams_ARGWa-F3F6F3F6", "streams_ARGWa-F2F5F8", "streams_ARGWg-F1F10", "streams_RBFi-slow", "streams_RBFi-fast", "streams_RTGa3", "streams_RTGa", "streams_RTGg3", "streams_SEAg-F2F4", "streams_SEAa-F2F4", "streams_ARGWa-F1F10", "streams_ARGWg-F3F6F3F6", "streams_RTGgD", "streams_RTGaD", "streams_RTGg3D", "streams_RTGa3D" );
    private static String DEFAULT_RESULTS_PATH = "../DSClassificationResults/DSClassificationResults_MOA";

    public static void main(String[] args){

        String bootstrapServer = "";
        String outputPath = DEFAULT_RESULTS_PATH;
        if(args.length == 2){
            bootstrapServer = args[0];
            outputPath = args[1];
        }

        BasicConfigurator.configure();
        ExecutorService executor = Executors.newFixedThreadPool(10);
        for(String clf: CLASSIFIERS){
            for(String topic: TOPICS){
                Runnable worker = new WorkerThread(topic, clf, bootstrapServer,true, outputPath);
                executor.execute(worker);
            }
        }
        executor.shutdown();
        while (!executor.isTerminated()){}
        System.out.println("Finished all threads");
    }

    private static class WorkerThread implements Runnable{

        private String topicName;
        private String classifierName;
        private boolean fromBeginning;
        private String resultsPath;
        private String bootstapServer;

        private WorkerThread(String topicName, String classifierName, String bootstapServer, boolean fromBeginning, String resultsPath) {
            this.topicName = topicName;
            this.classifierName = classifierName;
            this.fromBeginning = fromBeginning;
            this.resultsPath = resultsPath;
            this.bootstapServer = bootstapServer;
        }

        @Override
        public void run() {
            DSClassificationMOAConsumer.runConsumer(topicName, classifierName, bootstapServer ,fromBeginning, resultsPath);
        }
    }
}
