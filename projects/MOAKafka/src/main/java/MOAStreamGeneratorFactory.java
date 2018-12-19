import streams.moagenerators.*;

public class MOAStreamGeneratorFactory {

    public static MOAStreamGenerator getStreamGenerator(String name){

        ClassLoader classLoader = ClassLoader.getSystemClassLoader();

        if(name.equals(moa.streams.generators.AgrawalGenerator.class.getSimpleName())){
            return new AgrawalGenerator();
        }else if(name.equals(moa.streams.generators.HyperplaneGenerator.class.getSimpleName())) {
            return new HyperplaneGenerator();
        }else if(name.equals(moa.streams.generators.LEDGenerator.class.getSimpleName())) {
            return new LEDGenerator();
        }else if(name.equals(moa.streams.generators.LEDGeneratorDrift.class.getSimpleName())) {
            return new LEDGeneratorDrift();
        }else if(name.equals(moa.streams.generators.RandomRBFGenerator.class.getSimpleName())) {
            return new RandomRBFGenerator();
        }else if(name.equals(moa.streams.generators.RandomRBFGeneratorDrift.class.getSimpleName())) {
            return new RandomRBFGeneratorDrift();
        }else if(name.equals(moa.streams.generators.RandomTreeGenerator.class.getSimpleName())) {
            return new RandomTreeGenerator();
        }else if(name.equals(moa.streams.generators.SEAGenerator.class.getSimpleName())) {
            return new SEAGenerator();
        }else if(name.equals(moa.streams.generators.STAGGERGenerator.class.getSimpleName())) {
            return new STAGGERGenerator();
        }else if(name.equals(moa.streams.generators.WaveformGenerator.class.getSimpleName())) {
            return new WaveformGenerator();
        }else if(name.equals(moa.streams.generators.WaveformGeneratorDrift.class.getSimpleName())) {
            return new WaveformGeneratorDrift();
        }else if(name.equalsIgnoreCase("Electricity")) {
            String path = classLoader.getResource("datasets/elecNormNew.arff").getPath();
            return new ArffFileStream(path, -1);
        }else if(name.equalsIgnoreCase("PokerHand")) {
            String path = classLoader.getResource("datasets/poker-lsn.arff").getPath();
            return new ArffFileStream(path, -1);
        }else if(name.equalsIgnoreCase("Airlines")) {
            String path = classLoader.getResource("datasets/airlines.arff").getPath();
            return new ArffFileStream(path, -1);
        }else if(name.equalsIgnoreCase("ForestCovertype")) {
            String path = classLoader.getResource("datasets/covtypeNorm.arff").getPath();
            return new ArffFileStream(path, -1);
        }else{
            System.out.println("Stream generator not found. Returning an instance of RandomRBFGenerator");
            return new RandomRBFGenerator();
        }
    }
}
