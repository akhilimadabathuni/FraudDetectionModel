// --- Weka Core & Classifier Imports ---
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.CSVLoader;
// --- Weka Filter Imports ---
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;

// --- Standard Java I/O and Utility Imports ---
import java.io.File;
import java.util.Random;

/**
 * A complete, self-contained example of a fraud detection system using Java and the Weka machine learning library.
 * This version demonstrates a more realistic workflow by separating model training/saving from model loading/prediction.
 *
 * HOW TO RUN:
 * 1. Create a CSV file (e.g., "bank_data.csv") with your data. The last column must be the class/label.
 * 2. Place this Java file, your CSV, and the Weka JAR in the same directory.
 * 3. Compile the code:
 * (Windows) javac -cp ".;path/to/weka.jar" FraudDetectionDemo.java
 * 4. Run the code, providing your CSV filename as an argument:
 * (Windows) java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;path/to/weka.jar" FraudDetectionDemo bank_data.csv
 *
 * This will create a file named 'fraud_detector.model' in the same directory.
 */
public class FraudDetectionDemo {

    // --- Class Constants and Variables ---
    private static final String MODEL_FILENAME = "fraud_detector.model";
    private static File trainingDataFile;

    /**
     * The main entry point of the program.
     * It now accepts a CSV filename as a command-line argument.
     * @param args Command line arguments. Expects the path to the CSV file at args[0].
     * @throws Exception Can throw various exceptions from file I/O or Weka operations.
     */
    public static void main(String[] args) throws Exception {
        // Check if the user provided a filename.
        if (args.length == 0) {
            System.err.println("Please provide the path to your CSV dataset as an argument.");
            System.err.println("Usage: java FraudDetectionDemo <path-to-your-data.csv>");
            System.exit(1); // Exit with an error code.
        }
        
        String csvFilePath = args[0];
        trainingDataFile = new File(csvFilePath);

        if (!trainingDataFile.exists()) {
            System.err.println("Error: The file '" + csvFilePath + "' was not found.");
            System.exit(1);
        }

        // Step 1: Train a model using the provided CSV file.
        trainAndSaveModel();

        // Step 2: Load the saved model and use it to make predictions.
        loadAndPredict();
    }

    /**
     * Loads data from the user-provided CSV, preprocesses it, trains a classifier, evaluates it, and saves the model.
     */
    public static void trainAndSaveModel() throws Exception {
        System.out.println("--- Training and Saving Model from " + trainingDataFile.getName() + " ---");

        // --- 1. Load the User-Provided Dataset ---
        CSVLoader loader = new CSVLoader();
        loader.setSource(trainingDataFile);
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // --- 2. Preprocess Data ---
        // First, remove the identifier columns that are not useful for generalization.
        // Columns are 1-based. We remove customer (2), zipcodeOri (5), merchant (6), zipMerchant (7).
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("2,5-7");
        removeFilter.setInputFormat(data);
        Instances removedData = Filter.useFilter(data, removeFilter);

        // Second, convert any remaining string attributes to nominal.
        StringToNominal stringFilter = new StringToNominal();
        stringFilter.setOptions(new String[]{"-R", "first-last"});
        stringFilter.setInputFormat(removedData);
        Instances nominalData = Filter.useFilter(removedData, stringFilter);

        // Third, convert the numeric class attribute to nominal.
        NumericToNominal numFilter = new NumericToNominal();
        numFilter.setOptions(new String[]{"-R", "last"});
        numFilter.setInputFormat(nominalData);
        Instances finalData = Filter.useFilter(nominalData, numFilter);
        finalData.setClassIndex(finalData.numAttributes() - 1);


        // --- 3. Build and Train the Classifier ---
        J48 tree = new J48();
        tree.buildClassifier(finalData);

        System.out.println("=== Trained J48 Decision Tree ===");
        System.out.println(tree);

        // --- 4. Evaluate the Model ---
        Evaluation eval = new Evaluation(finalData);
        eval.crossValidateModel(tree, finalData, 10, new Random(1));
        System.out.println("=== Model Evaluation (10-fold Cross-Validation) ===");
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
        System.out.println(eval.toMatrixString("\nConfusion Matrix\n==============\n"));

        // --- 5. Save the Trained Model ---
        SerializationHelper.write(MODEL_FILENAME, tree);
        System.out.println("Model successfully trained and saved to '" + MODEL_FILENAME + "'");
        System.out.println("\n-----------------------------------\n");
    }

    /**
     * Loads a pre-trained model from a file and uses it to classify new instances.
     */
    public static void loadAndPredict() throws Exception {
        System.out.println("--- Loading Model and Making Predictions ---");

        // --- 1. Load the saved model ---
        Classifier loadedTree = (Classifier) SerializationHelper.read(MODEL_FILENAME);

        // --- 2. Create a dataset structure for new instances ---
        CSVLoader loader = new CSVLoader();
        loader.setSource(trainingDataFile);
        Instances rawStructure = loader.getDataSet();

        // **IMPORTANT**: We must apply the SAME filters in the SAME order to the prediction structure.
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("2,5-7");
        removeFilter.setInputFormat(rawStructure);
        Instances removedStructure = Filter.useFilter(rawStructure, removeFilter);

        StringToNominal stringFilter = new StringToNominal();
        stringFilter.setOptions(new String[]{"-R", "first-last"});
        stringFilter.setInputFormat(removedStructure);
        Instances nominalStructure = Filter.useFilter(removedStructure, stringFilter);

        NumericToNominal numFilter = new NumericToNominal();
        numFilter.setOptions(new String[]{"-R", "last"});
        numFilter.setInputFormat(nominalStructure);
        Instances structure = Filter.useFilter(nominalStructure, numFilter);

        structure.setClassIndex(structure.numAttributes() - 1);
        structure.clear();

        // --- 3. Create and classify new transactions ---
        // The new structure is: step, age, gender, category, amount, fraud
        
        // Example 1: A high-value travel transaction.
        Instance highRiskTx = new DenseInstance(structure.numAttributes());
        highRiskTx.setDataset(structure);
        highRiskTx.setValue(0, 180);                     // step
        highRiskTx.setValue(1, "3");                     // age
        highRiskTx.setValue(2, "M");                     // gender
        highRiskTx.setValue(3, "es_travel");             // category
        highRiskTx.setValue(4, 834.76);                  // amount
        predict(loadedTree, highRiskTx, "High-Risk Travel Transaction");

        // Example 2: A low-value food transaction.
        Instance lowRiskTx = new DenseInstance(structure.numAttributes());
        lowRiskTx.setDataset(structure);
        lowRiskTx.setValue(0, 180);                      // step
        lowRiskTx.setValue(1, "4");                      // age
        lowRiskTx.setValue(2, "M");                      // gender
        lowRiskTx.setValue(3, "es_food");                // category
        lowRiskTx.setValue(4, 22.50);                    // amount
        predict(loadedTree, lowRiskTx, "Low-Risk Food Transaction");
    }

    /**
     * Helper method to classify a single instance and print the result.
     */
    private static void predict(Classifier classifier, Instance transaction, String name) throws Exception {
        double predictionIndex = classifier.classifyInstance(transaction);
        String predictedClass = transaction.classAttribute().value((int) predictionIndex);
        System.out.println("Prediction for '" + name + "': " + (predictedClass.equals("1") ? "FRAUD" : "LEGIT"));
    }
}
