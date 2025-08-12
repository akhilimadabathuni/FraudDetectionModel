# Java Fraud Detection System with Weka

This project is a command-line application built in Java that uses the Weka machine learning library to train a model for detecting fraudulent transactions from a CSV dataset.

---

## Overview

The application performs a complete, end-to-end machine learning workflow:
1.  **Loads** transaction data from a user-provided CSV file.
2.  **Preprocesses** the data by removing unhelpful columns and converting data types into a format suitable for machine learning.
3.  **Trains** a J48 decision tree classifier on the prepared data.
4.  **Evaluates** the model's performance using 10-fold cross-validation and prints the results, including accuracy and a confusion matrix.
5.  **Saves** the trained model to a file named `fraud_detector.model`.
6.  **Loads** the saved model and uses it to make predictions on new, sample transactions.

---

## Prerequisites

Before running this project, you will need:
* **Java Development Kit (JDK)** - Version 9 or newer.
* **Weka JAR File** - The Weka library file (e.g., `weka-3-8-6.jar`).
* **A CSV Dataset** - A dataset of transactions where the last column is the label for fraud/legitimacy.

---

## How to Run

1.  **Place Files:** Ensure `FraudDetectionDemo.java`, your dataset (e.g., `bank_data.csv`), and the Weka JAR file are in the same project directory.

2.  **Compile:** Open a terminal or command prompt in your project directory and run the compile command.
    ```bash
    # On Windows/Linux/macOS
    javac -cp ".;path/to/weka.jar" FraudDetectionDemo.java
    ```
    *(Remember to replace `path/to/weka.jar` with the actual path, e.g., `Weka-3-8-6/weka-3-8-6.jar`)*

3.  **Execute:** Run the program from the terminal, providing your dataset's filename as an argument. The `--add-opens` flag is required for modern versions of Java.
    ```bash
    # On Windows/Linux/macOS
    java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;path/to/weka.jar" FraudDetectionDemo your_dataset.csv
    ```
    **Example:**
    ```bash
    java --add-opens java.base/java.lang=ALL-UNNAMED -cp ".;Weka-3-8-6/weka-3-8-6.jar" FraudDetectionDemo bank_data.csv
    ```

---

## Dataset Format

The program expects a CSV file with a header row. The **last column** in the CSV must be the **class attribute** (the label you want to predict), containing `0` for legitimate transactions and `1` for fraudulent ones.

The code is currently configured to work with the structure of the "BankSim" dataset from Kaggle. It automatically removes the `customer`, `zipcodeOri`, `merchant`, and `zipMerchant` columns as part of its preprocessing routine.

---

## How It Works

The core of the project is the data preprocessing pipeline, which applies a series of Weka **filters** to make the raw data compatible with the J48 algorithm:

1.  **`Remove` Filter:** Deletes identifier columns that do not help the model generalize.
2.  **`StringToNominal` Filter:** Converts all text-based attributes (like `category` or `gender`) into a fixed set of categories.
3.  **`NumericToNominal` Filter:** Converts the numeric `fraud` column (with values 0 and 1) into a categorical attribute that the classifier can predict.

This processed data is then used to build and evaluate the decision tree model.
