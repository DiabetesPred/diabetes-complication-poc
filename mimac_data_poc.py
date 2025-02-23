import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
import logging
import time
from tqdm import tqdm
from multiprocessing import freeze_support

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting data processing pipeline...")

    try:
        # Load dataset
        logger.info("Loading MIMIC3 dataset...")
        start_time = time.time()
        dataset = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
            tables=["DIAGNOSES_ICD"],
            code_mapping={"ICD9CM": "CCSCM"},
            dev=True
        )
        logger.info(f"Dataset loaded successfully in {time.time() - start_time:.2f} seconds")
        dataset.stat()
        dataset.info()

        # Save DIAGNOSES_ICD data to CSV
        logger.info("Saving DIAGNOSES_ICD data to CSV...")
        diagnoses_data = []
        
        for patient_id, patient in dataset.patients.items():
            for visit_id, visit in patient.visits.items():
                if 'DIAGNOSES_ICD' in visit.event_list_dict:
                    for event in visit.event_list_dict['DIAGNOSES_ICD']:
                        diagnoses_data.append({
                            'patient_id': patient_id,
                            'visit_id': visit_id,
                            'icd_code': event.code,
                            'icd_version': event.icd_version if hasattr(event, 'icd_version') else None,
                            'seq_num': event.seq_num if hasattr(event, 'seq_num') else None
                        })
        
        diagnoses_df = pd.DataFrame(diagnoses_data)
        csv_path = 'diagnoses_icd_data.csv'
        diagnoses_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(diagnoses_df)} diagnosis records to {csv_path}")
        logger.info(f"Sample of saved data:\n{diagnoses_df.head()}\n")

        # Extract relevant patient data
        logger.info("Extracting patient data and creating labels...")
        data = []
        labels = []
        
        # Debug counters
        total_diagnoses = 0
        diabetes_codes_found = []
        
        # Use dataset.patients instead of len(dataset)
        for i, (patient_id, patient) in enumerate(tqdm(dataset.patients.items(), desc="Processing patients")):
            diagnoses = []
            patient_diagnoses = set()  # Use set to avoid duplicates
            
            # Collect diagnoses from all visits
            for visit in patient.visits.values():
                if 'DIAGNOSES_ICD' in visit.event_list_dict:
                    visit_diagnoses = [str(event.code).strip() for event in visit.event_list_dict['DIAGNOSES_ICD']]
                    total_diagnoses += len(visit_diagnoses)
                    patient_diagnoses.update(visit_diagnoses)
            
            # Debug: Print diagnoses for first few patients
            if i < 5:
                logger.debug(f"Patient {patient_id} diagnoses: {patient_diagnoses}")
            
            # Check for diabetes codes
            has_diabetes = False
            for code in patient_diagnoses:
                if str(code).startswith('250'):
                    has_diabetes = True
                    diabetes_codes_found.append(code)
                    break
            
            # Check for complications
            has_cvd = any(str(code).startswith('410') for code in patient_diagnoses)
            has_hypertension = any(str(code).startswith('401') for code in patient_diagnoses)
            
            if has_diabetes:
                data.append(list(patient_diagnoses))
                labels.append([has_cvd, has_hypertension])
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Processed {i + 1} patients")

        # Log diagnostic information
        logger.info(f"Total diagnoses processed: {total_diagnoses}")
        logger.info(f"Unique diabetes codes found: {set(diabetes_codes_found)}")
        logger.info(f"Total diabetes codes found: {len(diabetes_codes_found)}")
        logger.info(f"Found {len(data)} patients with diabetes out of {len(dataset.patients)} total patients")

        # Convert to DataFrame
        logger.info("Converting data to DataFrame format...")
        df = pd.DataFrame(data)
        df_labels = pd.DataFrame(labels, columns=["CVD", "Hypertension"])

        # Check if we have enough data
        if len(data) < 10:
            logger.error(f"Insufficient data: only {len(data)} patients with diabetes found. Need at least 10 for meaningful analysis.")
            return

        # One-hot encode diagnoses and get unique codes
        logger.info("Performing one-hot encoding...")
        
        # Get all unique diagnosis codes
        all_codes = set()
        for diagnoses_list in data:
            all_codes.update(diagnoses_list)
        
        # Create feature matrix
        logger.info(f"Creating feature matrix with {len(all_codes)} unique diagnosis codes...")
        X = np.zeros((len(data), len(all_codes)))
        code_to_index = {code: i for i, code in enumerate(all_codes)}
        
        for i, diagnoses_list in enumerate(data):
            for code in diagnoses_list:
                if code in code_to_index:
                    X[i, code_to_index[code]] = 1

        # Convert labels to array
        y = df_labels.values

        # Train-test split
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

        # Define models for multi-label classification
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        }

        # Train and evaluate models
        logger.info("Starting model training and evaluation...")
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            start_time = time.time()
            
            if name == "Neural Network":
                model.verbose = True
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            
            # Calculate metrics for multi-label classification
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # For ROC AUC, we need to handle multi-label case
            try:
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
                    # Average ROC AUC across all classes
                    roc_auc = np.mean([roc_auc_score(y_test[:, i], y_pred_proba[:, i]) 
                                     for i in range(y_test.shape[1])])
                else:
                    roc_auc = 0.0
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC for {name}: {str(e)}")
                roc_auc = 0.0
            
            results[name] = [accuracy, f1, roc_auc]
            
            logger.info(f"{name} Results:")
            logger.info(f"Training Time: {training_time:.2f} seconds")
            logger.info(f"Accuracy: {accuracy:.3f}")
            logger.info(f"F1 Score: {f1:.3f}")
            logger.info(f"ROC AUC: {roc_auc:.3f}")
            logger.info("-" * 50)

            # Log confusion matrix for each condition
            for i, condition in enumerate(["CVD", "Hypertension"]):
                cm = confusion_matrix(y_test[:, i], y_pred[:, i])
                logger.info(f"Confusion Matrix for {condition}:")
                logger.info(f"\n{cm}")

        # Visualization
        logger.info("Creating visualization...")
        metrics = ["Accuracy", "F1 Score", "ROC AUC"]
        results_df = pd.DataFrame(results, index=metrics)

        plt.figure(figsize=(10, 5))
        sns.heatmap(results_df, annot=True, cmap="coolwarm", fmt=".3f")
        plt.title("Model Performance Comparison")
        plt.show()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    freeze_support()
    main()
