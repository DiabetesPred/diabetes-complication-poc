import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyhealth.datasets import MIMIC3Dataset
import logging
from collections import Counter
from tqdm import tqdm
from multiprocessing import freeze_support
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ICD-9 code descriptions (common codes)
ICD9_DESCRIPTIONS = {
    '250': 'Diabetes mellitus',
    '401': 'Essential hypertension',
    '410': 'Acute myocardial infarction',
    '414': 'Other forms of chronic ischemic heart disease',
    '428': 'Heart failure',
    '486': 'Pneumonia, organism unspecified',
    '491': 'Chronic bronchitis',
    '496': 'Chronic airway obstruction',
    '518': 'Other diseases of lung',
    '599': 'Other disorders of urethra and urinary tract',
    # Add more codes as needed
}

def get_disease_name(code):
    """Get disease name from ICD-9 code, checking for parent codes if exact match not found."""
    if code in ICD9_DESCRIPTIONS:
        return ICD9_DESCRIPTIONS[code]
    # Check if it's a subcategory of a known code
    parent_code = code[:3]
    if parent_code in ICD9_DESCRIPTIONS:
        return f"{ICD9_DESCRIPTIONS[parent_code]} (subtype {code})"
    return "Unknown diagnosis"

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

        # Count diagnoses per patient
        logger.info("Analyzing diagnosis patterns...")
        diagnosis_counts = Counter()  # Total occurrences
        patient_diagnosis_counts = Counter()  # Unique patients per diagnosis
        patients_per_diagnosis = {}  # Keep track of patients for each diagnosis

        # Process each patient
        for patient_id, patient in tqdm(dataset.patients.items(), desc="Processing patients"):
            patient_diagnoses = set()  # Track unique diagnoses for this patient
            
            # Collect diagnoses from all visits
            for visit in patient.visits.values():
                if 'DIAGNOSES_ICD' in visit.event_list_dict:
                    for event in visit.event_list_dict['DIAGNOSES_ICD']:
                        code = str(event.code).strip()
                        diagnosis_counts[code] += 1  # Count total occurrences
                        patient_diagnoses.add(code)  # Add to patient's unique diagnoses
                        
                        # Track patients per diagnosis
                        if code not in patients_per_diagnosis:
                            patients_per_diagnosis[code] = set()
                        patients_per_diagnosis[code].add(patient_id)
            
            # Count unique diagnoses per patient
            for code in patient_diagnoses:
                patient_diagnosis_counts[code] += 1

        # Get top 10 most common diagnoses
        logger.info("\nTop 10 Most Common Diagnoses:")
        logger.info("-" * 100)
        logger.info(f"{'ICD Code':<10} {'Disease Name':<40} {'Total Occurrences':<20} {'Unique Patients':<20}")
        logger.info("-" * 100)
        
        for code, count in diagnosis_counts.most_common(10):
            unique_patients = len(patients_per_diagnosis[code])
            disease_name = get_disease_name(code)
            logger.info(f"{code:<10} {disease_name:<40} {count:<20} {unique_patients:<20}")

        # Visualize top 10 diagnoses
        plt.figure(figsize=(15, 7))
        top_10_codes = [code for code, _ in diagnosis_counts.most_common(10)]
        top_10_names = [f"{code}\n{get_disease_name(code)[:20]}" for code in top_10_codes]
        top_10_total = [diagnosis_counts[code] for code in top_10_codes]
        top_10_unique = [len(patients_per_diagnosis[code]) for code in top_10_codes]

        x = np.arange(len(top_10_codes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(15, 7))
        rects1 = ax.bar(x - width/2, top_10_total, width, label='Total Occurrences')
        rects2 = ax.bar(x + width/2, top_10_unique, width, label='Unique Patients')

        ax.set_ylabel('Count')
        ax.set_title('Top 10 Most Common Diagnoses')
        ax.set_xticks(x)
        ax.set_xticklabels(top_10_names, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.savefig('top_10_diagnoses.png')
        logger.info("Visualization saved as 'top_10_diagnoses.png'")

        # Additional statistics
        logger.info("\nDiagnosis Statistics:")
        logger.info(f"Total unique diagnosis codes: {len(diagnosis_counts)}")
        logger.info(f"Average diagnoses per patient: {sum(diagnosis_counts.values()) / len(dataset.patients):.2f}")
        logger.info(f"Average unique diagnoses per patient: {sum(len(patients_per_diagnosis[code]) for code in diagnosis_counts) / len(dataset.patients):.2f}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    freeze_support()
    main()
        