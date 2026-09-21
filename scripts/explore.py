import requests
import pydicom
import json

PORT = 8042
PATH = f'http://localhost:{PORT}'
def explore():
    patients_resp = requests.get(f'{PATH}/patients', auth=('dev', 'dev'))    
    patients_json = patients_resp.json()
    
    first_patient_uid = patients_json[0]
    
    patient = requests.get(f'{PATH}/patients/{first_patient_uid}', auth=('dev', 'dev'))
 #   print(patients_resp.status_code)

 #   print(patient.text)
    
    patient_0 = patient.json()
    study_ids = patient_0['Studies']
    patient_0_study_0 = study_ids[0]

    studies = requests.get(f'{PATH}/studies/{patient_0_study_0}', auth=('dev', 'dev'))
    studies_json = studies.json()
    series_0 = studies_json["Series"][0]
    series = requests.get(f'{PATH}/series/{series_0}', auth=('dev', 'dev')) 
    series_json = series.json()
    
    instance_0_uid = series_json["Instances"][0]

    instance_0 = requests.get(f'{PATH}/instances/{instance_0_uid}', auth=('dev', 'dev'))
    print(instance_0.text)

if __name__ == "__main__":
    explore()
